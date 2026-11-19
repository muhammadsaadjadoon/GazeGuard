"""Local-only browser dashboard for GazeGuard.

No external web API: web UI and MJPEG camera preview are served exclusively from
127.0.0.1. MediaPipe runs in the local process, not in the user's browser.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

import cv2
import numpy as np

from .calibration import CalibrationWizard, align_lens_reference, load_calibration, save_calibration, teach_screen_reference
from .config import DEFAULT_HOME, load_settings, write_default_settings
from .geometry import face_bounds, measure_face, L_IRIS, R_IRIS, point
from .gestures import hand_points, detect_gesture, hand_overlaps_face, GestureDebouncer
from .monitor import Monitor
from .automatic import AutomaticMonitor
from .trained_gaze import extract_features, load_model
from .reporting import SessionRecorder
from .vision import ensure_models, TaskVision
from .live_capture import LatestFrameCamera

LOG = logging.getLogger('gazeguard')
ASSETS = Path(__file__).parent / 'web'

HAND_EDGES = ((0,1),(1,2),(2,3),(3,4),
              (0,5),(5,6),(6,7),(7,8),
              (5,9),(9,10),(10,11),(11,12),
              (9,13),(13,14),(14,15),(15,16),
              (13,17),(17,18),(18,19),(19,20),
              (0,17),(5,9),(9,13),(13,17))


def draw_hand_skeleton(image, coordinates):
    """Draw connected palm/finger bones and joints, not just 21 dots."""
    if len(coordinates) != 21:
        return image
    import numpy as np
    xy = np.asarray(coordinates, dtype=np.int32)
    for a, b in HAND_EDGES:
        cv2.line(image, tuple(xy[a]), tuple(xy[b]), (56, 210, 175), 3, cv2.LINE_AA)
    for idx, (x, y) in enumerate(xy):
        cv2.circle(image, (int(x), int(y)), 5 if idx in (0, 4, 8, 12, 16, 20) else 3,
                   (241, 244, 248), -1, cv2.LINE_AA)
        cv2.circle(image, (int(x), int(y)), 5 if idx in (0, 4, 8, 12, 16, 20) else 3,
                   (56, 210, 175), 1, cv2.LINE_AA)
    return image



class Engine:
    def __init__(self, data_dir: Path, camera_index: int | None = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.data_dir / 'settings.json'
        write_default_settings(config_path)
        self.settings = load_settings(config_path)
        self.camera_index = self.settings.camera_index if camera_index is None else camera_index
        self.cal_file = self.data_dir / f'calibration_camera_{self.camera_index}.json'
        profile = load_calibration(self.cal_file)
        # Public launch uses no stored user calibration by default. Old saved
        # profiles remain available only if Advanced calibration is requested.
        self.wizard = None
        self.manual_mode = False
        self.monitor = AutomaticMonitor()
        self.trained_model = None
        model_path = self.data_dir / 'trained_gaze.json'
        # Explicit opt-in only; a newly trained model must be externally evaluated.
        if os.getenv('GAZEGUARD_ENABLE_TRAINED_GAZE') == '1' and model_path.is_file():
            self.trained_model = load_model(model_path)
            LOG.info('Enabled user-trained gaze model: %s', model_path)
        self.saved_profile = profile
        self.lens_frames = deque(maxlen=48)
        self.screen_frames = deque(maxlen=44)
        self.screen_capture_active = False
        self.camera_on = False
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        self._frame = None
        self._frame_id = 0
        self.recorder = SessionRecorder(time.monotonic())
        self.gestures = GestureDebouncer(self.settings.gesture_frames, self.settings.gesture_cooldown_seconds)
        self.state = dict(status='CONNECTING', reason='Preparing automatic camera monitoring', eyes='UNKNOWN',
                          head='UNKNOWN', focus='—', hands=0, gesture='—', fps=0,
                          face=False, session='00:00', attention=None, calibration='AUTOMATIC',
                          step='', progress=100, step_ready=False, screen_ready=False, mode='AUTOMATIC',
                          camera=False, message='', latency_ms=None, frame_age_ms=None,
                          processing_ms=None, hand_visible=False)

    def start_camera(self):
        with self._lock:
            if self.camera_on:
                return 'Camera already running'
            self._stop.clear()
            self.state.update(status='LOADING', reason='Connecting to camera…', message='')
            self._thread = threading.Thread(target=self._capture, name='GazeGuardCapture', daemon=True)
            self._thread.start()
            return 'Starting camera'

    def stop_camera(self):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._lock:
            self.camera_on = False
            self.state.update(status='INACTIVE', reason='Camera stopped', camera=False, fps=0, face=False, latency_ms=None, frame_age_ms=None, processing_ms=None, hands=0, hand_visible=False)
            self._frame = None
            self._frame_id += 1
        return 'Camera stopped'

    def _capture(self):
        camera = None
        reader = None
        vision = None
        try:
            with self._lock:
                self.state.update(status='LOADING', reason='Preparing local tracking models…')
            models = ensure_models(self.data_dir)
            vision = TaskVision(models)
            vision.__enter__()
            import sys
            backend = cv2.CAP_DSHOW if sys.platform == 'win32' else cv2.CAP_ANY
            camera = cv2.VideoCapture(self.camera_index, backend)
            if not camera.isOpened():
                raise RuntimeError('Camera unavailable. Close Zoom/Teams or check Windows camera privacy settings.')
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.camera_width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.camera_height)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            reader = LatestFrameCamera(camera)
            reader.start()
            last_number = 0
            with self._lock:
                self.camera_on = True
                self.recorder = SessionRecorder(time.monotonic())
            last = time.monotonic()
            fps = 0.0
            while not self._stop.is_set():
                captured = reader.next(last_number, timeout=.30)
                if captured is None:
                    if reader.error:
                        raise RuntimeError(reader.error)
                    continue
                last_number = captured.number
                now = time.monotonic()
                fps = .83 * fps + .17 / max(now-last, .0001) if fps else 1 / max(now-last, .0001)
                last = now
                frame = captured.frame
                if self.settings.mirror:
                    frame = cv2.flip(frame, 1)
                # Limit inference work while keeping camera preview readable.
                # Coordinates remain correct because landmarks are measured on this frame.
                height, width = frame.shape[:2]
                if width > 960:
                    frame = cv2.resize(frame, (960, max(1, round(height * 960 / width))),
                                       interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
                processing_start = time.monotonic()
                faces, hands = vision.process(frame)
                face = faces[0] if faces else None
                try:
                    metrics = measure_face(face, w, h) if face else None
                except (ValueError, IndexError, FloatingPointError):
                    metrics = None
                    face = None
                hand_coords = [hand_points(item, w, h) for item in hands]
                gesture = detect_gesture(hand_coords)
                event = self.gestures.update(gesture, now)
                with self._lock:
                    if self.wizard:
                        if metrics is not None:
                            self.wizard.observe(metrics.vector, metrics.home,
                                                (metrics.eye_left, metrics.eye_right))
                        else:
                            self.wizard.clear_recent()
                        step = self.wizard.instruction
                        progress = len(self.wizard.recent_eyes)/self.wizard.sample_count * 100
                        self.state.update(status='CALIBRATING', reason='Look at instructed point, then confirm.',
                                          calibration=f'STEP {self.wizard.index + 1} / 5',
                                          step=step, progress=round(progress), step_ready=progress>=100,
                                          eyes='—', head='—', focus='SETUP', attention=None)
                    elif self.monitor:
                        obstructed = bool(face is not None and any(
                            hand_overlaps_face(hand, face_bounds(face, w, h)) for hand in hand_coords))
                        obs = self.monitor.update(metrics, now, obstructed)
                        # Optional model is a conservative SCREEN/OTHER veto only:
                        # it cannot force ACTIVE on top of a rejected baseline frame.
                        if (self.trained_model is not None and not self.manual_mode
                                and obs.attentive and metrics is not None):
                            from dataclasses import replace
                            try:
                                feature_vector = extract_features(metrics)
                                model_score = self.trained_model.probability(feature_vector)
                                if model_score < self.trained_model.threshold:
                                    obs = replace(obs, status='LOOKING_AWAY',
                                                  reason='Trained webcam model classifies gaze as non-lens')
                            except ValueError:
                                obs = replace(obs, status='LOOKING_AWAY',
                                              reason='Trained model could not verify eye landmarks')
                        if metrics is not None and metrics.eye_left is not None and metrics.eye_right is not None:
                            self.lens_frames.append((now, metrics.vector.copy(), metrics.home.copy(),
                                                     [metrics.eye_left, metrics.eye_right]))
                            if self.screen_capture_active:
                                self.screen_frames.append(metrics.vector.copy())
                                if len(self.screen_frames) >= 30:
                                    self._finish_screen_locked()
                        else:
                            self.lens_frames.clear()
                        self.recorder.record(now, obs.status, event, obs.speaking_likely and not obs.attentive)
                        pct = self.recorder.summary(now)['attentive_percent_observable']
                        screen_ready = bool(self.manual_mode and getattr(self.monitor, 'calibration', None) is not None and self.monitor.calibration.screen_reference is not None)
                        if obs.attentive:
                            focus = 'CAMERA LENS (CALIBRATED)' if self.manual_mode else 'CAMERA AXIS (ESTIMATE)'
                        elif obs.status == 'HANDS_OBSTRUCTING':
                            focus = 'OBSTRUCTED'
                        elif self.manual_mode and screen_ready and 'Screen gaze detected' in (obs.reason or ''):
                            focus = 'SCREEN'
                        elif obs.eyes == 'CENTER' and obs.head == 'CENTER':
                            focus = 'CAMERA AREA'
                        else:
                            focus = 'AWAY'
                        self.state.update(status=obs.status, reason=obs.reason, eyes=obs.eyes, head=obs.head,
                                          focus=focus, attention=(pct if self.manual_mode else None), step='', progress=100,
                                          calibration=('CALIBRATED' if screen_ready else 'ADVANCED SETUP' if self.manual_mode else 'AUTOMATIC'),
                                          screen_ready=screen_ready, step_ready=False, mode='MANUAL' if self.manual_mode else 'AUTOMATIC')
                    self.state.update(camera=True, face=metrics is not None,
                                      hands=len(hand_coords), gesture=gesture or '—', fps=round(fps),
                                      session=self._clock(now-self.recorder.started))
                if face is not None:
                    x1, y1, x2, y2 = face_bounds(face, w, h)
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(92,214,155),2)
                    for index in (L_IRIS, R_IRIS):
                        x,y = point(face,index,w,h).astype(int)
                        cv2.circle(frame,(int(x),int(y)),3,(255,216,107),-1)
                for hand in hand_coords:
                    draw_hand_skeleton(frame, hand)

                # Encode JPEG once per frame. Do not keep historical frames.
                img_w = min(w, 960)
                if img_w < w:
                    frame = cv2.resize(frame,(img_w,round(h*img_w/w)))
                success, encoded = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,78])
                if success:
                    with self._lock:
                        self._frame = encoded.tobytes()
                        self._frame_id += 1
                        self.state.update(
                            processing_ms=round((time.monotonic()-processing_start)*1000),
                            frame_age_ms=round((time.monotonic()-captured.when)*1000),
                            latency_ms=round((time.monotonic()-captured.when)*1000),
                            hand_visible=bool(hand_coords),
                        )
        except Exception as exc:
            LOG.exception('Camera capture failed')
            with self._lock:
                self.state.update(status='ERROR', reason=str(exc), camera=False, message=str(exc))
        finally:
            if reader is not None:
                reader.close()
            elif camera is not None:
                camera.release()
            if vision is not None:
                vision.__exit__(None,None,None)
            with self._lock:
                self.camera_on = False
                self.state['camera'] = False

    @staticmethod
    def _clock(seconds):
        secs = int(max(seconds,0))
        return f'{secs//60:02d}:{secs%60:02d}'

    def confirm(self):
        with self._lock:
            if not self.wizard:
                return 'Calibration already complete'
            if len(self.wizard.recent_eyes) < self.wizard.sample_count:
                raise ValueError('Wait until calibration progress reaches 100%')
            result = self.wizard.confirm()
            if result is not None:
                save_calibration(self.cal_file, result)
                self.monitor = Monitor(self.settings,result)
                self.wizard = None
                self.lens_frames.clear()
                self.state['message'] = 'Five-point calibration saved. Verify lens, then teach screen.'
            else:
                self.state['message'] = 'Step saved. Follow the next calibration direction.'
            return self.state['message']

    def reset(self):
        with self._lock:
            self.manual_mode = True
            self.wizard = CalibrationWizard(sample_count=24)
            self.monitor = None
            self.screen_capture_active = False
            self.screen_frames.clear()
            self.lens_frames.clear()
            self.recorder = SessionRecorder(time.monotonic())
            self.state['mode'] = 'MANUAL'
            return 'Advanced five-point calibration started'

    def automatic(self):
        with self._lock:
            self.manual_mode = False
            self.wizard = None
            self.monitor = AutomaticMonitor()
            self.screen_capture_active = False
            self.screen_frames.clear()
            self.lens_frames.clear()
            self.state.update(mode='AUTOMATIC', calibration='AUTOMATIC', progress=100,
                              step='', step_ready=False, screen_ready=False, attention=None)
            return 'Automatic monitoring enabled; no user calibration required'

    def verify(self):
        with self._lock:
            if not self.manual_mode or not self.monitor or self.wizard:
                raise ValueError('Advanced calibration is not enabled')
            frames = list(self.lens_frames)
            if len(frames) < 24 or frames[-1][0] - frames[0][0] > 5:
                raise ValueError('Look at physical camera lens for ~2 seconds, then try again')
            corrected = align_lens_reference(self.monitor.calibration,
                                            [f[1] for f in frames], [f[2] for f in frames], [f[3] for f in frames])
            save_calibration(self.cal_file, corrected)
            self.monitor = Monitor(self.settings,corrected)
            self.lens_frames.clear()
            return 'Camera lens reference verified'

    def teach(self):
        with self._lock:
            if not self.manual_mode or not self.monitor or self.wizard:
                raise ValueError('Advanced calibration is not enabled')
            self.screen_frames.clear()
            self.screen_capture_active = True
            return 'Now look at the CENTER OF YOUR SCREEN, not the camera lens (30 frames)'

    def _finish_screen_locked(self):
        self.screen_capture_active = False
        try:
            fitted = teach_screen_reference(self.monitor.calibration,list(self.screen_frames))
            save_calibration(self.cal_file,fitted)
            self.monitor = Monitor(self.settings,fitted)
            self.state['message'] = 'Screen reference saved — camera-lens-only mode enabled'
        except ValueError as exc:
            self.state['message'] = str(exc)
        finally:
            self.screen_frames.clear()

    def export(self):
        with self._lock:
            files = self.recorder.export(self.data_dir/'sessions',time.monotonic())
            return f'Exported: {files[0].name}'

    def snapshot(self):
        with self._lock:
            return dict(self.state)

    def latest_frame(self):
        with self._lock:
            return self._frame_id, self._frame


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None
    def log_message(self, fmt, *args):
        # Do not log user-facing query data; only local requests.
        LOG.debug(fmt,*args)

    def _send(self, code, data, content_type='application/json; charset=utf-8', extra_headers=None):
        self.send_response(code)
        self.send_header('Content-Type',content_type)
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        for name, value in (extra_headers or {}).items():
            self.send_header(name, str(value))
        self.end_headers()
        try: self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError): pass

    def do_GET(self):
        target = urlsplit(self.path).path
        if target in ('/','/index.html'):
            return self._send(200,(ASSETS/'index.html').read_bytes(),'text/html; charset=utf-8')
        if target == '/style.css':
            return self._send(200,(ASSETS/'style.css').read_bytes(),'text/css; charset=utf-8')
        if target == '/app.js':
            return self._send(200,(ASSETS/'app.js').read_bytes(),'application/javascript; charset=utf-8')
        if target == '/api/status':
            return self._send(200,json.dumps(self.engine.snapshot()).encode())
        if target == '/api/frame':
            identifier, frame = self.engine.latest_frame()
            try:
                after = int(parse_qs(urlsplit(self.path).query).get('after', ['-1'])[0])
            except (ValueError, TypeError):
                after = -1
            if frame is None or identifier <= after:
                return self._send(204,b'')
            return self._send(200,frame,'image/jpeg',{'X-Frame-Id': identifier})
        self._send(404,b'Not found','text/plain')

    def do_POST(self):
        commands = {'start':self.engine.start_camera,'stop':self.engine.stop_camera,
                    'confirm':self.engine.confirm,'reset':self.engine.reset,
                    'verify':self.engine.verify,'teach':self.engine.teach,'export':self.engine.export,
                    'automatic':self.engine.automatic}
        command = urlsplit(self.path).path.removeprefix('/api/')
        if command not in commands:
            return self._send(404,b'{"error":"Not found"}')
        if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
            return self._send(403,b'{"error":"Cross-origin request blocked"}')
        try:
            message = commands[command]()
            return self._send(200,json.dumps({'ok':True,'message':message}).encode())
        except ValueError as exc:
            return self._send(400,json.dumps({'ok':False,'error':str(exc)}).encode())
        except Exception:
            LOG.exception('API action failed')
            return self._send(500,b'{"ok":false,"error":"Action failed, see terminal"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='GazeGuard dark local web dashboard')
    parser.add_argument('--camera',type=int,default=None)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_HOME)
    parser.add_argument('--port',type=int,default=0,help='Local web port; 0 chooses unused port')
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--no-auto-camera',action='store_true',help='Diagnostic mode only: do not automatically open the camera')
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    engine=Engine(args.data_dir.expanduser(),args.camera)
    handler=type('BoundHandler',(Handler,),{'engine':engine})
    with ThreadingHTTPServer(('127.0.0.1',args.port),handler) as server:
        port=server.server_address[1]
        url=f'http://127.0.0.1:{port}/'
        print(f'GazeGuard web dashboard: {url}',flush=True)
        print('Press Ctrl+C in the terminal to stop the local dashboard.',flush=True)
        if not args.no_auto_camera:
            engine.start_camera()
        if not args.no_browser:
            webbrowser.open(url,new=1)
        try:
            server.serve_forever(poll_interval=.3)
        except KeyboardInterrupt:
            pass
        finally:
            engine.stop_camera()
    return 0

if __name__=='__main__':
    raise SystemExit(main())
