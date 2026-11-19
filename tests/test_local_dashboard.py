"""Verify the shipped dark dashboard, root-based launcher, and local HTTP routes."""
from http.server import ThreadingHTTPServer
from pathlib import Path
import json
import threading
import urllib.request

from gazeguard.webapp import Engine, Handler, ASSETS


def test_web_assets_exist_and_dark_ui_is_the_shipped_interface():
    page = (ASSETS / 'index.html').read_text(encoding='utf-8')
    stylesheet = (ASSETS / 'style.css').read_text(encoding='utf-8')
    script = (ASSETS / 'app.js').read_text(encoding='utf-8')
    assert 'Live attention status' in page
    assert 'AUTOMATIC LIVE MONITORING' in page
    assert 'Initializing live camera' in page
    assert 'id="start-center"' not in page
    assert 'Retry camera' in page
    assert 'Verify lens' in page and 'Teach screen' in page
    assert 'Your focus, clearly' not in page
    assert 'Main signals stay visible' not in page
    assert 'overflow-y:auto' in stylesheet
    assert "'/api/status'" in script
    assert 'webapp import main' in (Path(__file__).parents[1] / 'run.py').read_text()


def test_local_server_routes_without_hardware(tmp_path):
    engine = Engine(tmp_path)
    handler = type('TestHandler', (Handler,), {'engine': engine})
    with ThreadingHTTPServer(('127.0.0.1', 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        for route in ('/', '/style.css', '/app.js'):
            with urllib.request.urlopen(base + route, timeout=3) as response:
                assert response.status == 200
                assert response.read()
        with urllib.request.urlopen(base + '/api/status', timeout=3) as response:
            status = json.load(response)
            assert status['camera'] is False
        req = urllib.request.Request(base + '/api/reset', data=b'', method='POST')
        with urllib.request.urlopen(req, timeout=3) as response:
            assert json.load(response)['ok']
        server.shutdown()
        thread.join(timeout=3)


def test_synthetic_camera_stream_updates_browser_frame(tmp_path, monkeypatch):
    """Integration smoke: webcam backend to JPEG preview, no physical webcam needed."""
    import time
    import numpy as np
    import cv2
    import gazeguard.webapp as webapp

    class FakeCapture:
        def __init__(self, *args):
            self.open = True
        def isOpened(self):
            return self.open
        def set(self, *args):
            return True
        def read(self):
            time.sleep(0.025)
            return True, np.zeros((240, 320, 3), dtype=np.uint8)
        def release(self):
            self.open = False

    class FakeVision:
        def __init__(self, models):
            pass
        def __enter__(self):
            return self
        def process(self, frame):
            return [], []
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(cv2, 'VideoCapture', FakeCapture)
    monkeypatch.setattr(webapp, 'ensure_models', lambda path: {})
    monkeypatch.setattr(webapp, 'TaskVision', FakeVision)
    engine = Engine(tmp_path)
    engine.start_camera()
    try:
        for _ in range(80):
            _, frame = engine.latest_frame()
            if frame:
                break
            time.sleep(.025)
        assert frame is not None and frame[:2] == b'\xff\xd8'
        assert engine.snapshot()['camera']
    finally:
        engine.stop_camera()
    assert not engine.snapshot()['camera']
