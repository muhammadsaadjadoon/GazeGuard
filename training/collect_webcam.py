"""Consent-based local landmark data collection. Saves ONLY numeric features (no images).

Must run while the dashboard is closed, since many webcams allow a single owner.
Participant watches TARGET FIRST, then presses corresponding number key.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
import cv2
from gazeguard.geometry import measure_face
from gazeguard.trained_gaze import FEATURE_NAMES, extract_features
from gazeguard.vision import TaskVision, ensure_models
from gazeguard.config import DEFAULT_HOME

TARGETS={'1':'lens','2':'screen','3':'left','4':'right','5':'up','6':'down'}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Local opt-in camera gaze collection (features only)')
    parser.add_argument('--participant', required=True, help='Non-identifying participant code, e.g. p01')
    parser.add_argument('--out',type=Path,default=Path('datasets/private/landmarks.csv'))
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--seconds-per-target',type=float,default=9)
    parser.add_argument('--consent',action='store_true',help='Participant explicitly agreed to collect landmarks')
    args=parser.parse_args(argv)
    if not args.consent:
        parser.error('Explicit --consent required: inform participants what the landmark file contains')
    if not args.participant.replace('-','').replace('_','').isalnum():
        parser.error('participant must be an anonymous alphanumeric code')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    session=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW if hasattr(cv2,'CAP_DSHOW') and __import__('sys').platform=='win32' else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError('Cannot open camera. Close the dashboard and check Windows camera permissions.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)
    exists=args.out.exists() and args.out.stat().st_size>0
    fields=['participant_id','session_id','label','timestamp_utc',*FEATURE_NAMES]
    selected=None
    until=0.0
    saved={x:0 for x in TARGETS.values()}
    last_saved=0.
    try:
        with args.out.open('a',newline='',encoding='utf-8') as output, TaskVision(ensure_models(DEFAULT_HOME)) as vision:
            writer=csv.DictWriter(output,fieldnames=fields)
            if not exists: writer.writeheader()
            while True:
                ok, frame=cap.read()
                if not ok: raise RuntimeError('Camera disconnected')
                frame=cv2.flip(frame,1)
                h,w=frame.shape[:2]
                now=time.monotonic()
                metrics=None
                try:
                    faces,_=vision.process(frame)
                    if faces: metrics=measure_face(faces[0],w,h)
                except (ValueError,IndexError,FloatingPointError):
                    pass
                if selected and now < until and now-last_saved >= .12 and metrics is not None:
                    try:
                        features=extract_features(metrics)
                    except ValueError:
                        pass
                    else:
                        row=dict(participant_id=args.participant,session_id=session,label=selected,
                                 timestamp_utc=datetime.now(timezone.utc).isoformat())
                        row.update(dict(zip(FEATURE_NAMES,features)))
                        writer.writerow(row)
                        saved[selected]+=1
                        last_saved=now
                if selected and now>=until:
                    selected=None
                    output.flush()
                display=frame.copy()
                cv2.rectangle(display,(0,0),(display.shape[1],115),(15,26,43),-1)
                cv2.putText(display,'1 Lens | 2 Screen | 3 Left | 4 Right | 5 Up | 6 Down | Q Quit',
                            (14,32),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(display,f'Target: {selected or "PRESS KEY"}  Remaining: {max(0,until-now):.1f}s  Face: {bool(metrics)}',
                            (14,64),cv2.FONT_HERSHEY_SIMPLEX,.60,(94,228,182),2,cv2.LINE_AA)
                cv2.putText(display,'Features saved: '+', '.join(f'{k} {v}' for k,v in saved.items()),
                            (14,93),cv2.FONT_HERSHEY_SIMPLEX,.49,(228,231,241),1,cv2.LINE_AA)
                cv2.imshow('GazeGuard | Opt-in landmark collection (no images saved)',display)
                key=cv2.waitKey(1)&0xFF
                if key in (ord('q'),27): break
                chosen=TARGETS.get(chr(key)) if key < 128 else None
                if chosen:
                    selected=chosen
                    until=now+args.seconds_per_target
                    last_saved=0.
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print('Saved numeric landmark rows to:',args.out,'\nSession:',session,'\nCounts:',saved)
    print('Repeat collection in a NEW session and preferably with other consenting users.')


if __name__=='__main__':
    main()
