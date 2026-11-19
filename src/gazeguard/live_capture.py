"""Low-latency webcam reader.

A dedicated reader continually replaces its single-frame buffer; inference never
works through a queue of stale webcam frames.  OpenCV VideoCapture is only read
from the reader thread and released after that thread exits.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Captured:
    number: int
    when: float
    frame: object


class LatestFrameCamera:
    def __init__(self, capture, *, idle_sleep: float = .003):
        self.capture = capture
        self.idle_sleep = idle_sleep
        self._lock = threading.Condition()
        self._running = False
        self._thread = None
        self._latest: Captured | None = None
        self._number = 0
        self._failures = 0
        self.error: str | None = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader, name='GazeGuardCameraReader', daemon=True)
        self._thread.start()

    def _reader(self):
        try:
            while self._running:
                ok, frame = self.capture.read()
                with self._lock:
                    if not ok:
                        self._failures += 1
                        if self._failures >= 30:
                            self.error = 'Camera stopped providing frames'
                            self._running = False
                        self._lock.notify_all()
                    else:
                        self._failures = 0
                        self._number += 1
                        self._latest = Captured(self._number, time.monotonic(), frame)
                        self._lock.notify_all()
                if not ok:
                    time.sleep(.03)
                elif self.idle_sleep:
                    time.sleep(self.idle_sleep)
        finally:
            with self._lock:
                self._running = False
                self._lock.notify_all()

    def next(self, after: int, timeout: float = .3) -> Captured | None:
        deadline = time.monotonic() + timeout
        with self._lock:
            while self._running and (self._latest is None or self._latest.number <= after):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._lock.wait(remaining)
            if self._latest is not None and self._latest.number > after:
                return self._latest
            return None

    def close(self):
        self._running = False
        with self._lock:
            self._lock.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # The OS camera handle is deliberately released only after reader exits.
        if self._thread is None or not self._thread.is_alive():
            self.capture.release()
        else:
            # Avoid racing a read with release; reader should exit at next read.
            self.error = 'Camera reader did not stop promptly'
