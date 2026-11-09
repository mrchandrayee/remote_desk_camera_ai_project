import threading
import time
import cv2
import numpy as np

class FrameSource:
    """Background OpenCV source reader. source may be webcam index, file, or RTSP URL."""
    def __init__(self, source, width=1280):
        self.source = int(source) if str(source).isdigit() else source
        self.width = width
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.cap = None
        self.last_error = None

    def start(self):
        if self.running:
            return self
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.last_error = f"Cannot open source: {self.source}"
            return self
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while self.running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                if isinstance(self.source, str) and str(self.source).lower().endswith((".mp4",".avi",".mov",".mkv")):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(.05)
                    continue
                self.last_error = "Frame read failed"
                time.sleep(.5)
                continue
            h, w = frame.shape[:2]
            if w > self.width:
                nh = int(h * self.width / w)
                frame = cv2.resize(frame, (self.width, nh))
            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
            self.cap = None

class DesktopSource:
    """Cross-platform desktop capture using MSS (DXGI is used internally on supported Windows setups)."""
    def __init__(self, monitor=1, width=1280):
        self.monitor = monitor
        self.width = width

    def get_frame(self):
        try:
            import mss
            with mss.mss() as sct:
                monitors = sct.monitors
                index = self.monitor if 0 <= self.monitor < len(monitors) else 1
                shot = np.array(sct.grab(monitors[index]))
                frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                h, w = frame.shape[:2]
                if w > self.width:
                    frame = cv2.resize(frame, (self.width, int(h*self.width/w)))
                return frame
        except Exception:
            return None
