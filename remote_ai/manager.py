import threading, time
import cv2
from .capture import FrameSource, DesktopSource
from .analytics import MotionDetector

class SourceManager:
    def __init__(self, cfg):
        self.cfg = cfg
        app = cfg.get("app", {})
        self.width = app.get("frame_width",1280)
        self.quality = app.get("jpeg_quality",80)
        self.threshold = app.get("motion_threshold",22)
        self.min_area = app.get("motion_min_area",1200)
        self.sources = {}
        self.detectors = {}
        self.meta = {}
        self.lock = threading.Lock()
        self._build()

    def _build(self):
        ds = self.cfg.get("sources",{}).get("desktop",{})
        if ds.get("enabled",True):
            self.sources["desktop"] = ("This Desktop", DesktopSource(ds.get("monitor",1),self.width))
        for item in self.cfg.get("sources",{}).get("cameras",[]):
            if item.get("enabled",False):
                sid = item["id"]
                self.sources[sid] = (item.get("name",sid), FrameSource(item["source"],self.width).start())
        for sid in self.sources:
            self.detectors[sid] = MotionDetector(self.threshold,self.min_area)
            self.meta[sid] = {"motion":False,"motion_area":0,"updated":None}

    def ids(self):
        return list(self.sources.keys())

    def name(self, sid):
        return self.sources[sid][0]

    def get_jpeg(self, sid):
        if sid not in self.sources:
            return None
        _, source = self.sources[sid]
        frame = source.get_frame()
        if frame is None:
            return None
        annotated, motion, area = self.detectors[sid].apply(frame)
        with self.lock:
            self.meta[sid] = {"motion":motion,"motion_area":area,"updated":time.time()}
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY,self.quality])
        return buf.tobytes() if ok else None

    def status(self):
        out=[]
        for sid,(name,source) in self.sources.items():
            frame = source.get_frame() if hasattr(source,"get_frame") else None
            with self.lock:
                m = dict(self.meta[sid])
            out.append({"id":sid,"name":name,"online":frame is not None,
                        "motion":m["motion"],"motion_area":m["motion_area"]})
        return out

    def snapshot(self,sid):
        return self.get_jpeg(sid)

    def close(self):
        for _, source in self.sources.values():
            if hasattr(source,"stop"):
                source.stop()
