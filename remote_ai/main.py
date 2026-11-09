from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from .config import load_config
from .manager import SourceManager

BASE = Path(__file__).resolve().parent.parent
cfg = load_config(BASE / "config.yaml")
manager = SourceManager(cfg)
app = FastAPI(title=cfg.get("app",{}).get("title","Remote and Camera AI"))
templates = Jinja2Templates(directory=str(BASE / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request":request,"title":app.title})

@app.get("/api/sources")
def sources():
    return manager.status()

def frame_generator(sid):
    while True:
        jpg = manager.get_jpeg(sid)
        if jpg is None:
            import time
            time.sleep(.25)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"

@app.get("/stream/{sid}")
def stream(sid: str):
    if sid not in manager.ids():
        raise HTTPException(404,"Unknown source")
    return StreamingResponse(frame_generator(sid),media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/snapshot/{sid}")
def snapshot(sid: str):
    jpg = manager.snapshot(sid)
    if jpg is None:
        raise HTTPException(503,"Source has no frame yet")
    return Response(jpg,media_type="image/jpeg")

@app.on_event("shutdown")
def shutdown():
    manager.close()
