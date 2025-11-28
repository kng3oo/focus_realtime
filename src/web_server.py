import os
import json
import threading
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .camera_worker import start_camera_loop, shared_state, log_manager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static/* 경로 서빙
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup_event():
    print("🔥 startup_event → camera thread 시작")
    th = threading.Thread(target=start_camera_loop, daemon=True)
    th.start()


@app.get("/")
def index():
    """메인 대시보드 HTML"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/download_log")
def download_log():
    """CSV 로그 다운로드"""
    csv_path = log_manager.get_csv_path()
    return FileResponse(
        path=csv_path,
        filename=os.path.basename(csv_path),
        media_type="text/csv",
    )


@app.websocket("/ws/video")
async def ws_video(ws: WebSocket):
    """프레임 + 지표 실시간 전송"""
    await ws.accept()
    print("🟢 WebSocket connected")

    try:
        while True:
            frame_b64 = shared_state.get("frame_b64")
            if frame_b64 is None:
                await asyncio.sleep(0.1)
                continue

            payload = {
                "frame": frame_b64,
                "focus": shared_state.get("focus", 0.0),
                "avg_10s": shared_state.get("avg_10s", 0.0),
                "avg_1m": shared_state.get("avg_1m", 0.0),
                "avg_10m": shared_state.get("avg_10m", 0.0),
                "emotion_rank": shared_state.get("emotion_rank", []),
                "latest": shared_state.get("latest", {}),
            }

            await ws.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        print("🔴 WebSocket disconnected")
