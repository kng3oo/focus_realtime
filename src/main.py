# src/main.py
import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
import cv2

from .camera_worker import (
    process_frame,
    get_live_state,
    get_session_meta,
    save_snapshot,
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Focus Realtime")

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


@app.get("/api/live")
def api_live() -> Dict[str, Any]:
    return get_live_state()


@app.get("/api/session")
def api_session() -> Dict[str, Any]:
    return get_session_meta()


@app.post("/api/frame")
async def api_frame(
    user: str = Form("", description="사용자 이름"),
    frame: UploadFile = File(...),
):
    """
    브라우저에서 업로드한 프레임(JPEG)을 받아서 numpy BGR로 변환 후 처리.
    user는 Form 필드로 같이 온다.
    """
    data = await frame.read()
    npimg = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is not None:
        process_frame(img, user=user or None)
    return {"ok": True}


@app.post("/api/snapshot")
async def api_snapshot(payload: Optional[Dict[str, Any]] = Body(None)):
    """
    '현재 기준으로 이미지 저장' 버튼이 호출하는 엔드포인트.
    마지막 프레임/점수 기준으로 강제 저장(force=True).
    """
    user: Optional[str] = None
    if payload and isinstance(payload, dict):
        user = payload.get("user")
    ok = save_snapshot(user=user)
    return {"ok": ok}
