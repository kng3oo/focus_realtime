# src/main.py
import os
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .camera_worker import (
    process_frame,
    get_live_state,
    get_session_state,
    save_manual_snapshot,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Focus Realtime API")

# /static 경로 마운트
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index(request: Request):
    """프론트 대시보드 HTML"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)


@app.post("/api/frame")
async def api_frame(
    frame: UploadFile = File(...),
    user: str = Form(...),
):
    """
    프론트에서 200ms마다 보내는 프레임 업로드 엔드포인트
    - frame: JPEG 이미지
    - user: 사용자 이름
    """
    data = await frame.read()
    process_frame(user, data)
    return {"ok": True}


@app.get("/api/live")
async def api_live():
    """
    실시간 상태 조회
    - focus: 최근 집중도 리스트 (0~1 사이 float)
    - fps: 추정 FPS
    - latest: 마지막 프레임의 상세 정보
    """
    return get_live_state()


@app.get("/api/session")
async def api_session():
    """
    세션 상태 정보
    - frames: 처리한 총 프레임 수
    - saved: 저장된 스크린샷 개수
    - start_ts: 세션 시작 시각
    """
    return get_session_state()


@app.post("/api/snapshot")
async def api_snapshot(payload: dict):
    """
    프론트에서 '현재 이미지 저장' 버튼 눌렀을 때 호출
    - user: 사용자 이름
    """
    user = payload.get("user") or "anonymous"
    ok = save_manual_snapshot(user)
    return {"ok": ok}
