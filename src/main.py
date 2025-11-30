# src/main.py

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .camera_worker import shared_state, process_frame_bytes, start_camera_loop


# -----------------------------
# 경로 설정
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # focus_realtime/
STATIC_DIR = BASE_DIR / "static"
INDEX_HTML = STATIC_DIR / "index.html"

app = FastAPI(title="Focus Realtime API")

# ALB + HTTPS 뒤에서 쓰는 거라 CORS는 크게 상관 없지만,
# 로컬 테스트 및 확장성을 위해 열어둠.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 필요하면 kng3oo.site 로 좁혀도 됨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 (필요시 사용)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -----------------------------
# 초기화 (예전 호환용)
# -----------------------------
# 현재 start_camera_loop()는 no-op 이지만,
# 예전 코드를 깨지 않기 위해 한 번 호출만 해둔다.
start_camera_loop()


# -----------------------------
# 1) 기본 페이지: index.html
# -----------------------------
@app.get("/")
def root():
    """
    ALB Health Check + 브라우저 진입점
    """
    if INDEX_HTML.is_file():
        return FileResponse(str(INDEX_HTML))
    return JSONResponse({"ok": True, "msg": "Focus Realtime API Root"})


# -----------------------------
# 2) 프레임 업로드 (프론트에서 0.2초마다 호출)
# -----------------------------
@app.post("/api/frame")
async def api_frame(
    frame: UploadFile = File(...),
    user: str = Form(...),
):
    """
    - 프론트 index.html 에서 FormData(frame, user) 로 전송
    - frame: JPEG 이미지
    - user: 사용자 이름
    """
    data = await frame.read()
    process_frame_bytes(data, user=user)
    return {"ok": True}


# -----------------------------
# 3) 실시간 상태 조회 (프론트 poll())
# -----------------------------
@app.get("/api/live")
def api_live():
    """
    프론트가 집중도/감정/최근 값 표시용으로 주기적으로 호출.
    index.html 의 poll() 이 기대하는 구조에 맞춰 리턴.
    """
    with shared_state["lock"]:
        focus_hist = list(shared_state.get("focus", []))
        fps = float(shared_state.get("fps", 0.0))
        latest = shared_state.get("latest", {})

    return {
        "focus": focus_hist,
        "fps": fps,
        "latest": latest,
    }


@app.get("/api/session")
def api_session():
    """
    세션 메타 정보:
    - 총 프레임 수
    - 저장된 이미지 수 (집중도 40% 이하 자동 캡처 + 수동 스냅샷 포함)
    """
    with shared_state["lock"]:
        frames = int(shared_state.get("frames", 0))
        saved = int(shared_state.get("saved", 0))

    return {
        "frames": frames,
        "saved": saved,
    }


# -----------------------------
# 4) 수동 스냅샷 API (프론트 '현재 이미지 저장' 버튼)
# -----------------------------
@app.post("/api/snapshot")
async def api_snapshot(payload: dict):
    """
    현재 버전에서는 auto-screenshot(40% 미만) 은 FocusPipeline 내부에서
    이미 저장 중이고, 수동 스냅샷은 선택 사항이라
    우선은 '아직 구현 안 됨' 형태로 안전하게 응답.

    나중에 필요하면:
      - shared_state 에 마지막 프레임을 보관해두고
      - FocusPipeline 에 'save_only' 같은 메서드 추가해서
      - 여기서 불러주는 식으로 확장하면 됨.
    """
    return {"ok": False, "msg": "수동 스냅샷은 아직 별도 구현 안 됨 (자동 캡처는 동작 중)"}
