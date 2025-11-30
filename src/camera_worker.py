# src/camera_worker.py
import os
import csv
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from .model_utils import FocusPipeline, FocusResult

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
SCREEN_DIR = os.path.join(BASE_DIR, "screenshots")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(SCREEN_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# 세션 식별용
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH = os.path.join(RUNS_DIR, f"focus_log_{SESSION_ID}.csv")

# 파이프라인 (감정 + 깜빡임 → 집중도)
_pipeline = FocusPipeline(models_dir=MODELS_DIR)

# 공유 상태 (프론트 /api/live, /api/session 용)
_state_lock = threading.Lock()
_state = {
    "start_ts": datetime.now().isoformat(),
    "frames": 0,
    "saved": 0,
    "focus_hist": deque(maxlen=6000),  # 최근 값들
    "fps": 0.0,
    "latest": {},
}

# FPS 계산용
_ts_hist = deque(maxlen=60)

# 마지막 프레임/결과 (수동 스냅샷용)
_last_frame_lock = threading.Lock()
_last_frame_bgr = None
_last_result = None
_last_user = None

# --- CSV 로깅 초기화 ---
_csv_lock = threading.Lock()
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time_iso",
                "user",
                "focus",
                "emotion_score",
                "blink_score",
                "top_emotion",
            ]
        )


def _append_csv(row):
    with _csv_lock:
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(row)


def _draw_overlay(frame_bgr, user, result: FocusResult) -> np.ndarray:
    """
    저장용 이미지에 텍스트 오버레이
    - user 이름
    - 집중도/감정/깜빡임 점수 (퍼센트 표기)
    - 가중치 비율(Emotion 70%, Blink 30%)
    """
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    # 반투명 바탕
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 130), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    y = 35
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 사용자 이름 (한글은 OpenCV 기본폰트에서 깨질 수 있음)
    cv2.putText(
        img,
        f"User: {user}",
        (20, y),
        font,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y += 30

    focus_pct = int(round(result.focus * 100))
    emo_pct = int(round(result.emotion_score * 100))
    blink_pct = int(round(result.blink_score * 100))

    cv2.putText(
        img,
        f"Focus: {focus_pct}%",
        (20, y),
        font,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y += 25

    cv2.putText(
        img,
        f"EmotionScore: {emo_pct}% (w=0.7)",
        (20, y),
        font,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    y += 25

    cv2.putText(
        img,
        f"BlinkScore: {blink_pct}% (w=0.3)",
        (20, y),
        font,
        0.6,
        (255, 200, 0),
        2,
        cv2.LINE_AA,
    )
    y += 25

    if result.top_emotion:
        cv2.putText(
            img,
            f"Top Emotion: {result.top_emotion}",
            (20, y),
            font,
            0.6,
            (200, 200, 255),
            2,
            cv2.LINE_AA,
        )

    return img


def process_frame(user: str, frame_bytes: bytes):
    """
    /api/frame 에서 호출.
    1) JPEG → BGR 이미지 디코딩
    2) 모델 추론 (emotion + blink → focus)
    3) CSV 로그 기록
    4) focus < 0.4 이면 annotated 스크린샷 자동 저장
    5) /api/live /api/session용 상태 업데이트
    """
    global _last_frame_bgr, _last_result, _last_user

    # 1) 디코드
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return

    # 2) 모델 추론
    result: FocusResult = _pipeline.score(frame)

    # 3) CSV 로그
    now_iso = datetime.utcnow().isoformat()
    _append_csv(
        [
            now_iso,
            user,
            f"{result.focus:.4f}",
            f"{result.emotion_score:.4f}",
            f"{result.blink_score:.4f}",
            result.top_emotion or "",
        ]
    )

    # 4) focus < 0.4 -> 스크린샷 자동 저장
    if result.focus < 0.4:
        annotated = _draw_overlay(frame, user, result)
        fname = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user}_focus_{int(result.focus*100)}.jpg"
        # 한글 파일명 문제가 있으면 필요하면 나중에 user 부분만 별도 처리
        save_path = os.path.join(SCREEN_DIR, fname)
        cv2.imwrite(save_path, annotated)
        with _state_lock:
            _state["saved"] += 1

    # 5) 상태 업데이트
    now_ts = time.time()
    with _state_lock:
        _state["frames"] += 1
        _state["focus_hist"].append(result.focus)

        _ts_hist.append(now_ts)
        if len(_ts_hist) >= 2:
            _state["fps"] = len(_ts_hist) / max(
                1e-6, (_ts_hist[-1] - _ts_hist[0])
            )

        _state["latest"] = {
            "user": user,
            "focus": float(result.focus),
            "emotion_score": float(result.emotion_score),
            "blink_score": float(result.blink_score),
            "top_emotion": result.top_emotion,
        }

    # 마지막 프레임/결과 저장 (수동 스냅샷용)
    with _last_frame_lock:
        _last_frame_bgr = frame
        _last_result = result
        _last_user = user


def get_live_state():
    with _state_lock:
        return {
            "focus": list(_state["focus_hist"]),
            "fps": float(_state["fps"]),
            "latest": _state["latest"],
        }


def get_session_state():
    with _state_lock:
        return {
            "start_ts": _state["start_ts"],
            "frames": int(_state["frames"]),
            "saved": int(_state["saved"]),
        }


def save_manual_snapshot(user: str) -> bool:
    """
    /api/snapshot 에서 호출.
    - 마지막으로 처리된 프레임을 기준으로 즉시 저장
    """
    with _last_frame_lock:
        if _last_frame_bgr is None or _last_result is None:
            return False

        frame = _last_frame_bgr.copy()
        result = _last_result

    annotated = _draw_overlay(frame, user, result)
    fname = f"manual_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{user}_focus_{int(result.focus*100)}.jpg"
    save_path = os.path.join(SCREEN_DIR, fname)
    cv2.imwrite(save_path, annotated)

    with _state_lock:
        _state["saved"] += 1

    return True
