import os
import time
import threading
from collections import deque
from typing import Dict, Any, Optional

import cv2
import numpy as np

from .model_utils import FocusPipeline, FocusResult

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")
SCREEN_DIR = os.path.join(PROJECT_ROOT, "screenshots")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(SCREEN_DIR, exist_ok=True)

LOG_PATH = os.path.join(RUNS_DIR, f"focus_log_{int(time.time())}.csv")

shared_state: Dict[str, Any] = dict(
    lock=threading.Lock(),
    time=deque(maxlen=5000),
    focus=deque(maxlen=5000),
    emotion=deque(maxlen=5000),
    blink=deque(maxlen=5000),
    latest={},
    frames=0,
    saved=0,
    fps=0.0,
    start_ts=time.strftime("%Y-%m-%d %H:%M:%S"),
)

_pipeline = FocusPipeline()

_last_frame_bgr: Optional[np.ndarray] = None
_last_result: Optional[FocusResult] = None

_last_save_ts = 0.0
_focus_threshold = 0.4
_save_cooldown = 5.0

_frame_count = 0
_last_fps_ts = time.time()

_user_name = "anonymous"
_user_slug = "anonymous"


def slugify_korean(text: str) -> str:
    """한국어 → 로마자 비슷한 ASCII 변환 (단순 치환)"""
    import re
    mapping = {
        "가":"ga","나":"na","다":"da","라":"ra","마":"ma","바":"ba","사":"sa",
        "아":"a","자":"ja","차":"cha","카":"ka","타":"ta","파":"pa","하":"ha",
        "강":"kang","김":"kim","박":"park","정":"jung","이":"lee","최":"choi",
        "홍":"hong","길":"gil","동":"dong"
    }
    out = ""
    for ch in text:
        if ch in mapping:
            out += mapping[ch]
        elif ch.isalnum():
            out += ch.lower()
        # 그 외 문자는 제거
    return re.sub(r"[^a-z0-9]+", "", out.lower()) or "user"


def set_user_name(name: Optional[str]):
    global _user_name, _user_slug
    if not name:
        _user_name = "anonymous"
        _user_slug = "anonymous"
    else:
        _user_name = name
        _user_slug = slugify_korean(name)


def _save_log(elapsed: float, res: FocusResult):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(
            f"{_user_name},{elapsed:.2f},{res.focus:.4f},{res.emotion_score:.4f},"
            f"{res.blink_score:.4f},{res.ear or 0:.4f},{res.blink_rate or 0:.2f},{res.top_emotion}\n"
        )


def _render_overlay(frame: np.ndarray, res: FocusResult):
    """저장용 이미지에 텍스트 표시 (영문 ASCII username 사용)"""
    # 크게 보기 좋게 1280x960으로 업스케일
    overlay = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_CUBIC)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 40

    cv2.putText(overlay, f"User: {_user_slug}", (20, y), font, 1.0, (255,255,255), 2)
    y += 40
    cv2.putText(overlay, f"Focus: {res.focus*100:.1f}%", (20, y), font, 1.0, (0,255,255), 2)
    y += 40
    cv2.putText(overlay, f"EmotionScore: {res.emotion_score*100:.1f}% (w=0.7)", (20,y), font, 0.9, (0,255,0), 2)
    y += 35
    cv2.putText(overlay, f"BlinkScore: {res.blink_score*100:.1f}% (w=0.3)", (20,y), font, 0.9, (0,200,255), 2)
    y += 35
    if res.top_emotion:
        cv2.putText(overlay, f"TopEmotion: {res.top_emotion}", (20,y), font, 0.9, (255,200,0), 2)

    return overlay


def _save_image(frame_bgr: np.ndarray, res: FocusResult, force=False):
    global _last_save_ts

    now = time.time()
    if not force:
        if res.focus >= _focus_threshold:
            return False
        if now - _last_save_ts < _save_cooldown:
            return False

    ts = time.strftime("%Y%m%d_%H%M%S")
    pct = int(res.focus * 100)

    filename = f"{_user_slug}_focus{pct}_{ts}.jpg"  # ASCII-only
    path = os.path.join(SCREEN_DIR, filename)

    overlay = _render_overlay(frame_bgr, res)
    cv2.imwrite(path, overlay)

    with shared_state["lock"]:
        shared_state["saved"] += 1

    _last_save_ts = now
    print(f"[Saved] {path}")
    return True


def process_frame(frame_bgr: np.ndarray, user: Optional[str] = None):
    global _last_frame_bgr, _last_result
    global _frame_count, _last_fps_ts

    if user:
        set_user_name(user)

    t_abs = time.time()
    elapsed = t_abs - _pipeline.start_ts

    res: FocusResult = _pipeline.process(frame_bgr)
    _last_frame_bgr = frame_bgr.copy()
    _last_result = res

    _save_log(elapsed, res)
    _save_image(frame_bgr, res, force=False)

    _frame_count += 1
    dt = t_abs - _last_fps_ts
    if dt >= 1.0:
        shared_state["fps"] = _frame_count / dt
        _frame_count = 0
        _last_fps_ts = t_abs

    with shared_state["lock"]:
        shared_state["time"].append(elapsed)
        shared_state["focus"].append(res.focus)
        shared_state["emotion"].append(res.emotion_score)
        shared_state["blink"].append(res.blink_score)
        shared_state["frames"] += 1
        shared_state["latest"] = dict(
            user=_user_name,
            focus=res.focus,
            emotion_score=res.emotion_score,
            blink_score=res.blink_score,
            ear=res.ear,
            blink_rate=res.blink_rate,
            top_emotion=res.top_emotion,
        )


def save_snapshot(user=None):
    if user:
        set_user_name(user)

    if _last_frame_bgr is None or _last_result is None:
        return False

    return _save_image(_last_frame_bgr, _last_result, force=True)


def get_live_state():
    with shared_state["lock"]:
        return dict(
            time=list(shared_state["time"]),
            focus=list(shared_state["focus"]),
            emotion=list(shared_state["emotion"]),
            blink=list(shared_state["blink"]),
            latest=shared_state["latest"],
            fps=shared_state["fps"],
        )


def get_session_meta():
    with shared_state["lock"]:
        return dict(
            start_ts=shared_state["start_ts"],
            frames=shared_state["frames"],
            saved=shared_state["saved"],
        )
