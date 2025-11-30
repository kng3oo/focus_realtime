# src/camera_worker.py

import time
import threading
from collections import deque
from typing import Dict, Any

import cv2
import numpy as np

from .model_utils import FocusPipeline  # 기존 모델 파이프라인 그대로 사용


# -----------------------------
# 전역 파이프라인 & 공유 상태
# -----------------------------
_pipeline = FocusPipeline()

# 프론트가 보는 실시간 상태
shared_state: Dict[str, Any] = {
    "lock": threading.Lock(),
    "focus": deque(maxlen=6000),   # 최근 집중도 히스토리 (최대 6000샘플 저장)
    "fps": 0.0,
    "frames": 0,
    "saved": 0,
    "last_ts": None,              # FPS 계산용
    "latest": {},                 # 최근 1프레임 상세 정보
}


# -----------------------------
# API에서 호출하는 프레임 처리 함수
# -----------------------------
def process_frame_bytes(jpeg_bytes: bytes, user: str) -> None:
    """
    /api/frame 에서 올라온 JPEG 바이트를 받아서:
      1) BGR 프레임으로 디코딩
      2) FocusPipeline 으로 추론
      3) shared_state 에 실시간 상태 갱신
    """
    # 1) JPEG → BGR 이미지
    np_arr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return

    # 2) 모델 파이프라인 실행 (로그/스크린샷 저장은 내부에서 그대로 수행)
    #    run() 이 FocusResult 같은 객체를 리턴한다고 가정
    result = _pipeline.run(frame, user)

    now = time.time()

    # 3) FPS / 히스토리 / latest 업데이트
    with shared_state["lock"]:
        shared_state["frames"] += 1

        # FPS 계산
        last_ts = shared_state.get("last_ts")
        if last_ts is not None:
            dt = now - last_ts
            if dt > 0:
                shared_state["fps"] = 1.0 / dt
        shared_state["last_ts"] = now

        # 집중도 히스토리
        focus_val = float(getattr(result, "focus", 0.0))
        shared_state["focus"].append(focus_val)

        # 저장된 이미지 개수(파이프라인에서 저장 여부를 알려준다고 가정)
        saved_flag = (
            getattr(result, "saved", False)
            or bool(getattr(result, "saved_image_path", None))
        )
        if saved_flag:
            shared_state["saved"] += 1

        # 프론트에서 보는 latest 상세 정보
        shared_state["latest"] = {
            "user": user,
            "focus": focus_val,
            "emotion_score": float(getattr(result, "emotion_score", 0.0)),
            "blink_score": float(getattr(result, "blink_score", 0.0)),
            "top_emotion": getattr(result, "top_emotion", None),
            # 필요하면 나중에 yaw/pitch/roll, raw dict 등도 추가 가능
        }


# -----------------------------
# 이전 버전 호환용 (카메라 루프 안 쓰는 버전)
# -----------------------------
def start_camera_loop() -> None:
    """
    예전 구조에서 main.py가 import 하면서 호출하던 함수.
    지금은 API 프레임 방식이라 별도의 카메라 루프가 필요 없어서 no-op 처리.
    """
    return
