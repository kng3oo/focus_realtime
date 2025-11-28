# src/camera_worker.py

import os
import time
import base64
from collections import deque, Counter
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

from .model_utils import EmotionPredictor
from .pose_utils import HeadPoseEstimator


# =========================
# 로그 매니저
# =========================
class LogManager:
    def __init__(self, base_dir: str = "runs"):
        os.makedirs(base_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, ts)
        os.makedirs(self.session_dir, exist_ok=True)

        self.csv_path = os.path.join(self.session_dir, "focus_log.csv")
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write("time,focus,ear,yaw,pitch,roll,emotion\n")

    def write_row(self, t, focus, ear, yaw, pitch, roll, emo):
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{t:.2f},{focus:.4f},{ear:.4f},{yaw:.2f},{pitch:.2f},{roll:.2f},{emo}\n"
            )

    def get_csv_path(self):
        return self.csv_path


log_manager = LogManager()


# =========================
# 웹 공유 상태
# =========================
shared_state = {
    "frame_b64": None,
    "focus": 0.0,
    "avg_10s": 0.0,
    "avg_1m": 0.0,
    "avg_10m": 0.0,
    "emotion_rank": [],
    "latest": {},
    "user": "",
}

# =========================
# EAR 기준 (초기 버전으로 복구)
# =========================
EAR_CLOSE = 0.18   # 눈 감김
EAR_OPEN = 0.30    # 눈 완전 뜬 상태


class BlinkEstimator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
        self.left_idx = [33, 160, 158, 133, 153, 144]
        self.right_idx = [263, 387, 385, 362, 380, 373]

    def _ear(self, e):
        A = np.linalg.norm(e[1] - e[5])
        B = np.linalg.norm(e[2] - e[4])
        C = np.linalg.norm(e[0] - e[3]) + 1e-6
        return (A + B) / (2.0 * C)

    def score(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        res = self.face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            # 얼굴 못 찾으면 중립값
            return 0.7, 0.0

        lm = res.multi_face_landmarks[0].landmark
        xy = lambda i: np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

        L = np.array([xy(i) for i in self.left_idx])
        R = np.array([xy(i) for i in self.right_idx])

        ear = (self._ear(L) + self._ear(R)) / 2.0
        s = (ear - EAR_CLOSE) / (EAR_OPEN - EAR_CLOSE + 1e-6)
        score = float(np.clip(s, 0.0, 1.0))
        return score, float(ear)


# =========================
# 집중도 가중합 (조금 더 높게 나오도록 조정)
# =========================
def concentration_score(
    gaze, neck, emotion, blink,
    w_gaze=0.35, w_neck=0.30, w_emotion=0.25, w_blink=0.10
):
    vals = [gaze or 0.0, neck or 0.0, emotion or 0.0, blink or 0.0]
    ws = [w_gaze, w_neck, w_emotion, w_blink]
    s = sum(a * b for a, b in zip(vals, ws))
    return float(np.clip(s, 0.0, 1.0))


# =========================
# 메인 카메라 루프 (스레드에서 실행)
# =========================
def start_camera_loop():
    print("📸 start_camera_loop 시작")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
        return

    blink = BlinkEstimator()
    pose = HeadPoseEstimator()
    emo = EmotionPredictor()  # models/best_model.pth + meta.csv 사용

    focus_hist = deque(maxlen=60 * 10 * 30)  # 최대 10분치 근사
    emotion_counter = Counter()
    t0 = time.time()

    os.makedirs("screenshots", exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        t = time.time() - t0

        # --- 개별 지표 ---
        blink_score, ear = blink.score(frame)
        neck_score, (yaw, pitch, roll) = pose.score(frame)
        emo_score, probs, emo_top = emo.predict(frame)
        gaze_score = 1.0  # 아직 시선추적 없음 → 항상 정면 본다고 가정

        # --- 최종 집중도 ---
        final_focus = concentration_score(
            gaze_score, neck_score, emo_score, blink_score
        )
        focus_hist.append(final_focus)

        # --- 평균값 (샘플 개수 기반 근사) ---
        arr = list(focus_hist)
        avg_10s = float(np.mean(arr[-300:])) if len(arr) >= 30 else final_focus
        avg_1m = float(np.mean(arr[-1800:])) if len(arr) >= 180 else final_focus
        avg_10m = float(np.mean(arr)) if len(arr) >= 600 else final_focus

        # --- 감정 빈도 카운트 ---
        emotion_counter[emo_top] += 1
        rank = emotion_counter.most_common(5)  # [(label, count), ...]

        # --- 30% 미만 → 스크린샷 저장 (텍스트 오버레이 포함) ---
        if final_focus < 0.3:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            img = frame.copy()
            cv2.putText(
                img, f"Focus: {final_focus*100:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )
            cv2.putText(
                img, ts, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
            out_path = os.path.join("screenshots", f"focus_drop_{ts}.jpg")
            cv2.imwrite(out_path, img)

        # --- CSV 로그 기록 ---
        log_manager.write_row(t, final_focus, ear, yaw, pitch, roll, emo_top)

        # --- 프레임 → base64 인코딩 ---
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        frame_b64 = base64.b64encode(buf).decode("ascii")

        # --- shared_state 업데이트 ---
        shared_state["frame_b64"] = frame_b64
        shared_state["focus"] = final_focus
        shared_state["avg_10s"] = avg_10s
        shared_state["avg_1m"] = avg_1m
        shared_state["avg_10m"] = avg_10m
        shared_state["emotion_rank"] = [[k, v] for k, v in rank]
        shared_state["latest"] = {
            "ear": ear,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "emotion": emo_top,
        }

        time.sleep(0.03)  # ~30fps 정도
