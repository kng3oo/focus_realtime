import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import timm
from torchvision import transforms

try:
    import mediapipe as mp
except ImportError:
    mp = None  # mediapipe 설치 실패 시 크래시 방지


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
META_CSV = os.path.join(PROJECT_ROOT, "meta.csv")


@dataclass
class FocusResult:
    """한 프레임 분석 결과 구조체"""
    focus: float
    emotion_score: float
    top_emotion: Optional[str]
    blink_score: float
    ear: Optional[float]
    blink_rate: Optional[float]
    raw_probs: Dict[str, float]


# ----------------------------------------------------
# Emotion (감정 모델)
# ----------------------------------------------------
class EmotionModel:
    """ConvNeXt Tiny 기반 감정 분류 모델 (score 함수 유지)"""

    def __init__(self, device: Optional[str] = None, img_size: int = 384):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        # meta.csv → 라벨 로드
        if os.path.exists(META_CSV):
            df = pd.read_csv(META_CSV)
            labels = sorted(df["label"].unique())
        else:
            labels = ["anger","disgust","fear","happy","none","sad","surprise"]

        self.labels = labels
        self.num_classes = len(labels)

        # 모델 생성
        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=self.num_classes,
        )

        # 가중치 로드
        if os.path.exists(MODEL_PATH):
            sd = torch.load(MODEL_PATH, map_location=self.device)
            if isinstance(sd, dict):
                if "model" in sd:
                    sd = sd["model"]
                elif "state_dict" in sd:
                    sd = sd["state_dict"]
            try:
                self.model.load_state_dict(sd, strict=False)
                print("✅ Emotion model loaded.")
            except Exception as e:
                print("⚠️ 모델 weights 로드 실패:", e)
        else:
            print("⚠️ best_model.pth 없음 → 감정 점수 dummy 반환됨.")

        self.model.to(self.device).eval()

        # Transform 정의
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225],
            ),
        ])

        # 부정 감정
        self.negative_labels = {"anger","disgust","fear","sad"}

    def _crop_center(self, bgr):
        """센터 크롭 (얼굴 인식 없는 경우 대비)"""
        h, w = bgr.shape[:2]
        side = min(h, w)
        y1 = (h - side) // 2
        x1 = (w - side) // 2
        return bgr[y1:y1+side, x1:x1+side]

    def score(self, frame_bgr: np.ndarray) -> Tuple[float, Dict[str, float], Optional[str]]:
        """
        감정 점수, 전체 확률, top emotion 반환
        —> 함수명 유지(score)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.5, {}, None

        img = self._crop_center(frame_bgr)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()

        probs = {lab: float(p) for lab, p in zip(self.labels, prob)}

        neg_sum = sum(probs.get(k, 0.0) for k in self.negative_labels)
        emotion_score = float(np.clip(1.0 - neg_sum, 0.0, 1.0))
        top = max(probs, key=probs.get) if probs else None

        return emotion_score, probs, top


# ----------------------------------------------------
# Blink (EAR + 눈뜸 점수)
# ----------------------------------------------------
class BlinkEstimator:
    """EAR 기반 깜빡임 탐지"""

    def __init__(self, ear_close=0.18, ear_open=0.30, hist_len=60):
        self.ear_close = ear_close
        self.ear_open = ear_open
        self.hist = []
        self.max_hist = hist_len
        self.blinks = 0
        self.prev_closed = False

        if mp:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
            self.left_idx = [33,160,158,133,153,144]
            self.right_idx = [263,387,385,362,380,373]
        else:
            self.face_mesh = None

    def _ear(self, e: np.ndarray) -> float:
        A = np.linalg.norm(e[1] - e[5])
        B = np.linalg.norm(e[2] - e[4])
        C = np.linalg.norm(e[0] - e[3]) + 1e-6
        return (A + B) / (2.0 * C)

    def score(self, frame_bgr: np.ndarray) -> Tuple[float, Optional[float], Optional[float]]:
        """
        눈뜸 점수, EAR, blink_rate 반환
        """
        if self.face_mesh is None:
            return 1.0, None, None

        h, w = frame_bgr.shape[:2]
        res = self.face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return 0.5, None, None

        lm = res.multi_face_landmarks[0].landmark

        def xy(i):
            return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

        L = np.array([xy(i) for i in self.left_idx])
        R = np.array([xy(i) for i in self.right_idx])
        ear = (self._ear(L) + self._ear(R)) / 2.0

        # EAR → 0~1 score
        s = (ear - self.ear_close) / (self.ear_open - self.ear_close + 1e-6)
        frame_score = float(np.clip(s, 0.0, 1.0))

        closed = ear < self.ear_close
        self.hist.append(closed)
        if len(self.hist) > self.max_hist:
            self.hist.pop(0)

        curr_closed = closed
        if self.prev_closed and not curr_closed:
            self.blinks += 1
        self.prev_closed = curr_closed

        blink_rate = None
        if self.max_hist > 0:
            blink_rate = float(self.blinks * (60.0 / max(1, len(self.hist))))

        return frame_score, float(ear), blink_rate


# ----------------------------------------------------
# Focus Pipeline (emotion + blink)
# ----------------------------------------------------
class FocusPipeline:
    """브라우저 카메라 프레임 1장 → 집중도 계산"""

    def __init__(self):
        self.emotion = EmotionModel()
        self.blink = BlinkEstimator()
        self.start_ts = time.time()

    def process(self, frame_bgr: np.ndarray) -> FocusResult:
        # Emotion
        emo_score, probs, top = self.emotion.score(frame_bgr)

        # Blink
        blink_score, ear, blink_rate = self.blink.score(frame_bgr)

        # 간단 가중합
        final_focus = 0.7 * emo_score + 0.3 * blink_score
        final_focus = float(np.clip(final_focus, 0.0, 1.0))

        return FocusResult(
            focus=final_focus,
            emotion_score=emo_score,
            top_emotion=top,
            blink_score=blink_score,
            ear=ear,
            blink_rate=blink_rate,
            raw_probs=probs,
        )
