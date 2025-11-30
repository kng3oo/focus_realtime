# src/model_utils.py
import os
from dataclasses import dataclass
from typing import Optional, Dict

import cv2
import numpy as np
import torch
import timm
from torchvision import transforms
import mediapipe as mp


# ---------------------------
# FocusResult: 한 프레임의 결과
# ---------------------------
@dataclass
class FocusResult:
    focus: float
    emotion_score: float
    blink_score: float
    top_emotion: Optional[str] = None
    probs: Optional[Dict[str, float]] = None


# ---------------------------
# Emotion 모델
# ---------------------------
DEFAULT_LABELS = ["anger", "disgust", "fear", "happy", "none", "sad", "surprise"]


class EmotionModel:
    def __init__(
        self,
        model_path: str,
        labels=None,
        img_size: int = 384,
        device: Optional[str] = None,
    ):
        self.labels = labels or DEFAULT_LABELS
        self.num_classes = len(self.labels)

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=self.num_classes,
        )

        sd = torch.load(model_path, map_location=self.device)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

        self.negative = {"anger", "disgust", "fear", "sad"}

    def score(self, frame_bgr: np.ndarray) -> tuple[float, Dict[str, float], str]:
        # BGR -> RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()

        probs = {lab: float(p) for lab, p in zip(self.labels, prob)}
        neg_sum = sum(probs.get(k, 0.0) for k in self.negative)
        score = float(np.clip(1.0 - neg_sum, 0.0, 1.0))
        top = max(probs, key=probs.get)
        return score, probs, top


# ---------------------------
# Blink / EAR 기반 점수
# ---------------------------
class BlinkModel:
    def __init__(self, ear_close=0.18, ear_open=0.30, win=3):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1
        )
        self.left_idx = [33, 160, 158, 133, 153, 144]
        self.right_idx = [263, 387, 385, 362, 380, 373]
        self.ear_close = ear_close
        self.ear_open = ear_open
        self.win = win
        self.hist = []
        self.prev_closed = False
        self.blinks = 0

    @staticmethod
    def _ear(e):
        A = np.linalg.norm(e[1] - e[5])
        B = np.linalg.norm(e[2] - e[4])
        C = np.linalg.norm(e[0] - e[3]) + 1e-6
        return (A + B) / (2.0 * C)

    def score(self, frame_bgr: np.ndarray) -> tuple[float, float, float]:
        h, w = frame_bgr.shape[:2]
        res = self.face_mesh.process(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )
        if not res.multi_face_landmarks:
            return 1.0, 0.3, 0.0  # 눈이 감지 안되면 대충 무난한 값

        lm = res.multi_face_landmarks[0].landmark
        ptsL = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in self.left_idx],
            dtype=np.float32,
        )
        ptsR = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in self.right_idx],
            dtype=np.float32,
        )
        earL = self._ear(ptsL)
        earR = self._ear(ptsR)
        ear = (earL + earR) / 2.0

        # EAR → 점수 (open > close)
        s = (ear - self.ear_close) / (
            self.ear_open - self.ear_close + 1e-6
        )
        frame_score = float(np.clip(s, 0.0, 1.0))

        # 깜빡임 계산
        closed = ear < self.ear_close
        self.hist.append(closed)
        if len(self.hist) > self.win:
            self.hist.pop(0)
        now_closed = sum(self.hist) >= (self.win // 2 + 1)
        blink_evt = self.prev_closed and not now_closed
        if blink_evt:
            self.blinks += 1
        self.prev_closed = now_closed

        blink_rate = float(self.blinks)
        return frame_score, ear, blink_rate


# ---------------------------
# 최종 파이프라인
# ---------------------------
class FocusPipeline:
    def __init__(self, models_dir: str):
        model_path = os.path.join(models_dir, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Emotion model not found: {model_path}"
            )
        self.emotion = EmotionModel(model_path)
        self.blink = BlinkModel()

    def score(self, frame_bgr: np.ndarray) -> FocusResult:
        emo_score, probs, top = self.emotion.score(frame_bgr)
        blink_score, ear, blink_rate = self.blink.score(frame_bgr)

        # Emotion 70% + Blink 30%
        focus = 0.7 * emo_score + 0.3 * blink_score
        focus = float(np.clip(focus, 0.0, 1.0))

        return FocusResult(
            focus=focus,
            emotion_score=emo_score,
            blink_score=blink_score,
            top_emotion=top,
            probs=probs,
        )
