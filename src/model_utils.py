# src/model_utils.py

import os
import cv2
import torch
import timm
import numpy as np
import pandas as pd
from torchvision import transforms
from collections import OrderedDict

# 프로젝트 루트 기준 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_PATH = os.path.join(BASE_DIR, "meta.csv")


def load_labels_from_meta():
    """
    meta.csv에서 label 리스트를 읽어서 학습 시 사용된 것과 동일한 순서(정렬)로 반환.
    학습 스크립트에서 sorted(df["label"].unique()) 로 만들었기 때문에 그대로 재현.
    """
    if not os.path.exists(META_PATH):
        # meta.csv가 없으면 fallback
        return ["anger", "disgust", "fear", "happy", "none", "sad", "surprise"]

    df = pd.read_csv(META_PATH)
    uniq_labels = sorted(df["label"].unique())
    return list(uniq_labels)


class EmotionPredictor:
    """
    ConvNeXt Tiny 기반 감정 분류 모델.
    - weight_path: models/best_model.pth (기본)
    - 라벨 순서는 meta.csv의 정렬된 label 기준으로 맞춰서, 학습 시와 동일하게 맞춤.
    """

    def __init__(self,
                 weight_path: str = "models/best_model.pth",
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 라벨 로딩 (meta.csv 기준)
        self.labels = load_labels_from_meta()

        # 모델 생성
        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=len(self.labels)
        )

        # weight_path를 절대경로로 보정
        if not os.path.isabs(weight_path):
            weight_path = os.path.join(BASE_DIR, weight_path)

        sd = torch.load(weight_path, map_location=self.device)

        # best_model.pth 안에 state_dict만 있는 형식/딕셔너리 래핑 둘 다 허용
        if isinstance(sd, OrderedDict):
            state = sd
        else:
            state = sd.get("model", sd)

        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
        ])

        # 부정 감정(집중도 감점용) 집합
        neg_names = {"anger", "disgust", "fear", "sad"}
        self.negative = {lab for lab in self.labels
                         if any(k in lab.lower() for k in neg_names)}

    def predict(self, frame_bgr):
        """
        BGR 프레임 → (집중 점수, 확률 딕셔너리, 최빈 라벨)
        - score: 0~1 (부정 감정 확률 합으로 감점)
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()

        probs = {self.labels[i]: float(prob[i]) for i in range(len(self.labels))}
        neg_sum = sum(probs.get(k, 0.0) for k in self.negative)

        # 감정 기반 집중 점수: 1 - (부정 확률 합)
        emo_score = float(np.clip(1.0 - neg_sum, 0.0, 1.0))

        top_label = max(probs, key=probs.get)
        return emo_score, probs, top_label
