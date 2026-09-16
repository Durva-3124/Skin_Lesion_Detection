"""
src/pipeline.py
End-to-end TejaLens skin pre-screening pipeline.
Chains: Image Quality Gate → Segmentation → Classification → Confidence + Grad-CAM → Risk output.

Usage:
    from src.pipeline import TejaLensPipeline
    pipeline = TejaLensPipeline(seg_weights="unet_vgg16.pth", cls_weights="efficientnet_b0_ham10000.pth")
    result = pipeline.run("path/to/image.jpg")
    print(result)
"""

import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.preprocessing.transforms import get_val_transforms, remove_hair
from src.segmentation.model import get_segmentation_model
from src.classification.model import get_classification_model
from src.preprocessing.dataset import HAM_CLASSES

import cv2


RISK_THRESHOLDS = {
    "high":   0.70,   # confidence >= 70% on a malignant class → high risk
    "medium": 0.40,   # 40–70% → medium risk
}

MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}   # HAM10000 malignant classes


class TejaLensPipeline:
    def __init__(
        self,
        seg_weights: str,
        cls_weights: str,
        seg_encoder: str = "vgg16",
        cls_model: str = "efficientnet_b0",
        num_classes: int = 7,
        class_names: list = None,
        image_size: int = 224,
        mc_dropout_passes: int = 30,
        device: str = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size
        self.mc_passes = mc_dropout_passes
        self.class_names = class_names or HAM_CLASSES
        self.transform = get_val_transforms(image_size)

        # Segmentation model
        self.seg_model = get_segmentation_model(encoder=seg_encoder).to(self.device)
        self.seg_model.load_state_dict(torch.load(seg_weights, map_location=self.device))
        self.seg_model.eval()

        # Classification model
        self.cls_model = get_classification_model(cls_model, num_classes=num_classes).to(self.device)
        self.cls_model.load_state_dict(torch.load(cls_weights, map_location=self.device))
        self.cls_model.eval()

    # ------------------------------------------------------------------
    # Stage 0: Image quality gate
    # ------------------------------------------------------------------
    def _quality_check(self, image: np.ndarray) -> tuple[bool, str]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            return False, f"Image too blurry (sharpness={laplacian_var:.1f}) — retake photo"
        if gray.mean() < 30:
            return False, "Image too dark — improve lighting and retake"
        if gray.mean() > 225:
            return False, "Image overexposed — reduce lighting and retake"
        return True, "ok"

    # ------------------------------------------------------------------
    # Stage 1: Segmentation
    # ------------------------------------------------------------------
    def _segment(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mask_logit = self.seg_model(tensor.unsqueeze(0).to(self.device))
            mask = torch.sigmoid(mask_logit).squeeze()
        return mask  # H x W, values 0–1

    # ------------------------------------------------------------------
    # Stage 2: Classification with MC Dropout uncertainty
    # ------------------------------------------------------------------
    def _classify_with_uncertainty(self, tensor: torch.Tensor) -> tuple[int, float, float, np.ndarray]:
        inp = tensor.unsqueeze(0).to(self.device)

        # Enable dropout at inference for MC Dropout
        def enable_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()

        self.cls_model.apply(enable_dropout)

        probs_list = []
        with torch.no_grad():
            for _ in range(self.mc_passes):
                logits = self.cls_model(inp)
                probs_list.append(F.softmax(logits, dim=1).cpu().numpy())

        self.cls_model.eval()  # restore eval mode

        probs_stack = np.stack(probs_list, axis=0)  # (passes, 1, num_classes)
        mean_probs = probs_stack.mean(axis=0).squeeze()   # (num_classes,)
        uncertainty = probs_stack.var(axis=0).squeeze().mean()  # scalar

        pred_class = int(mean_probs.argmax())
        confidence = float(mean_probs.max())
        return pred_class, confidence, float(uncertainty), mean_probs

    # ------------------------------------------------------------------
    # Stage 3: Risk level
    # ------------------------------------------------------------------
    def _risk_level(self, pred_class: int, confidence: float) -> str:
        class_name = self.class_names[pred_class]
        if class_name in MALIGNANT_CLASSES:
            if confidence >= RISK_THRESHOLDS["high"]:
                return "HIGH — lesion flagged as high-risk, recommend dermatologist review"
            return "MEDIUM — lesion shows potentially concerning features, clinical follow-up advised"
        if confidence >= RISK_THRESHOLDS["high"]:
            return "LOW — no immediate concern detected; routine monitoring recommended"
        return "MEDIUM — low-confidence prediction; clinical follow-up advised"

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self, image_path: str) -> dict:
        image_np = np.array(Image.open(image_path).convert("RGB"))

        # Stage 0: quality gate
        passed, reason = self._quality_check(image_np)
        if not passed:
            return {"status": "rejected", "reason": reason}

        # Hair removal
        image_clean = remove_hair(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        image_clean = cv2.cvtColor(image_clean, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(image_clean))

        # Stage 1: segmentation
        seg_mask = self._segment(tensor)

        # Stage 2: classification + uncertainty
        pred_class, confidence, uncertainty, mean_probs = self._classify_with_uncertainty(tensor)

        # Stage 3: risk level
        risk = self._risk_level(pred_class, confidence)

        return {
            "status": "processed",
            "predicted_class": self.class_names[pred_class],
            "confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 6),
            "risk_level": risk,
            "class_probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(self.class_names, mean_probs)
            },
            "segmentation_mask": seg_mask.cpu().numpy(),
        }
