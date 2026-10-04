"""
src/pipeline.py
End-to-end TejaLens skin pre-screening pipeline.
Chains: Image Quality Gate → Segmentation → Classification → Confidence + Grad-CAM → Risk output.

Usage:
    from src.pipeline import TejaLensPipeline
    pipeline = TejaLensPipeline(seg_weights="unet_vgg16.pth", cls_weights="efficientnet_b0_ham10000.pth")
    result = pipeline.run("path/to/image.jpg")
    result = pipeline.run("path/to/image.jpg", explain=True)  # also returns gradcam_heatmap
    print(result)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.preprocessing.transforms import get_val_transforms, remove_hair
from src.segmentation.model import get_segmentation_model
from src.classification.model import get_classification_model
from src.preprocessing.dataset import HAM_CLASSES

import cv2


RISK_HIGH_THRESHOLD = 0.70   # confidence >= 70% on a malignant class → high risk

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
        # U-Net VGG16 has 5 pooling stages; minimum safe input is 512px
        self.seg_size = 512
        self.seg_transform = get_val_transforms(self.seg_size)

        # Segmentation model
        self.seg_model = get_segmentation_model(encoder=seg_encoder).to(self.device)
        self.seg_model.load_state_dict(torch.load(seg_weights, map_location=self.device, weights_only=True))
        self.seg_model.eval()

        # Classification model
        self.cls_model = get_classification_model(cls_model, num_classes=num_classes).to(self.device)
        self.cls_model.load_state_dict(torch.load(cls_weights, map_location=self.device, weights_only=True))
        self.cls_model.eval()

        # Optional calibration (temperature + threshold)
        self.temperature = 1.0
        self.threshold   = None
        from src.config import MODELS_DIR
        import json as _json
        cal_path = MODELS_DIR / "calibration.json"
        if cal_path.exists():
            cal = _json.loads(cal_path.read_text())
            self.temperature = cal.get("temperature", 1.0)
            self.threshold   = cal.get("threshold", None)

    # ------------------------------------------------------------------
    # Stage 0: Image quality gate
    # ------------------------------------------------------------------
    def _quality_check(self, image: np.ndarray) -> tuple[bool, str]:
        # Normalise to fixed size before sharpness check to avoid resolution bias
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (224, 224))
        # Check brightness first — a dark image is dark, not blurry
        mean_brightness = float(gray_resized.mean())
        if mean_brightness < 30:
            return False, "Image too dark — improve lighting and retake"
        if mean_brightness > 225:
            return False, "Image overexposed — reduce lighting and retake"
        laplacian_var = cv2.Laplacian(gray_resized, cv2.CV_64F).var()
        if laplacian_var < 50:
            return False, f"Image too blurry (sharpness={laplacian_var:.1f}) — retake photo"
        return True, "ok"

    # ------------------------------------------------------------------
    # Stage 1: Segmentation (runs at seg_size=512 to avoid spatial collapse)
    # ------------------------------------------------------------------
    def _segment(self, image_pil) -> torch.Tensor:
        seg_tensor = self.seg_transform(image_pil)
        with torch.no_grad():
            mask_logit = self.seg_model(seg_tensor.unsqueeze(0).to(self.device))
            mask = torch.sigmoid(mask_logit).squeeze(0).squeeze(0)
        return mask  # H x W, values 0–1

    # ------------------------------------------------------------------
    # Stage 2: Classification with MC Dropout uncertainty
    # ------------------------------------------------------------------
    def _classify_with_uncertainty(self, tensor: torch.Tensor) -> tuple[int, float, float, np.ndarray]:
        inp = tensor.unsqueeze(0).to(self.device)

        def enable_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()

        self.cls_model.apply(enable_dropout)
        probs_list = []
        try:
            with torch.no_grad():
                for _ in range(self.mc_passes):
                    logits = self.cls_model(inp) / self.temperature  # apply calibration temperature
                    probs_list.append(F.softmax(logits, dim=1).cpu().numpy())
        finally:
            self.cls_model.eval()  # always restore eval mode

        probs_stack = np.stack(probs_list, axis=0)  # (passes, 1, num_classes)
        mean_probs  = probs_stack.mean(axis=0).squeeze()   # (num_classes,)
        uncertainty = float(probs_stack.var(axis=0).squeeze().mean())

        pred_class = int(mean_probs.argmax())
        confidence = float(mean_probs.max())
        return pred_class, confidence, uncertainty, mean_probs

    # ------------------------------------------------------------------
    # Stage 3: Risk level
    # ------------------------------------------------------------------
    def _risk_level(self, pred_class: int, confidence: float, mean_probs: np.ndarray) -> str:
        class_name = self.class_names[pred_class]
        # If calibration threshold is set, use it for malignant decision
        if self.threshold is not None:
            malignant_idx = [i for i, c in enumerate(self.class_names) if c in MALIGNANT_CLASSES]
            prob_malignant = float(mean_probs[malignant_idx].sum())
            if prob_malignant >= self.threshold:
                return "HIGH — lesion flagged as high-risk, recommend dermatologist review"
            return "LOW — no immediate concern detected; routine monitoring recommended"
        # Fallback: argmax-based risk
        if class_name in MALIGNANT_CLASSES:
            if confidence >= RISK_HIGH_THRESHOLD:
                return "HIGH — lesion flagged as high-risk, recommend dermatologist review"
            return "MEDIUM — lesion shows potentially concerning features, clinical follow-up advised"
        if confidence >= RISK_HIGH_THRESHOLD:
            return "LOW — no immediate concern detected; routine monitoring recommended"
        return "MEDIUM — low-confidence prediction; clinical follow-up advised"

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self, image_path: str, explain: bool = False) -> dict:
        image_np = np.array(Image.open(image_path).convert("RGB"))

        # Stage 0: quality gate
        passed, reason = self._quality_check(image_np)
        if not passed:
            return {"status": "rejected", "reason": reason}

        # Hair removal
        image_clean = remove_hair(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        image_clean_rgb = cv2.cvtColor(image_clean, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_clean_rgb)
        tensor = self.transform(image_pil)

        # Stage 1: segmentation (at 512px)
        seg_mask = self._segment(image_pil)

        # Stage 2: classification + uncertainty
        pred_class, confidence, uncertainty, mean_probs = self._classify_with_uncertainty(tensor)

        # Stage 3: risk level
        risk = self._risk_level(pred_class, confidence, mean_probs)

        result = {
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

        if explain:
            from src.explainability import GradCAM, overlay_heatmap
            gradcam = GradCAM(self.cls_model, self.device)
            cam = gradcam.generate(tensor, pred_class)
            display = np.array(image_pil.resize((self.image_size, self.image_size)))
            result["gradcam_heatmap"] = overlay_heatmap(display, cam)

        return result
