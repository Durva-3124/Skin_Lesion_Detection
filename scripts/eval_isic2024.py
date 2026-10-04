"""
scripts/eval_isic2024.py
Generalization test: EfficientNet-B0 (trained on HAM10000) evaluated on ISIC 2024 SLICE-3D.
ISIC 2024 has 2 classes: benign (0) / malignant (1).
We map HAM10000 predictions → binary using MALIGNANT_CLASSES.
Reports pAUC above 80% TPR, ROC AUC, sensitivity at 90% and 95% specificity.
Accuracy is not meaningful at ~1000:1 imbalance — not reported.
Saves results to reports/eval_isic2024.json.

Run on Kaggle with src/ and models/ datasets attached:
    python scripts/eval_isic2024.py
"""

import sys
import json
import pathlib
import datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report,
)

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.classification.model import get_classification_model
from src.preprocessing.transforms import get_val_transforms
from src.preprocessing.dataset import HAM_CLASSES

_KAGGLE_I24    = pathlib.Path("/kaggle/input/isic-2024-challenge")
_KAGGLE_MODELS = pathlib.Path("/kaggle/input/datasets/durvapawar/models/models")

IMAGE_SIZE    = 224
BATCH_SIZE    = 64

MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}
MALIGNANT_IDX     = {HAM_CLASSES.index(c) for c in MALIGNANT_CLASSES}


class ISIC2024Dataset(Dataset):
    """
    ISIC 2024 SLICE-3D binary dataset.
    CSV columns: isic_id, target (0=benign, 1=malignant).
    Images: jpg files under image_dir.
    """
    def __init__(self, csv_path: pathlib.Path, image_dir: pathlib.Path, transform):
        self.meta      = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row   = self.meta.iloc[idx]
        image = Image.open(self.image_dir / f"{row['isic_id']}.jpg").convert("RGB")
        label = int(row["target"])
        return self.transform(image), label


def _resolve_paths():
    if _KAGGLE_I24.exists():
        csv_path  = _KAGGLE_I24 / "train-metadata.csv"
        image_dir = _KAGGLE_I24 / "train-image" / "image"
        weights   = _KAGGLE_MODELS / "efficientnet_b0_ham10000.pth"
    else:
        csv_path  = DATA_DIR / "isic2024" / "train-metadata.csv"
        image_dir = DATA_DIR / "isic2024" / "train-image" / "image"
        weights   = MODELS_DIR / "efficientnet_b0_ham10000.pth"
    return csv_path, image_dir, weights


def _pauc_above_tpr(y_true, y_score, min_tpr: float = 0.80) -> float:
    """Partial AUC above min_tpr TPR, normalized to [0, 1]."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Keep only points where TPR >= min_tpr
    mask = tpr >= min_tpr
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(tpr[mask], fpr[mask]) / (1.0 - fpr[mask].min() + 1e-8))


def _sensitivity_at_specificity(y_true, y_score, target_specificity: float) -> float:
    """Sensitivity at a fixed specificity threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # specificity = 1 - fpr
    target_fpr = 1.0 - target_specificity
    idx = np.argmin(np.abs(fpr - target_fpr))
    return float(tpr[idx])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    csv_path, image_dir, weights_path = _resolve_paths()

    transform = get_val_transforms(IMAGE_SIZE)
    ds = ISIC2024Dataset(csv_path, image_dir, transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
    print(f"ISIC 2024 samples: {len(ds)}")

    model = get_classification_model("efficientnet_b0", num_classes=7)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()

    all_preds_binary, all_probs_malignant, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()  # (B, 7)
            # Binary prediction: malignant if argmax is in MALIGNANT_IDX
            preds_7class = probs.argmax(axis=1)
            preds_binary = np.isin(preds_7class, list(MALIGNANT_IDX)).astype(int)
            # Malignant probability = sum of malignant class probs
            prob_malignant = probs[:, list(MALIGNANT_IDX)].sum(axis=1)

            all_preds_binary.extend(preds_binary.tolist())
            all_probs_malignant.extend(prob_malignant.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_labels       = np.array(all_labels)
    all_preds_binary = np.array(all_preds_binary)
    all_probs_malignant = np.array(all_probs_malignant)

    auc    = roc_auc_score(all_labels, all_probs_malignant)
    pauc   = _pauc_above_tpr(all_labels, all_probs_malignant, min_tpr=0.80)
    sens90 = _sensitivity_at_specificity(all_labels, all_probs_malignant, target_specificity=0.90)
    sens95 = _sensitivity_at_specificity(all_labels, all_probs_malignant, target_specificity=0.95)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds_binary).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print(f"ROC AUC:                    {auc:.4f}")
    print(f"pAUC (TPR>=80%):            {pauc:.4f}")
    print(f"Sensitivity @ 90% spec:     {sens90:.4f}")
    print(f"Sensitivity @ 95% spec:     {sens95:.4f}")
    print(f"Sensitivity (argmax thresh): {sensitivity:.4f}")
    print(f"Specificity (argmax thresh): {specificity:.4f}")
    print("NOTE: Accuracy not reported — not meaningful at ~1000:1 imbalance.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp":        datetime.datetime.now().isoformat(),
        "device":           str(device),
        "model":            "efficientnet_b0_ham10000",
        "weights_file":     str(weights_path),
        "test_dataset":     "ISIC 2024 SLICE-3D",
        "test_samples":     len(ds),
        "task":             "binary (benign vs malignant)",
        "malignant_classes_used": list(MALIGNANT_CLASSES),
        "auc_roc":          round(float(auc), 4),
        "pauc_above_80tpr": round(float(pauc), 4),
        "sensitivity_at_90pct_specificity": round(float(sens90), 4),
        "sensitivity_at_95pct_specificity": round(float(sens95), 4),
        "sensitivity_argmax": round(float(sensitivity), 4),
        "specificity_argmax": round(float(specificity), 4),
        "note": "Accuracy not reported — not meaningful at ~1000:1 imbalance.",
    }
    out = REPORTS_DIR / "eval_isic2024.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
