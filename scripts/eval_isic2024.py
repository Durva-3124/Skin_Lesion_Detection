"""
scripts/eval_isic2024.py
Generalization test: EfficientNet-B0 (trained on HAM10000) evaluated on ISIC 2024 SLICE-3D.
ISIC 2024 has 2 classes: benign (0) / malignant (1).
We map HAM10000 predictions → binary using MALIGNANT_CLASSES.
Saves results to reports/eval_isic2024.json.
Run on Kaggle with src/ and models/ datasets attached.
"""

import sys
import json
import pathlib
import datetime
# Kaggle notebook: __file__ is undefined; src/ is uploaded as a dataset
sys.path.insert(0, "/kaggle/input/datasets/durvapawar/tejalens-src")

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

from src.classification.model import get_classification_model
from src.preprocessing.transforms import get_val_transforms
from src.preprocessing.dataset import HAM_CLASSES

_KAGGLE_I24    = pathlib.Path("/kaggle/input/isic-2024-challenge")
_KAGGLE_MODELS = pathlib.Path("/kaggle/input/datasets/durvapawar/models/models")

REPORT_PATH   = pathlib.Path("/kaggle/tmp/eval_isic2024.json")
IMAGE_SIZE    = 224
BATCH_SIZE    = 64

# HAM10000 classes that map to malignant
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
        raise RuntimeError("ISIC 2024 dataset not found. Run on Kaggle with isic-2024-challenge attached.")
    return csv_path, image_dir, weights


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
    model.load_state_dict(torch.load(weights_path, map_location=device))
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

    acc    = accuracy_score(all_labels, all_preds_binary)
    f1     = f1_score(all_labels, all_preds_binary, average="macro", zero_division=0)
    auc    = roc_auc_score(all_labels, all_probs_malignant)
    cm     = confusion_matrix(all_labels, all_preds_binary).tolist()
    report = classification_report(all_labels, all_preds_binary,
                                   target_names=["benign", "malignant"], zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds_binary).ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {f1:.4f}")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(report)

    result = {
        "timestamp":        datetime.datetime.now().isoformat(),
        "device":           str(device),
        "model":            "efficientnet_b0_ham10000",
        "weights_file":     str(weights_path),
        "test_dataset":     "ISIC 2024 SLICE-3D",
        "test_samples":     len(ds),
        "task":             "binary (benign vs malignant)",
        "malignant_classes_used": list(MALIGNANT_CLASSES),
        "accuracy":         acc,
        "macro_f1":         f1,
        "auc_roc":          auc,
        "sensitivity":      sensitivity,
        "specificity":      specificity,
        "confusion_matrix": cm,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
