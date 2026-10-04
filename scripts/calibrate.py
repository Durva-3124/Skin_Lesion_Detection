"""
scripts/calibrate.py
Temperature scaling calibration + malignant-probability threshold selection.
Fitted on val split, evaluated on test split.
Saves models/calibration.json (temperature, threshold, val ECE, test sensitivity/specificity).

Usage:
    python scripts/calibrate.py --weights models/efficientnet_b0_ham10000.pth \
        --target-sensitivity 0.95
"""

import sys
import argparse
import pathlib
import json
import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.config import MODELS_DIR, REPORTS_DIR
from src.classification.model import get_classification_model
from src.preprocessing.dataset import HAM10000Dataset, HAM_CLASSES
from src.preprocessing.transforms import get_val_transforms

MALIGNANT_IDX = {HAM_CLASSES.index(c) for c in {"mel", "bcc", "akiec"}}


def _collect_logits(model, loader, device):
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            ece += mask.sum() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece / len(labels))


def _find_threshold(probs_malignant: np.ndarray, binary_labels: np.ndarray, target_sensitivity: float) -> float:
    """Find the lowest threshold that achieves target sensitivity on val."""
    best_thresh = 0.5
    for thresh in np.linspace(0.01, 0.99, 200):
        preds = (probs_malignant >= thresh).astype(int)
        tp = ((preds == 1) & (binary_labels == 1)).sum()
        fn = ((preds == 0) & (binary_labels == 1)).sum()
        sens = tp / (tp + fn + 1e-8)
        if sens >= target_sensitivity:
            best_thresh = float(thresh)
            break
    return best_thresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",            required=True)
    parser.add_argument("--target-sensitivity", type=float, default=0.95)
    parser.add_argument("--batch-size",         type=int,   default=32)
    args = parser.parse_args()

    weights_path = pathlib.Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"ERROR: Weights not found at {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = get_classification_model("efficientnet_b0", num_classes=7, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()

    val_ds   = HAM10000Dataset(split="val",  transform=get_val_transforms(224))
    test_ds  = HAM10000Dataset(split="test", transform=get_val_transforms(224))
    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Temperature scaling on val ---
    val_logits, val_labels = _collect_logits(model, val_loader, device)
    temperature = nn.Parameter(torch.ones(1))
    optimizer   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    criterion   = nn.CrossEntropyLoss()

    def _step():
        optimizer.zero_grad()
        loss = criterion(val_logits / temperature, val_labels)
        loss.backward()
        return loss

    optimizer.step(_step)
    T = float(temperature.item())
    print(f"Learned temperature: {T:.4f}")

    # ECE before/after on val
    raw_probs  = torch.softmax(val_logits, dim=1).numpy()
    cal_probs  = torch.softmax(val_logits / temperature.detach(), dim=1).numpy()
    labels_np  = val_labels.numpy()
    ece_before = _ece(raw_probs, labels_np)
    ece_after  = _ece(cal_probs, labels_np)
    print(f"Val ECE before: {ece_before:.4f} | after: {ece_after:.4f}")

    # --- Threshold selection on val ---
    prob_mal_val   = cal_probs[:, list(MALIGNANT_IDX)].sum(axis=1)
    binary_val     = np.isin(labels_np, list(MALIGNANT_IDX)).astype(int)
    threshold      = _find_threshold(prob_mal_val, binary_val, args.target_sensitivity)
    print(f"Threshold for sensitivity≥{args.target_sensitivity}: {threshold:.4f}")

    # --- Evaluate on test ---
    test_logits, test_labels = _collect_logits(model, test_loader, device)
    test_cal_probs = torch.softmax(test_logits / temperature.detach(), dim=1).numpy()
    prob_mal_test  = test_cal_probs[:, list(MALIGNANT_IDX)].sum(axis=1)
    binary_test    = np.isin(test_labels.numpy(), list(MALIGNANT_IDX)).astype(int)
    test_preds     = (prob_mal_test >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(binary_test, test_preds, labels=[0, 1]).ravel()
    test_sensitivity = float(tp / (tp + fn + 1e-8))
    test_specificity = float(tn / (tn + fp + 1e-8))
    print(f"Test sensitivity: {test_sensitivity:.4f} | specificity: {test_specificity:.4f}")

    calibration = {
        "timestamp":          datetime.datetime.now().isoformat(),
        "weights":            str(weights_path),
        "temperature":        round(T, 6),
        "threshold":          round(threshold, 4),
        "target_sensitivity": args.target_sensitivity,
        "val_ece_before":     round(ece_before, 4),
        "val_ece_after":      round(ece_after, 4),
        "test_sensitivity":   round(test_sensitivity, 4),
        "test_specificity":   round(test_specificity, 4),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "calibration.json"
    out.write_text(json.dumps(calibration, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
