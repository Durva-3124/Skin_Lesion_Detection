"""
scripts/eval_classifier.py
Evaluate a trained classifier on the held-out test split.
Reports per-class metrics, balanced accuracy, AUC, binary malignant sensitivity/specificity.
Saves to reports/eval_<model>_<dataset>.json.

Usage:
    python scripts/eval_classifier.py --model efficientnet_b0 \
        --weights models/efficientnet_b0_ham10000.pth
"""

import sys
import argparse
import pathlib
import json
import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
)
from torch.utils.data import DataLoader

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, SPLITS_DIR
from src.classification.model import get_classification_model
from src.preprocessing.dataset import HAM10000Dataset, HAM_CLASSES
from src.preprocessing.transforms import get_val_transforms

MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}
MALIGNANT_IDX     = {HAM_CLASSES.index(c) for c in MALIGNANT_CLASSES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="efficientnet_b0",
                        choices=["efficientnet_b0", "swin_small", "efficientformerv2"])
    parser.add_argument("--dataset",    default="ham10000", choices=["ham10000"])
    parser.add_argument("--weights",    required=True, help="Path to .pth checkpoint")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    weights_path = pathlib.Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"ERROR: Weights not found at {weights_path}")

    test_csv = SPLITS_DIR / f"{args.dataset}_test.csv"
    if not test_csv.exists():
        sys.exit("ERROR: Split files not found. Run: python scripts/make_splits.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | Weights: {weights_path}")

    num_classes = 7
    model = get_classification_model(args.model, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()

    test_ds = HAM10000Dataset(split="test", transform=get_val_transforms(224))
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # Per-class recall from confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    recall = {}
    for i, cls in enumerate(HAM_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        recall[cls] = float(tp / (tp + fn + 1e-8))

    # Binary malignant vs benign
    binary_labels = np.isin(all_labels, list(MALIGNANT_IDX)).astype(int)
    binary_preds  = np.isin(all_preds,  list(MALIGNANT_IDX)).astype(int)
    prob_malignant = all_probs[:, list(MALIGNANT_IDX)].sum(axis=1)

    tn, fp, fn_b, tp_b = confusion_matrix(binary_labels, binary_preds, labels=[0, 1]).ravel()
    sensitivity = float(tp_b / (tp_b + fn_b + 1e-8))
    specificity = float(tn   / (tn   + fp   + 1e-8))

    try:
        auc_ovr = float(roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro"))
    except Exception:
        auc_ovr = None

    try:
        binary_auc = float(roc_auc_score(binary_labels, prob_malignant))
    except Exception:
        binary_auc = None

    results = {
        "timestamp":          datetime.datetime.now().isoformat(),
        "model":              args.model,
        "weights":            str(weights_path),
        "split":              "test",
        "test_samples":       len(test_ds),
        "accuracy":           round(float(accuracy_score(all_labels, all_preds)), 4),
        "balanced_accuracy":  round(float(balanced_accuracy_score(all_labels, all_preds)), 4),
        "macro_f1":           round(float(f1_score(all_labels, all_preds, average="macro", zero_division=0)), 4),
        "auc_ovr_macro":      round(auc_ovr, 4) if auc_ovr else "not computed",
        "per_class_recall":   {k: round(v, 4) for k, v in recall.items()},
        "binary_malignant": {
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "auc":         round(binary_auc, 4) if binary_auc else "not computed",
        },
    }

    print(f"\nAccuracy:          {results['accuracy']:.4f}")
    print(f"Balanced accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {results['macro_f1']:.4f}")
    print(f"Malignant sensitivity: {sensitivity:.4f} | specificity: {specificity:.4f}")
    print("\nPer-class recall:")
    for cls, r in recall.items():
        print(f"  {cls:6s}: {r:.4f}")
    print("\n" + classification_report(all_labels, all_preds, target_names=HAM_CLASSES, zero_division=0))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"eval_{args.model}_{args.dataset}_{ts}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
