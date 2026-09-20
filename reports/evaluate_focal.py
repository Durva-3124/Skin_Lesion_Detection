"""
reports/evaluate_focal.py
Evaluates efficientnet_b0_ham10000_focal.pth against the four Module 4 targets.
Saves results to reports/eval_focal.json and moves weights to models/ on PASS.
Run from repo root: python reports/evaluate_focal.py
"""

import sys, pathlib, json, shutil, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader

from src.classification.model import get_classification_model
from src.preprocessing.dataset import HAM10000Dataset, HAM_CLASSES
from src.preprocessing.transforms import get_val_transforms

WEIGHTS_SRC  = pathlib.Path("efficientnet_b0_ham10000_focal.pth")
WEIGHTS_DEST = pathlib.Path("models/efficientnet_b0_ham10000_focal.pth")
REPORT_OUT   = pathlib.Path("reports/eval_focal.json")

TARGETS = {
    "mel":   0.85,
    "akiec": 0.78,
    "nv":    0.85,
}
ACC_TARGET = 0.75

BEFORE = {          # CE-loss baseline from previous run
    "mel":   0.794,
    "akiec": 0.704,
    "nv":    0.742,
    "acc":   0.769,
}


def main():
    if not WEIGHTS_SRC.exists():
        sys.exit(f"ERROR: {WEIGHTS_SRC} not found. Run from repo root.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = get_classification_model("efficientnet_b0", num_classes=7, pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS_SRC, map_location=device, weights_only=True))
    model.to(device).eval()

    val_ds = HAM10000Dataset(split="val", transform=get_val_transforms(224))
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm  = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
    acc = accuracy_score(all_labels, all_preds)

    recall = {}
    for i, cls in enumerate(HAM_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        recall[cls] = float(tp / (tp + fn + 1e-8))

    # Four-target verdict
    targets_met = {
        "mel":   recall["mel"]   >= TARGETS["mel"],
        "akiec": recall["akiec"] >= TARGETS["akiec"],
        "nv":    recall["nv"]    >= TARGETS["nv"],
        "acc":   acc             >= ACC_TARGET,
    }
    PASS = all(targets_met.values())

    # Print comparison table
    print("\n" + "=" * 62)
    print("FOUR-TARGET EVALUATION — EfficientNet-B0 Focal Loss")
    print("=" * 62)
    print(f"{'Metric':<20} {'Before (CE)':>12} {'After (Focal)':>14} {'Target':>10}")
    print("-" * 62)
    print(f"{'mel recall':<20} {BEFORE['mel']:>11.1%} {recall['mel']:>13.1%} {'>=85%':>10}  {'✓' if targets_met['mel'] else '✗'}")
    print(f"{'akiec recall':<20} {BEFORE['akiec']:>11.1%} {recall['akiec']:>13.1%} {'>=78%':>10}  {'✓' if targets_met['akiec'] else '✗'}")
    print(f"{'nv recall':<20} {BEFORE['nv']:>11.1%} {recall['nv']:>13.1%} {'>=85%':>10}  {'✓' if targets_met['nv'] else '✗'}")
    print(f"{'overall accuracy':<20} {BEFORE['acc']:>11.1%} {acc:>13.1%} {'~76%+':>10}  {'✓' if targets_met['acc'] else '✗'}")
    print("=" * 62)
    print(f"\nVerdict: {'✅ PASS — lock in as deployment classifier' if PASS else '❌ NEEDS TUNING — do not lock in yet'}")

    print("\nFull classification report:")
    print(classification_report(all_labels, all_preds, target_names=HAM_CLASSES, digits=4, zero_division=0))

    # Save JSON
    results = {
        "timestamp":    datetime.datetime.now().isoformat(),
        "weights_src":  str(WEIGHTS_SRC),
        "device":       str(device),
        "loss":         "focal",
        "overall_accuracy": round(acc, 4),
        "recall":       {k: round(v, 4) for k, v in recall.items()},
        "targets_met":  targets_met,
        "pass":         PASS,
        "before_ce":    BEFORE,
    }
    REPORT_OUT.parent.mkdir(exist_ok=True)
    REPORT_OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {REPORT_OUT}")

    # Move weights to models/ if PASS
    if PASS:
        WEIGHTS_DEST.parent.mkdir(exist_ok=True)
        shutil.move(str(WEIGHTS_SRC), str(WEIGHTS_DEST))
        print(f"Moved weights → {WEIGHTS_DEST}")
    else:
        print(f"Weights left at {WEIGHTS_SRC} — tune before promoting to models/")


if __name__ == "__main__":
    main()
