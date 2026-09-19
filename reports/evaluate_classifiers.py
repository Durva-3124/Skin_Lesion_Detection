"""
reports/evaluate_classifiers.py
Per-class evaluation of trained classifiers on HAM10000 val split.
Saves full results to reports/eval_ham10000_<timestamp>.json
Run from repo root: python reports/evaluate_classifiers.py
"""

import sys, pathlib, json, datetime
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader

from src.classification.model import get_classification_model
from src.preprocessing.dataset import HAM10000Dataset, HAM_CLASSES
from src.preprocessing.transforms import get_val_transforms

MALIGNANT = {"mel", "bcc", "akiec"}
MODELS = {
    "efficientnet_b0": pathlib.Path("models/efficientnet_b0_ham10000.pth"),
    "swin_small":      pathlib.Path("models/swin_small_ham10000.pth"),
}


def evaluate(model_name: str, weights_path: pathlib.Path, device: torch.device) -> dict:
    model = get_classification_model(model_name, num_classes=7, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()

    val_ds = HAM10000Dataset(split="val", transform=get_val_transforms(224))
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
    acc = accuracy_score(all_labels, all_preds)
    report_str = classification_report(
        all_labels, all_preds,
        target_names=HAM_CLASSES,
        digits=4, zero_division=0
    )

    per_class = {}
    for i, cls in enumerate(HAM_CLASSES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        support = int(cm[i, :].sum())
        per_class[cls] = {
            "precision": round(tp / (tp + fp + 1e-8), 4),
            "recall":    round(tp / (tp + fn + 1e-8), 4),
            "f1":        round(2*tp / (2*tp + fp + fn + 1e-8), 4),
            "support":   support,
            "malignant": cls in MALIGNANT,
        }

    mal_recalls = [per_class[c]["recall"] for c in MALIGNANT]

    return {
        "model_name":        model_name,
        "weights_file":      str(weights_path.resolve()),
        "weights_size_mb":   round(weights_path.stat().st_size / 1e6, 2),
        "val_split":         "HAM10000Dataset(split='val', val_fraction=0.15, seed=42)",
        "val_samples":       len(all_labels),
        "preprocessing":     "get_val_transforms(224): Resize(224x224) + ToTensor + ImageNet Normalize",
        "overall_accuracy":  round(acc, 4),
        "per_class":         per_class,
        "malignant_recall":  {c: per_class[c]["recall"] for c in MALIGNANT},
        "mean_malignant_recall": round(float(np.mean(mal_recalls)), 4),
        "classification_report": report_str,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = pathlib.Path("reports") / f"eval_ham10000_{timestamp}.json"

    results = {
        "timestamp":    timestamp,
        "device":       str(device),
        "ham_classes":  HAM_CLASSES,
        "script":       "reports/evaluate_classifiers.py",
        "models":       {}
    }

    for model_name, weights_path in MODELS.items():
        if not weights_path.exists():
            print(f"SKIP {model_name} — {weights_path} not found")
            results["models"][model_name] = {"error": f"{weights_path} not found"}
            continue

        print(f"\nEvaluating {model_name} from {weights_path} ...")
        r = evaluate(model_name, weights_path, device)
        results["models"][model_name] = r

        print(r["classification_report"])
        print(f"Overall accuracy : {r['overall_accuracy']:.4f}")
        print(f"Malignant recall : mel={r['malignant_recall']['mel']:.4f}  "
              f"bcc={r['malignant_recall']['bcc']:.4f}  "
              f"akiec={r['malignant_recall']['akiec']:.4f}  "
              f"mean={r['mean_malignant_recall']:.4f}")

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_path}")

    # Side-by-side comparison
    if all(m in results["models"] and "error" not in results["models"][m]
           for m in ["efficientnet_b0", "swin_small"]):
        b0  = results["models"]["efficientnet_b0"]
        swn = results["models"]["swin_small"]
        print("\n" + "="*65)
        print("SIDE-BY-SIDE COMPARISON")
        print("="*65)
        print(f"{'Metric':<25} {'EfficientNet-B0':>16} {'Swin-Small':>12}")
        print("-"*65)
        print(f"{'overall_accuracy':<25} {b0['overall_accuracy']:>16.4f} {swn['overall_accuracy']:>12.4f}")
        for cls in HAM_CLASSES:
            tag = " [MAL]" if cls in MALIGNANT else ""
            print(f"{'recall_'+cls+tag:<25} {b0['per_class'][cls]['recall']:>16.4f} "
                  f"{swn['per_class'][cls]['recall']:>12.4f}")
        print(f"{'mean_malignant_recall':<25} {b0['mean_malignant_recall']:>16.4f} "
              f"{swn['mean_malignant_recall']:>12.4f}")
        print("="*65)
        print(f"\nCheckpoints used:")
        print(f"  EfficientNet-B0 : {b0['weights_file']}")
        print(f"  Swin-Small      : {swn['weights_file']}")


if __name__ == "__main__":
    main()
