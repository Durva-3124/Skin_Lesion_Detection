"""
reports/evaluate_classifiers.py
Per-class evaluation of trained classifiers on HAM10000 val split.
Reports recall on malignant classes (mel, bcc, akiec) explicitly.
Run from repo root: python reports/evaluate_classifiers.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
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
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
    report = classification_report(
        all_labels, all_preds,
        target_names=HAM_CLASSES,
        digits=4, zero_division=0
    )

    # Per-class recall (sensitivity)
    per_class_recall = {}
    for i, cls in enumerate(HAM_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        per_class_recall[cls] = tp / (tp + fn + 1e-8)

    return {"report": report, "recall": per_class_recall, "cm": cm}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print(f"HAM_CLASSES order: {HAM_CLASSES}")
    print(f"Malignant classes: {MALIGNANT}\n")
    print("=" * 70)

    results = {}
    for model_name, weights_path in MODELS.items():
        if not weights_path.exists():
            print(f"SKIP {model_name} — {weights_path} not found\n")
            continue

        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}  |  Weights: {weights_path}")
        print("=" * 70)

        r = evaluate(model_name, weights_path, device)
        results[model_name] = r

        print(r["report"])

        print("Per-class recall:")
        for cls, recall in r["recall"].items():
            tag = "[MALIGNANT]" if cls in MALIGNANT else ""
            print(f"  {cls:6s}: {recall:.4f}  {tag}")

        print("\nMalignant class recall summary:")
        mal_recalls = [r["recall"][c] for c in MALIGNANT]
        print(f"  mel   : {r['recall']['mel']:.4f}")
        print(f"  bcc   : {r['recall']['bcc']:.4f}")
        print(f"  akiec : {r['recall']['akiec']:.4f}")
        print(f"  mean  : {np.mean(mal_recalls):.4f}")

    # Side-by-side malignant recall comparison
    if len(results) == 2:
        print("\n" + "=" * 70)
        print("MALIGNANT RECALL COMPARISON")
        print("=" * 70)
        print(f"{'Class':<10} {'EfficientNet-B0':>16} {'Swin-Small':>12}")
        for cls in MALIGNANT:
            b0  = results["efficientnet_b0"]["recall"][cls]
            swn = results["swin_small"]["recall"][cls]
            print(f"{cls:<10} {b0:>16.4f} {swn:>12.4f}")
        b0_mean  = np.mean([results["efficientnet_b0"]["recall"][c] for c in MALIGNANT])
        swn_mean = np.mean([results["swin_small"]["recall"][c] for c in MALIGNANT])
        print(f"{'mean':<10} {b0_mean:>16.4f} {swn_mean:>12.4f}")
        print("\nLiterature target (Module 2): sensitivity 89–97% on malignant classes")
        print(f"EfficientNet-B0 mean malignant recall: {b0_mean:.1%}")
        print(f"Swin-Small      mean malignant recall: {swn_mean:.1%}")


if __name__ == "__main__":
    main()
