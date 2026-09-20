"""
scripts/retrain_swin_small.py
Retrain Swin-Small on HAM10000 with lr=1e-5 and 5-epoch linear warmup.
Fixes nv recall collapse (0.2776) caused by lr=1e-4 overshooting.
Run on Kaggle with src/ and models/ datasets attached.
"""

import sys
import pathlib
# Kaggle notebook: __file__ is undefined; src/ is uploaded as a dataset
sys.path.insert(0, "/kaggle/input/datasets/durvapawar/tejalens-src")

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.classification.model import get_classification_model
from src.preprocessing.dataset import get_dataloaders, HAM_CLASSES

SAVE_PATH     = "/kaggle/working/swin_small_ham10000_v2.pth"
REPORT_PATH   = pathlib.Path("/kaggle/tmp/eval_swin_v2.json")
EPOCHS        = 40
LR            = 1e-5
WARMUP_EPOCHS = 5
BATCH_SIZE    = 32


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    nv_mask = np.array(all_labels) == HAM_CLASSES.index("nv")
    return {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "macro_f1":  f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "nv_recall": (np.array(all_preds)[nv_mask] == HAM_CLASSES.index("nv")).mean(),
        "report":    classification_report(all_labels, all_preds,
                                           target_names=HAM_CLASSES, zero_division=0),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, class_weights = get_dataloaders("ham10000", BATCH_SIZE)
    class_weights = class_weights.to(device)

    model = get_classification_model("swin_small", num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR / 10, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS
    )

    best_f1 = 0.0
    for epoch in range(EPOCHS):
        # Linear warmup: ramp lr from LR/10 to LR over first WARMUP_EPOCHS
        if epoch < WARMUP_EPOCHS:
            for pg in optimizer.param_groups:
                pg["lr"] = LR / 10 + (LR - LR / 10) * (epoch + 1) / WARMUP_EPOCHS

        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        m = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | loss={train_loss/len(train_loader):.4f} "
            f"| acc={m['accuracy']:.4f} | f1={m['macro_f1']:.4f} | nv_recall={m['nv_recall']:.4f}"
        )

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved (f1={best_f1:.4f})")

    import json, datetime
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    m = evaluate(model, val_loader, device)
    print(f"\nBest macro F1: {best_f1:.4f}")
    print(m["report"])
    REPORT_PATH.write_text(json.dumps({
        "timestamp": datetime.datetime.now().isoformat(),
        "model": "swin_small_v2",
        "epochs": EPOCHS, "lr": LR, "warmup_epochs": WARMUP_EPOCHS,
        "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "nv_recall": float(m["nv_recall"]),
    }, indent=2))
    print(f"Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
