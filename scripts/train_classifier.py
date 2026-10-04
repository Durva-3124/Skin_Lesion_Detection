"""
scripts/train_classifier.py
Train EfficientNet-B0 or Swin-Small on HAM10000 or ISIC 2019.
Saves checkpoint, run_config.json, and per-epoch CSV.

Usage:
    python scripts/train_classifier.py --model efficientnet_b0 --dataset ham10000 \
        --loss ce --imbalance sampler --epochs 30 --lr 1e-4
"""

import sys
import argparse
import pathlib
import json
import csv
import datetime
import hashlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, SPLITS_DIR
from src.classification.model import get_classification_model
from src.preprocessing.dataset import get_dataloaders, HAM_CLASSES


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
    }


def _split_hash(splits_dir: pathlib.Path, dataset: str) -> dict:
    hashes = {}
    for split in ("train", "val", "test"):
        p = splits_dir / f"{dataset}_{split}.csv"
        if p.exists():
            hashes[split] = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="efficientnet_b0", choices=["efficientnet_b0", "swin_small", "efficientformerv2"])
    parser.add_argument("--dataset",   default="ham10000",         choices=["ham10000", "isic2019"])
    parser.add_argument("--loss",      default="ce",               choices=["ce", "focal"])
    parser.add_argument("--imbalance", default="sampler",          choices=["sampler", "weights", "both", "none"])
    parser.add_argument("--epochs",    type=int,   default=30)
    parser.add_argument("--lr",        type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int,  default=32)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--amp",       action="store_true", help="Use mixed precision (CUDA only)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | Dataset: {args.dataset} | Loss: {args.loss} | Imbalance: {args.imbalance}")

    num_classes = 7 if args.dataset == "ham10000" else 8
    train_loader, val_loader, class_weights = get_dataloaders(
        args.dataset, args.batch_size, imbalance=args.imbalance
    )
    class_weights = class_weights.to(device)

    use_weights_in_loss = args.imbalance in ("weights", "both")
    loss_weight = class_weights if use_weights_in_loss else None

    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0, weight=loss_weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=args.label_smoothing)

    model = get_classification_model(args.model, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = MODELS_DIR / f"{args.model}_{args.dataset}.pth"
    csv_path  = REPORTS_DIR / f"train_{args.model}_{args.dataset}_{ts}.csv"

    # Save run config
    try:
        import subprocess
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           cwd=str(pathlib.Path(__file__).parent.parent),
                                           stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "unknown"

    run_config = {
        "timestamp": ts,
        "model": args.model, "dataset": args.dataset,
        "loss": args.loss, "imbalance": args.imbalance,
        "epochs": args.epochs, "lr": args.lr,
        "label_smoothing": args.label_smoothing,
        "batch_size": args.batch_size, "seed": args.seed,
        "amp": args.amp,
        "torch_version": torch.__version__,
        "git_hash": git_hash,
        "split_hashes": _split_hash(SPLITS_DIR, args.dataset),
    }
    (MODELS_DIR / f"run_config_{args.model}_{args.dataset}_{ts}.json").write_text(
        json.dumps(run_config, indent=2)
    )

    best_f1 = 0.0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_accuracy", "val_macro_f1"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        loss = criterion(model(images), labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()

            scheduler.step()
            m = evaluate(model, val_loader, device, num_classes)
            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | acc={m['accuracy']:.4f} | f1={m['macro_f1']:.4f}")
            writer.writerow([epoch, round(avg_loss, 4), round(m["accuracy"], 4), round(m["macro_f1"], 4)])

            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Saved (f1={best_f1:.4f})")

    print(f"\nBest macro F1: {best_f1:.4f} | Weights: {save_path}")
    print(f"Per-epoch CSV: {csv_path}")


if __name__ == "__main__":
    main()
