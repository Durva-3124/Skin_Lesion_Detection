"""
scripts/train_segmentation.py
Train U-Net segmentation on ISIC 2018 Task 1 using fixed splits from data/splits/.

Usage:
    python scripts/train_segmentation.py --encoder vgg16 --epochs 30
    python scripts/train_segmentation.py --encoder mobilenetv2 --epochs 30
"""

import sys
import argparse
import pathlib
import json
import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, SPLITS_DIR
from src.segmentation.model import get_segmentation_model
from src.segmentation.metrics import combined_loss, dice_score, jaccard_score, pixel_accuracy
from src.preprocessing.transforms import get_train_transforms, get_val_transforms
from torchvision import transforms


class ISIC2018SegDataset(Dataset):
    """ISIC 2018 Task 1 segmentation dataset. Reads image IDs from a split CSV."""

    def __init__(self, split_csv: pathlib.Path, image_size: int = 256):
        seg_dir = DATA_DIR / "isic2018_seg"
        self.img_dir  = seg_dir / "ISIC2018_Task1-2_Training_Input"
        self.mask_dir = seg_dir / "ISIC2018_Task1_Training_GroundTruth"

        ids = pd.read_csv(split_csv)["image_id"].tolist()
        self.image_paths = [self.img_dir / f"{i}.jpg" for i in ids]
        self.mask_paths  = [self.mask_dir / f"{i}_segmentation.png" for i in ids]
        # Filter to pairs that exist on disk
        pairs = [(i, m) for i, m in zip(self.image_paths, self.mask_paths) if i.exists() and m.exists()]
        self.image_paths, self.mask_paths = zip(*pairs) if pairs else ([], [])

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx]).convert("L")
        return self.img_tf(image), (self.mask_tf(mask) > 0.5).float()


def evaluate(model, loader, device):
    model.eval()
    dice_scores, jacc_scores, pix_accs = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            dice_scores.append(dice_score(preds, masks))
            jacc_scores.append(jaccard_score(preds, masks))
            pix_accs.append(pixel_accuracy(preds, masks))
    return {
        "dice":     float(np.mean(dice_scores)),
        "jaccard":  float(np.mean(jacc_scores)),
        "pix_acc":  float(np.mean(pix_accs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",    default="vgg16", choices=["vgg16", "mobilenetv2"])
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Encoder: {args.encoder}")

    train_csv = SPLITS_DIR / "isic2018_seg_train.csv"
    val_csv   = SPLITS_DIR / "isic2018_seg_val.csv"
    if not train_csv.exists():
        sys.exit("ERROR: Split files not found. Run: python scripts/make_splits.py")

    train_ds = ISIC2018SegDataset(train_csv, args.image_size)
    val_ds   = ISIC2018SegDataset(val_csv,   args.image_size)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model     = get_segmentation_model(encoder=args.encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / f"unet_{args.encoder}.pth"

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = combined_loss(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        m = evaluate(model, val_loader, device)
        scheduler.step(train_loss / len(train_loader))
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={train_loss/len(train_loader):.4f} "
              f"| dice={m['dice']:.4f} | jaccard={m['jaccard']:.4f} | pix_acc={m['pix_acc']:.4f}")

        if m["dice"] > best_dice:
            best_dice = m["dice"]
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved (dice={best_dice:.4f})")

    print(f"\nBest Dice: {best_dice:.4f} | Weights: {save_path}")


if __name__ == "__main__":
    main()
