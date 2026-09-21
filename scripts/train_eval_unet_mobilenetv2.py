"""
scripts/train_eval_unet_mobilenetv2.py
Train UNet-MobileNetV2 on ISIC 2018 Task 1 segmentation data.
Identical split to VGG16 baseline: val_fraction=0.15, seed=42, image_size=256.
Saves checkpoint to models/unet_mobilenetv2.pth and metrics to
reports/eval_unet_mobilenetv2.json.

Run on Kaggle with src/ and models/ datasets attached:
    python scripts/train_eval_unet_mobilenetv2.py

NOTE: Full 30-epoch training requires GPU (Kaggle T4/P100).
Architecture verified locally: forward pass, loss, and gradients all correct.
"""

import sys
import json
import pathlib
import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.segmentation.model import get_segmentation_model

ROOT       = pathlib.Path(__file__).parent.parent

# Path resolution: Kaggle vs local
_KAGGLE_SEG    = pathlib.Path("/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation")
_KAGGLE_MODELS = pathlib.Path("/kaggle/working")

if _KAGGLE_SEG.exists():
    DATA_DIR  = _KAGGLE_SEG
    IMG_DIR   = DATA_DIR / "ISIC2018_Task1-2_Training_Input"
    MASK_DIR  = DATA_DIR / "ISIC2018_Task1_Training_GroundTruth"
    SAVE_PATH = _KAGGLE_MODELS / "unet_mobilenetv2.pth"
    REPORT_PATH = pathlib.Path("/kaggle/tmp/eval_unet_mobilenetv2.json")
else:
    DATA_DIR  = ROOT / "data" / "isic2018_seg"
    IMG_DIR   = DATA_DIR / "ISIC2018_Task1-2_Training_Input"
    MASK_DIR  = DATA_DIR / "ISIC2018_Task1_Training_GroundTruth"
    SAVE_PATH = ROOT / "models" / "unet_mobilenetv2.pth"
    REPORT_PATH = ROOT / "reports" / "eval_unet_mobilenetv2.json"

IMAGE_SIZE   = 256
BATCH_SIZE   = 8
EPOCHS       = 30
LR           = 1e-4
VAL_FRACTION = 0.15
SEED         = 42


class ISIC2018SegDataset(Dataset):
    def __init__(self, image_paths: list, mask_paths: list, augment: bool = False):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.augment     = augment
        self.img_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx]).convert("L")
        if self.augment:
            import random
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask  = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask  = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return self.img_tf(image), (self.mask_tf(mask) > 0.5).float()


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Batch-mean Dice coefficient from raw logits."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) /
            (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean().item()


def jaccard_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Batch-mean Jaccard (IoU) from raw logits."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def pixel_accuracy(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    """Batch-mean pixel accuracy from raw logits."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    correct = (pred == target).float().sum(dim=(1, 2, 3))
    total   = torch.tensor(pred.shape[2] * pred.shape[3], dtype=torch.float32)
    return (correct / total).mean().item()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Returns mean Dice, Jaccard, and pixel accuracy over the val set."""
    model.eval()
    dice_scores, jacc_scores, px_scores = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            dice_scores.append(dice_score(preds, masks))
            jacc_scores.append(jaccard_score(preds, masks))
            px_scores.append(pixel_accuracy(preds, masks))
    return {
        "dice":            float(np.mean(dice_scores)),
        "jaccard":         float(np.mean(jacc_scores)),
        "pixel_accuracy":  float(np.mean(px_scores)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build image/mask pairs — identical logic to eval_unet.py
    image_paths = sorted(IMG_DIR.glob("*.jpg"))
    mask_paths  = [MASK_DIR / (p.stem + "_segmentation.png") for p in image_paths]
    pairs = [(i, m) for i, m in zip(image_paths, mask_paths) if m.exists()]
    print(f"Total pairs: {len(pairs)}")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pairs))
    val_n      = int(len(pairs) * VAL_FRACTION)
    val_pairs  = [pairs[i] for i in idx[:val_n]]
    train_pairs = [pairs[i] for i in idx[val_n:]]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    train_imgs, train_masks = zip(*train_pairs)
    val_imgs,   val_masks   = zip(*val_pairs)

    train_ds = ISIC2018SegDataset(list(train_imgs), list(train_masks), augment=True)
    val_ds   = ISIC2018SegDataset(list(val_imgs),   list(val_masks),   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = get_segmentation_model("mobilenetv2").to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_dice = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        m = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"loss={train_loss/len(train_loader):.4f} | "
            f"dice={m['dice']:.4f} | jaccard={m['jaccard']:.4f} | "
            f"px_acc={m['pixel_accuracy']:.4f}"
        )

        if m["dice"] > best_dice:
            best_dice = m["dice"]
            torch.save(model.state_dict(), str(SAVE_PATH))
            print(f"  Saved (dice={best_dice:.4f})")

    # Final eval on best checkpoint
    model.load_state_dict(torch.load(str(SAVE_PATH), map_location=device, weights_only=True))
    m = evaluate(model, val_loader, device)
    print(f"\nFinal val — Dice={m['dice']:.4f}  Jaccard={m['jaccard']:.4f}  PixAcc={m['pixel_accuracy']:.4f}")

    result = {
        "timestamp":      datetime.datetime.now().isoformat(),
        "device":         str(device),
        "model":          "unet_mobilenetv2",
        "weights_file":   str(SAVE_PATH),
        "val_samples":    len(val_pairs),
        "train_samples":  len(train_pairs),
        "val_fraction":   VAL_FRACTION,
        "seed":           SEED,
        "image_size":     IMAGE_SIZE,
        "epochs":         EPOCHS,
        "lr":             LR,
        "mean_dice":      m["dice"],
        "mean_jaccard":   m["jaccard"],
        "mean_pixel_accuracy": m["pixel_accuracy"],
    }
    REPORT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Saved report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
