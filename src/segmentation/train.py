"""
src/segmentation/train.py
Trains U-Net segmentation model on ISIC 2018 Task 1.
Loss: BCE + Dice combined.
"""

import pathlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from src.segmentation.model import get_segmentation_model

DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data"
_colab = pathlib.Path('/content/data')
if _colab.exists():
    DATA_DIR = _colab
SEG_DIR = DATA_DIR / "isic2018_seg"
IMG_DIR = SEG_DIR / "ISIC2018_Task1-2_Training_Input"
MASK_DIR = SEG_DIR / "ISIC2018_Task1_Training_GroundTruth"


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=256):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
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
        mask = Image.open(self.mask_paths[idx]).convert("L")
        return self.img_tf(image), self.mask_tf(mask)


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def combined_loss(pred, target):
    return nn.BCEWithLogitsLoss()(pred, target) + dice_loss(pred, target)


def dice_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2 * intersection + 1) / (pred.sum() + target.sum() + 1)


def get_seg_dataloaders(image_size=256, batch_size=8, val_fraction=0.15, seed=42):
    images = sorted(IMG_DIR.glob("*.jpg"))
    masks = [MASK_DIR / (p.stem + "_segmentation.png") for p in images]
    pairs = [(i, m) for i, m in zip(images, masks) if m.exists()]

    val_n = int(len(pairs) * val_fraction)
    train_n = len(pairs) - val_n
    train_pairs, val_pairs = random_split(pairs, [train_n, val_n],
                                          generator=torch.Generator().manual_seed(seed))

    train_ds = SegmentationDataset([p[0] for p in train_pairs], [p[1] for p in train_pairs], image_size)
    val_ds = SegmentationDataset([p[0] for p in val_pairs], [p[1] for p in val_pairs], image_size)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
    )


def train(encoder="vgg16", epochs=30, lr=1e-4, image_size=256, batch_size=8, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Encoder: {encoder}")

    model = get_segmentation_model(encoder=encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_loader, val_loader = get_seg_dataloaders(image_size=image_size, batch_size=batch_size)

    best_dice = 0.0
    save_path = pathlib.Path(save_path or f"unet_{encoder}.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = combined_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, val_dice = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_loss += combined_loss(preds, masks).item()
                val_dice += dice_score(preds, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss/len(train_loader):.4f} "
              f"| val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (dice={best_dice:.4f})")

    print(f"\nBest Dice: {best_dice:.4f}")
    return model


if __name__ == "__main__":
    train(encoder="vgg16", epochs=30)
