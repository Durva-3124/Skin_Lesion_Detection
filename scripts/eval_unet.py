"""
scripts/eval_unet.py
Evaluates UNet-VGG16 on the ISIC 2018 segmentation val split.
Saves Dice score and per-image stats to reports/eval_unet.json.
Run on Kaggle with src/ and models/ datasets attached.
"""

import sys
import json
import pathlib
import datetime
# Kaggle notebook: __file__ is undefined; src/ is uploaded as a dataset
sys.path.insert(0, "/kaggle/input/datasets/durvapawar/tejalens-src")

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.segmentation.model import get_segmentation_model

_KAGGLE_SEG    = pathlib.Path("/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation")
_KAGGLE_MODELS = pathlib.Path("/kaggle/input/datasets/durvapawar/models/models")
_LOCAL_DATA    = pathlib.Path("/kaggle/tmp")
_LOCAL_MODELS  = pathlib.Path("/kaggle/tmp")

REPORT_PATH  = pathlib.Path("/kaggle/tmp/eval_unet.json")
IMAGE_SIZE   = 256
BATCH_SIZE   = 16
VAL_FRACTION = 0.15
SEED         = 42


class ISIC2018SegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=256):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.img_tf  = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
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


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, eps=1e-8) -> float:
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean().item()


def _resolve_paths():
    if _KAGGLE_SEG.exists():
        img_dir  = _KAGGLE_SEG / "ISIC2018_Task1-2_Training_Input"
        mask_dir = _KAGGLE_SEG / "ISIC2018_Task1_Training_GroundTruth"
        weights  = _KAGGLE_MODELS / "unet_vgg16.pth"
    else:
        img_dir  = _LOCAL_DATA / "ISIC2018_Task1-2_Training_Input"
        mask_dir = _LOCAL_DATA / "ISIC2018_Task1_Training_GroundTruth"
        weights  = _LOCAL_MODELS / "unet_vgg16.pth"
    return img_dir, mask_dir, weights


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    img_dir, mask_dir, weights_path = _resolve_paths()

    image_paths = sorted(img_dir.glob("*.jpg"))
    mask_paths  = [mask_dir / (p.stem + "_segmentation.png") for p in image_paths]
    # Filter to pairs that exist
    pairs = [(i, m) for i, m in zip(image_paths, mask_paths) if m.exists()]
    print(f"Found {len(pairs)} image/mask pairs")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pairs))
    val_n = int(len(pairs) * VAL_FRACTION)
    val_pairs = [pairs[i] for i in idx[:val_n]]

    val_imgs, val_masks = zip(*val_pairs)
    val_ds = ISIC2018SegDataset(list(val_imgs), list(val_masks), IMAGE_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = get_segmentation_model("vgg16").to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    dice_scores = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            batch_dice = dice_score(preds, masks)
            dice_scores.append(batch_dice)

    mean_dice = float(np.mean(dice_scores))
    print(f"Val Dice: {mean_dice:.4f}  (n={len(val_pairs)})")

    result = {
        "timestamp":    datetime.datetime.now().isoformat(),
        "device":       str(device),
        "model":        "unet_vgg16",
        "weights_file": str(weights_path),
        "val_samples":  len(val_pairs),
        "val_fraction": VAL_FRACTION,
        "seed":         SEED,
        "image_size":   IMAGE_SIZE,
        "mean_dice":    mean_dice,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2))
    print(f"Saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
