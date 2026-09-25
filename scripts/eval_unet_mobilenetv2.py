"""
scripts/eval_unet_mobilenetv2.py
Evaluate the trained UNet-MobileNetV2 checkpoint on the ISIC 2018 validation split.
Saves comparable Dice, Jaccard, and pixel-accuracy metrics to reports/eval_unet_mobilenetv2.json.
"""

import datetime
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.segmentation.model import get_segmentation_model
from scripts.train_eval_unet_mobilenetv2 import ISIC2018SegDataset


ROOT = pathlib.Path(__file__).parent.parent
IMG_DIR = ROOT / "data" / "isic2018_seg" / "ISIC2018_Task1-2_Training_Input"
MASK_DIR = ROOT / "data" / "isic2018_seg" / "ISIC2018_Task1_Training_GroundTruth"
WEIGHTS_PATH = ROOT / "models" / "unet_mobilenetv2.pth"
REPORT_PATH = ROOT / "reports" / "eval_unet_mobilenetv2.json"
IMAGE_SIZE = 256
BATCH_SIZE = 16
VAL_FRACTION = 0.15
SEED = 42


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Return the batch-mean Dice coefficient at a 0.5 sigmoid threshold."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) /
            (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean().item()


def jaccard_score(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Return the batch-mean Jaccard score at a 0.5 sigmoid threshold."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def pixel_accuracy(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
    """Return the batch-mean pixel accuracy at a 0.5 sigmoid threshold."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    correct = (pred == target).float().sum(dim=(1, 2, 3))
    total = torch.tensor(pred.shape[2] * pred.shape[3], dtype=torch.float32)
    return (correct / total).mean().item()


def main() -> None:
    """Evaluate MobileNetV2 on the deterministic VGG16-comparable validation split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = sorted(IMG_DIR.glob("*.jpg"))
    mask_paths = [MASK_DIR / f"{path.stem}_segmentation.png" for path in image_paths]
    pairs = [(image, mask) for image, mask in zip(image_paths, mask_paths) if mask.exists()]

    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(pairs))
    val_count = int(len(pairs) * VAL_FRACTION)
    val_pairs = [pairs[index] for index in indices[:val_count]]
    val_images, val_masks = zip(*val_pairs)

    val_dataset = ISIC2018SegDataset(list(val_images), list(val_masks), augment=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_segmentation_model("mobilenetv2").to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model.eval()

    dice_scores = []
    jaccard_scores = []
    pixel_scores = []
    with torch.no_grad():
        for images, masks in val_loader:
            predictions = model(images.to(device))
            targets = masks.to(device)
            dice_scores.append(dice_score(predictions, targets))
            jaccard_scores.append(jaccard_score(predictions, targets))
            pixel_scores.append(pixel_accuracy(predictions, targets))

    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "device": str(device),
        "model": "unet_mobilenetv2",
        "weights_file": str(WEIGHTS_PATH),
        "val_samples": len(val_pairs),
        "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "mean_dice": float(np.mean(dice_scores)),
        "mean_jaccard": float(np.mean(jaccard_scores)),
        "mean_pixel_accuracy": float(np.mean(pixel_scores)),
    }
    REPORT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()