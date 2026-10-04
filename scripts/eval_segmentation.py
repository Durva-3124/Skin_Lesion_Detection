"""
scripts/eval_segmentation.py
Evaluate U-Net segmentation on the held-out test split (data/splits/isic2018_seg_test.csv).
Reports Dice, Jaccard, pixel accuracy. Saves to reports/eval_unet_<encoder>.json.

Usage:
    python scripts/eval_segmentation.py --encoder vgg16
    python scripts/eval_segmentation.py --encoder mobilenetv2
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
from torchvision import transforms

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, SPLITS_DIR
from src.segmentation.model import get_segmentation_model
from src.segmentation.metrics import dice_score, jaccard_score, pixel_accuracy


class ISIC2018SegDataset(Dataset):
    def __init__(self, split_csv: pathlib.Path, image_size: int = 256):
        seg_dir = DATA_DIR / "isic2018_seg"
        img_dir  = seg_dir / "ISIC2018_Task1-2_Training_Input"
        mask_dir = seg_dir / "ISIC2018_Task1_Training_GroundTruth"

        ids = pd.read_csv(split_csv)["image_id"].tolist()
        pairs = [
            (img_dir / f"{i}.jpg", mask_dir / f"{i}_segmentation.png")
            for i in ids
            if (img_dir / f"{i}.jpg").exists() and (mask_dir / f"{i}_segmentation.png").exists()
        ]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",    default="vgg16", choices=["vgg16", "mobilenetv2"])
    parser.add_argument("--weights",    default=None, help="Path to .pth file (default: models/unet_<encoder>.pth)")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    weights_path = pathlib.Path(args.weights) if args.weights else MODELS_DIR / f"unet_{args.encoder}.pth"
    if not weights_path.exists():
        sys.exit(f"ERROR: Weights not found at {weights_path}")

    test_csv = SPLITS_DIR / "isic2018_seg_test.csv"
    if not test_csv.exists():
        sys.exit("ERROR: Split files not found. Run: python scripts/make_splits.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Encoder: {args.encoder} | Weights: {weights_path}")

    test_ds = ISIC2018SegDataset(test_csv, args.image_size)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Test samples: {len(test_ds)}")

    model = get_segmentation_model(encoder=args.encoder)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device).eval()

    dice_scores, jacc_scores, pix_accs = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            dice_scores.append(dice_score(preds, masks))
            jacc_scores.append(jaccard_score(preds, masks))
            pix_accs.append(pixel_accuracy(preds, masks))

    results = {
        "timestamp":    datetime.datetime.now().isoformat(),
        "encoder":      args.encoder,
        "weights":      str(weights_path),
        "split":        "test",
        "split_file":   str(test_csv),
        "test_samples": len(test_ds),
        "image_size":   args.image_size,
        "dice":         round(float(np.mean(dice_scores)), 4),
        "jaccard":      round(float(np.mean(jacc_scores)), 4),
        "pixel_accuracy": round(float(np.mean(pix_accs)), 4),
    }

    print(f"\nDice:           {results['dice']:.4f}")
    print(f"Jaccard (IoU):  {results['jaccard']:.4f}")
    print(f"Pixel accuracy: {results['pixel_accuracy']:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"eval_unet_{args.encoder}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
