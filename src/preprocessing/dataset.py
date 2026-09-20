"""
src/preprocessing/dataset.py
Dataset classes for HAM10000 and ISIC 2019.
Computes class weights from actual downloaded data (not hardcoded literature numbers).
Auto-detects data paths for Kaggle, Colab, and local environments.
"""

import pathlib
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch

# --- Path resolution ---
_COLAB_DATA  = pathlib.Path("/content/data")
_KAGGLE_HAM  = pathlib.Path("/kaggle/input/datasets/kmader/skin-cancer-mnist-ham10000")
_KAGGLE_SEG  = pathlib.Path("/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation")
_KAGGLE_I19  = pathlib.Path("/kaggle/input/datasets/andrewmvd/isic-2019")
# _REPO_DATA: safe fallback that works whether src is run locally or loaded as a Kaggle dataset
_REPO_DATA   = (
    pathlib.Path("/kaggle/working/data")
    if pathlib.Path("/kaggle").exists()
    else pathlib.Path(__file__).parent.parent.parent / "data"
)

def _on_kaggle():
    return _KAGGLE_HAM.exists()

def _on_colab():
    return _COLAB_DATA.exists()

# HAM10000 label map
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM_CLASS_TO_IDX = {c: i for i, c in enumerate(HAM_CLASSES)}

# ISIC 2019 label map (8 classes — HAM7 + SCC, drop UNK)
ISIC19_CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
ISIC19_CLASS_TO_IDX = {c: i for i, c in enumerate(ISIC19_CLASSES)}


class HAM10000Dataset(Dataset):
    def __init__(self, split="train", transform=None, val_fraction=0.15, seed=42):
        if _on_kaggle():
            base = _KAGGLE_HAM
            meta = pd.read_csv(base / "HAM10000_metadata.csv")
            # Images split across two part folders on Kaggle
            self.image_dirs = [
                base / "HAM10000_images_part_1",
                base / "HAM10000_images_part_2",
            ]
        elif _on_colab():
            base = _COLAB_DATA / "ham10000"
            meta = pd.read_csv(base / "HAM10000_metadata.tab", sep="\t")
            self.image_dirs = [base]
        else:
            base = _REPO_DATA / "ham10000"
            meta = pd.read_csv(base / "HAM10000_metadata.tab", sep="\t")
            self.image_dirs = [base]

        meta = meta[meta["dx"].isin(HAM_CLASSES)].reset_index(drop=True)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(meta))
        val_n = int(len(meta) * val_fraction)
        val_idx = idx[:val_n]
        train_idx = idx[val_n:]

        self.meta = meta.iloc[train_idx if split == "train" else val_idx].reset_index(drop=True)
        self.transform = transform
        self.labels = [HAM_CLASS_TO_IDX[dx] for dx in self.meta["dx"]]

    def _find_image(self, image_id):
        for d in self.image_dirs:
            p = d / f"{image_id}.jpg"
            if p.exists():
                return p
        raise FileNotFoundError(f"{image_id}.jpg not found in {self.image_dirs}")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        image = Image.open(self._find_image(row["image_id"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights, normalized to num_classes mean=1 for stable loss scaling."""
        counts = np.bincount(self.labels, minlength=len(HAM_CLASSES)).astype(float)
        weights = 1.0 / np.where(counts == 0, 1, counts)
        weights = weights / weights.mean()  # scale so mean=1, not sum=1
        return torch.tensor(weights, dtype=torch.float32)

    def sample_weights(self) -> list:
        cw = self.class_weights().numpy()
        return [cw[label] for label in self.labels]


class ISIC2019Dataset(Dataset):
    def __init__(self, split="train", transform=None, val_fraction=0.15, seed=42):
        if _KAGGLE_I19.exists():
            gt = pd.read_csv(_KAGGLE_I19 / "ISIC_2019_Training_GroundTruth.csv")
            self.image_dir = _KAGGLE_I19 / "ISIC_2019_Training_Input" / "ISIC_2019_Training_Input"
        elif _on_colab():
            gt = pd.read_csv(_COLAB_DATA / "isic2019" / "ISIC_2019_Training_GroundTruth.csv")
            self.image_dir = _COLAB_DATA / "isic2019" / "ISIC_2019_Training_Input"
        else:
            gt = pd.read_csv(_REPO_DATA / "isic2019" / "ISIC_2019_Training_GroundTruth.csv")
            self.image_dir = _REPO_DATA / "isic2019" / "ISIC_2019_Training_Input"

        gt = gt[gt["UNK"] == 0].reset_index(drop=True)
        class_cols = [c for c in ISIC19_CLASSES if c in gt.columns]
        gt["label"] = gt[class_cols].values.argmax(axis=1)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(gt))
        val_n = int(len(gt) * val_fraction)
        val_idx = idx[:val_n]
        train_idx = idx[val_n:]

        self.meta = gt.iloc[train_idx if split == "train" else val_idx].reset_index(drop=True)
        self.transform = transform
        self.labels = self.meta["label"].tolist()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = self.image_dir / f"{row['image']}.jpg"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights, normalized to num_classes mean=1 for stable loss scaling."""
        counts = np.bincount(self.labels, minlength=len(ISIC19_CLASSES)).astype(float)
        weights = 1.0 / np.where(counts == 0, 1, counts)
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    def sample_weights(self) -> list:
        cw = self.class_weights().numpy()
        return [cw[label] for label in self.labels]


def get_dataloaders(dataset_name="ham10000", batch_size=32, image_size=224, num_workers=0):
    from src.preprocessing.transforms import get_train_transforms, get_val_transforms

    DatasetClass = HAM10000Dataset if dataset_name == "ham10000" else ISIC2019Dataset

    train_ds = DatasetClass(split="train", transform=get_train_transforms(image_size))
    val_ds   = DatasetClass(split="val",   transform=get_val_transforms(image_size))

    sampler = WeightedRandomSampler(
        weights=train_ds.sample_weights(),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,   num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,     num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.class_weights()
