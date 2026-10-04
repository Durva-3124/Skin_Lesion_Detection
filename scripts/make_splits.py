"""
scripts/make_splits.py
Creates fixed, reproducible split CSV files for all datasets.

HAM10000: group split by lesion_id (stratified by dx) → train/val/test 70/15/15
ISIC 2018 seg: random split by image id → train/val/test 70/15/15

Writes to data/splits/:
  ham10000_train.csv, ham10000_val.csv, ham10000_test.csv
  isic2018_seg_train.csv, isic2018_seg_val.csv, isic2018_seg_test.csv

Usage:
    python scripts/make_splits.py
"""

import sys
import pathlib
import argparse
import hashlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import DATA_DIR, SPLITS_DIR


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def make_ham10000_splits(seed: int = 42) -> dict:
    """Group split by lesion_id, stratified by dx. Returns dict of split→DataFrame."""
    meta_path = DATA_DIR / "ham10000" / "HAM10000_metadata.tab"
    if not meta_path.exists():
        meta_path = DATA_DIR / "ham10000" / "HAM10000_metadata.csv"
    if not meta_path.exists():
        print(f"  WARNING: HAM10000 metadata not found at {meta_path.parent} — skipping.")
        return {}

    sep = "\t" if meta_path.suffix == ".tab" else ","
    meta = pd.read_csv(meta_path, sep=sep)
    meta = meta[meta["dx"].isin(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])].copy()

    # Group by lesion_id, take the dx of the first image per lesion
    lesion_df = meta.groupby("lesion_id")["dx"].first().reset_index()

    rng = np.random.default_rng(seed)
    # Stratified split: shuffle within each class then assign proportions
    train_ids, val_ids, test_ids = [], [], []
    for dx, group in lesion_df.groupby("dx"):
        ids = group["lesion_id"].values
        ids = rng.permutation(ids)
        n = len(ids)
        n_val  = max(1, int(n * 0.15))
        n_test = max(1, int(n * 0.15))
        test_ids.extend(ids[:n_test])
        val_ids.extend(ids[n_test:n_test + n_val])
        train_ids.extend(ids[n_test + n_val:])

    train_set = set(train_ids)
    val_set   = set(val_ids)
    test_set  = set(test_ids)

    splits = {
        "train": meta[meta["lesion_id"].isin(train_set)].reset_index(drop=True),
        "val":   meta[meta["lesion_id"].isin(val_set)].reset_index(drop=True),
        "test":  meta[meta["lesion_id"].isin(test_set)].reset_index(drop=True),
    }

    # Leakage check
    all_pairs = [(s1, s2) for s1 in ["train", "val", "test"] for s2 in ["train", "val", "test"] if s1 < s2]
    for s1, s2 in all_pairs:
        overlap = set(splits[s1]["lesion_id"]) & set(splits[s2]["lesion_id"])
        if overlap:
            print(f"  WARNING: {len(overlap)} lesion_ids appear in both {s1} and {s2}!")
        else:
            print(f"  OK: no lesion_id overlap between {s1} and {s2}")

    for split_name, df in splits.items():
        print(f"  HAM10000 {split_name}: {len(df)} images, {df['lesion_id'].nunique()} lesions")

    return splits


def make_isic2018_seg_splits(seed: int = 42) -> dict:
    """Random split of ISIC 2018 Task 1 image IDs. Returns dict of split→list of image stems."""
    img_dir = DATA_DIR / "isic2018_seg" / "ISIC2018_Task1-2_Training_Input"
    if not img_dir.exists():
        print(f"  WARNING: ISIC 2018 seg images not found at {img_dir} — skipping.")
        return {}

    image_stems = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if not image_stems:
        print(f"  WARNING: No .jpg files found in {img_dir} — skipping.")
        return {}

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(image_stems))
    n = len(image_stems)
    n_test = max(1, int(n * 0.15))
    n_val  = max(1, int(n * 0.15))

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    stems = np.array(image_stems)
    splits = {
        "train": pd.DataFrame({"image_id": stems[train_idx]}),
        "val":   pd.DataFrame({"image_id": stems[val_idx]}),
        "test":  pd.DataFrame({"image_id": stems[test_idx]}),
    }
    for split_name, df in splits.items():
        print(f"  ISIC2018 seg {split_name}: {len(df)} images")
    return splits


def main():
    parser = argparse.ArgumentParser(description="Create fixed split CSV files.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing splits to {SPLITS_DIR}\n")

    print("=== HAM10000 (group split by lesion_id) ===")
    ham_splits = make_ham10000_splits(seed=args.seed)
    for split_name, df in ham_splits.items():
        out = SPLITS_DIR / f"ham10000_{split_name}.csv"
        df.to_csv(out, index=False)
        print(f"  Wrote {out} (sha256={_sha256(out)})")

    print("\n=== ISIC 2018 Segmentation ===")
    seg_splits = make_isic2018_seg_splits(seed=args.seed)
    for split_name, df in seg_splits.items():
        out = SPLITS_DIR / f"isic2018_seg_{split_name}.csv"
        df.to_csv(out, index=False)
        print(f"  Wrote {out} (sha256={_sha256(out)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
