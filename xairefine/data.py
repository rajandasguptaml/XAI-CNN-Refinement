"""Datasets, splits and external test sets.

Dataset-1 is the Msoud brain tumor MRI dataset (Nickparvar, Kaggle, CC BY 4.0) and Dataset-2 is the SARTAJ / NeuroMRI
dataset (Bhuvaji et al., Kaggle, MIT). Image-level duplicate information comes from the Brain Tumor MRI Duplicate &
Leakage Audit (`image_metadata.csv`, CC BY 4.0), which hashes every image of both datasets.

Two protocols are supported:

- ``paper``: the protocol of the paper, a stratified random 80/10/10 split of Dataset-1 by image.
- ``clean``: the audit's group-aware 80/10/10 split (``recommended_split``), in which near-duplicate images
  (perceptual-hash distance <= 4) never straddle train, validation and test.

For every protocol, test sets are also reported without leaked images: a test image is "leaked" when an image of the
same duplicate group is in the training or validation partition. For Dataset-2 this removes the SARTAJ images that are
copies of Dataset-1 training images, which is what an "unseen dataset" claim requires.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES

_LABEL_MAP = {
    "glioma": "glioma", "glioma_tumor": "glioma",
    "meningioma": "meningioma", "meningioma_tumor": "meningioma",
    "notumor": "notumor", "no_tumor": "notumor",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary",
}


def load_metadata(audit_csv: str | Path, image_root: str | Path, include_aug: bool = False) -> pd.DataFrame:
    """Read the audit table and attach absolute paths and integer labels.

    ``image_root`` must contain ``msoud/Training``, ``msoud/Testing``, ``sartaj/Training`` and ``sartaj/Testing`` as
    unpacked from the two Kaggle datasets. The 203 Msoud files named ``*aug*`` were added to the Kaggle dataset after
    the paper (which reports 7,023 images) and are excluded unless ``include_aug``.
    """
    m = pd.read_csv(audit_csv)
    m["label"] = m["class_dir"].map(_LABEL_MAP).fillna(m["label"])
    m["y"] = m["label"].map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    root = Path(image_root)
    m["path"] = [str(root / p) for p in m["image_id"]]
    if not include_aug:
        m = m[~((m["dataset"] == "msoud") & m["filename"].str.contains("aug", case=False))]
    return m.reset_index(drop=True)


def check_images(meta: pd.DataFrame, n: int = 20) -> None:
    missing = [p for p in meta["path"].head(n) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of the first {n} images are missing, e.g. {missing[0]}. Download the two Kaggle datasets "
            "into the image root as described in data/README.md."
        )


def stratified_split(df: pd.DataFrame, seed: int, frac=(0.8, 0.1, 0.1)) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, g in df.groupby("y"):
        idx = rng.permutation(g.index.to_numpy())
        n_tr = int(round(frac[0] * len(idx)))
        n_va = int(round(frac[1] * len(idx)))
        parts["train"].append(idx[:n_tr])
        parts["val"].append(idx[n_tr : n_tr + n_va])
        parts["test"].append(idx[n_tr + n_va :])
    return {k: df.loc[np.concatenate(v)].reset_index(drop=True) for k, v in parts.items()}


def make_splits(meta: pd.DataFrame, protocol: str, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Train / val / test of Dataset-1 plus the evaluation sets of both datasets."""
    d1 = meta[meta["dataset"] == "msoud"]
    d2 = meta[meta["dataset"] == "sartaj"].reset_index(drop=True)
    if protocol == "paper":
        s = stratified_split(d1, seed)
    elif protocol == "clean":
        name = {"train": "train", "val": "validation", "test": "test"}
        s = {k: d1[d1["recommended_split"] == v].reset_index(drop=True) for k, v in name.items()}
    else:
        raise ValueError("protocol must be 'paper' or 'clean'")

    seen_groups = set(s["train"]["group_phash4"]) | set(s["val"]["group_phash4"])
    seen_md5 = set(s["train"]["md5_pixels"]) | set(s["val"]["md5_pixels"])

    def unseen(df: pd.DataFrame) -> pd.DataFrame:
        keep = ~df["group_phash4"].isin(seen_groups) & ~df["md5_pixels"].isin(seen_md5)
        return df[keep].reset_index(drop=True)

    s["test_leakfree"] = unseen(s["test"])
    s["d2_all"] = d2
    s["d2_unseen"] = unseen(d2)
    return s


def describe(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in splits.items():
        counts = df["label"].value_counts()
        rows.append({"set": name, "images": len(df), **{c: int(counts.get(c, 0)) for c in CLASSES}})
    return pd.DataFrame(rows)
