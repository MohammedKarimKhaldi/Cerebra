"""Real brain-MRI data ingestion for the Medical Segmentation Decathlon (MSD).

Uses the publicly accessible **Task01_BrainTumour** dataset — 484 multimodal
(FLAIR, T1w, T1gd/T1ce, T2w) brain MRI studies with expert tumour annotations.
This is the same imaging cohort as BraTS but distributed openly (no Synapse
registration required) via the MONAI/decathlon S3 mirror.

We frame the learning problem as **binary whole-tumour segmentation** (any
tumour sub-region vs. background). This is exactly what Stage 1 triage needs —
"is there a tumour, and with what probability" — and it is markedly faster and
more reliable to train than the 3-class BraTS problem.

The four MSD channels are ordered [FLAIR, T1w, T1gd, T2w]. We reorder them to
Cerebra's canonical [T1, T1ce, T2, FLAIR] so training and inference share one
channel convention.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import structlog
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
)

log = structlog.get_logger()

# MSD Task01 channel order → Cerebra canonical order [T1, T1ce, T2, FLAIR]
# MSD order is [FLAIR(0), T1w(1), T1gd(2), T2w(3)]
MSD_TO_CANONICAL = [1, 2, 3, 0]

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
DATASET_DIR = DATA_ROOT / "Task01_BrainTumour"


class ConvertToWholeTumourd(MapTransform):
    """Collapse MSD multi-label mask into a binary whole-tumour mask.

    MSD Task01 labels: 0=background, 1=oedema, 2=non-enhancing, 3=enhancing.
    Any positive label becomes 1 (tumour present); background stays 0.

    Storing the target as a single-channel {0,1} map lets us train a binary
    segmentation head, which converges far faster on limited compute than the
    full 3-class task while directly serving the triage question.
    """

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            label = d[key]
            d[key] = (label > 0).astype(np.float32)
        return d


class ReorderChannelsd(MapTransform):
    """Reorder MSD's [FLAIR, T1w, T1gd, T2w] channels to canonical [T1, T1ce, T2, FLAIR]."""

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            img = d[key]
            d[key] = img[MSD_TO_CANONICAL]
        return d


def read_dataset_manifest(dataset_dir: Path = DATASET_DIR) -> list[dict[str, str]]:
    """Read MSD dataset.json and return a list of {image, label} absolute-path dicts."""
    manifest = dataset_dir / "dataset.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"MSD manifest not found at {manifest}. "
            "Run `python scripts/download_data.py` first."
        )
    with open(manifest) as f:
        meta = json.load(f)

    entries: list[dict[str, str]] = []
    for item in meta["training"]:
        # Paths in dataset.json are like "./imagesTr/BRATS_001.nii.gz"
        img = (dataset_dir / item["image"].lstrip("./")).resolve()
        lbl = (dataset_dir / item["label"].lstrip("./")).resolve()
        if img.exists() and lbl.exists():
            entries.append({"image": str(img), "label": str(lbl)})
    log.info("data.read_manifest", n=len(entries), dataset_dir=str(dataset_dir))
    return entries


def split_train_val(
    entries: list[dict[str, str]],
    val_fraction: float = 0.2,
    seed: int = 42,
    limit: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Deterministically split manifest entries into train/val lists.

    `limit` caps the total number of studies used — handy for CPU-bound runs.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(entries))
    rng.shuffle(idx)
    if limit is not None:
        idx = idx[:limit]
    shuffled = [entries[i] for i in idx]
    n_val = max(1, int(len(shuffled) * val_fraction))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    log.info("data.split", n_train=len(train), n_val=len(val))
    return train, val


def train_transforms(roi_size: tuple[int, int, int] = (128, 128, 128)) -> Compose:
    """Training transforms: load → canonical layout → 1mm³ → crop patch → augment."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            ReorderChannelsd(keys=["image"]),
            ConvertToWholeTumourd(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def val_transforms() -> Compose:
    """Validation transforms: full-volume, deterministic (no cropping/augmentation)."""
    return Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"]),
            ReorderChannelsd(keys=["image"]),
            ConvertToWholeTumourd(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_loaders(
    train_files: list[dict],
    val_files: list[dict],
    roi_size: tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 1,
    num_workers: int = 2,
    cache_rate: float = 0.0,
) -> tuple[DataLoader, DataLoader]:
    """Build cached train/val DataLoaders from manifest file lists."""
    # Allow an empty train set (e.g. calibration-only runs need just the val loader).
    train_loader = None
    if train_files:
        train_ds = CacheDataset(
            data=train_files, transform=train_transforms(roi_size),
            cache_rate=cache_rate, num_workers=num_workers,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False,
        )

    val_ds = CacheDataset(
        data=val_files, transform=val_transforms(),
        cache_rate=cache_rate, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader
