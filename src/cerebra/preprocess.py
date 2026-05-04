"""Load and preprocess brain MRI scans into a 4-channel MONAI tensor.

Supports two input formats:
  - BraTS-style directory: four NIfTI files named *_t1.nii.gz, *_t1ce.nii.gz,
    *_t2.nii.gz, *_flair.nii.gz (case-insensitive suffix matching)
  - DICOM directory: a flat or nested directory of .dcm files (4 series expected)

Returns a (1, 4, H, W, D) float32 torch.Tensor suitable for the MONAI bundle.
All volumes are resampled to 1 mm³ isotropic and z-score normalised per channel.
Skull-stripping is intentionally skipped — the brats_mri_segmentation bundle
expects skull-on inputs.
"""

from __future__ import annotations

import re
from pathlib import Path

import nibabel as nib
import numpy as np
import structlog
import torch
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

log = structlog.get_logger()

MODALITY_SUFFIXES: dict[str, list[str]] = {
    "t1":    ["_t1.nii", "_t1.nii.gz"],
    "t1ce":  ["_t1ce.nii", "_t1ce.nii.gz", "_t1c.nii", "_t1c.nii.gz"],
    "t2":    ["_t2.nii", "_t2.nii.gz"],
    "flair": ["_flair.nii", "_flair.nii.gz"],
}

TARGET_SPACING = (1.0, 1.0, 1.0)
# Crop to a fixed spatial size accepted by most BraTS models.
# The MONAI brats_mri_segmentation bundle uses 240×240×155.
SPATIAL_SIZE = (240, 240, 155)


def _find_nifti_files(directory: Path) -> dict[str, Path]:
    """Locate one NIfTI file per modality in a BraTS-style directory."""
    found: dict[str, Path] = {}
    files = sorted(directory.rglob("*.nii.gz")) + sorted(directory.rglob("*.nii"))
    for modality, suffixes in MODALITY_SUFFIXES.items():
        for f in files:
            lower = f.name.lower()
            if any(lower.endswith(s.lower()) for s in suffixes):
                found[modality] = f
                break
    missing = set(MODALITY_SUFFIXES) - set(found)
    if missing:
        raise FileNotFoundError(
            f"Could not find NIfTI files for modalities: {missing} in {directory}"
        )
    return found


def _load_dicom_series(directory: Path) -> dict[str, np.ndarray]:
    """Load 4 DICOM series from a directory and return them keyed by modality.

    Heuristic: assumes series are sorted alphabetically and maps them to
    t1, t1ce, t2, flair in that order. Production code should use SeriesDescription
    or ProtocolName tags — see TODO below.
    # TODO(real-product): parse DICOM SeriesDescription/ProtocolName tags to
    #   reliably identify modality rather than relying on alphabetical order.
    """
    import pydicom  # local import — only needed for DICOM path

    series_dirs: list[Path] = []
    dcm_files = sorted(directory.rglob("*.dcm"))
    by_series: dict[str, list[Path]] = {}
    for dcm in dcm_files:
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
            uid = str(ds.SeriesInstanceUID)
        except Exception:
            uid = dcm.parent.name
        by_series.setdefault(uid, []).append(dcm)

    if len(by_series) < 4:
        raise ValueError(
            f"Expected at least 4 DICOM series, found {len(by_series)} in {directory}"
        )

    modalities = ["t1", "t1ce", "t2", "flair"]
    result: dict[str, np.ndarray] = {}
    for modality, (_, files) in zip(modalities, sorted(by_series.items())):
        slices = []
        for f in sorted(files):
            ds = pydicom.dcmread(str(f))
            slices.append(ds.pixel_array.astype(np.float32))
        result[modality] = np.stack(slices, axis=-1)  # H x W x D
    return result


def load_study(input_path: Path) -> dict[str, Path | np.ndarray]:
    """Detect input format (NIfTI dir or DICOM dir) and return modality paths or arrays."""
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input must be a directory, got: {input_path}")

    # Check for NIfTI files first
    nifti_files = list(input_path.rglob("*.nii.gz")) + list(input_path.rglob("*.nii"))
    if nifti_files:
        log.info("preprocess.load_study", format="nifti", path=str(input_path))
        return _find_nifti_files(input_path)  # type: ignore[return-value]

    # Fall back to DICOM
    dcm_files = list(input_path.rglob("*.dcm"))
    if dcm_files:
        log.info("preprocess.load_study", format="dicom", path=str(input_path))
        arrays = _load_dicom_series(input_path)
        # Write temporaries as NIfTI so MONAI transforms work uniformly
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="cerebra_dcm_"))
        paths: dict[str, Path] = {}
        for modality, arr in arrays.items():
            p = tmp / f"dcm_{modality}.nii.gz"
            nib.save(nib.Nifti1Image(arr, np.eye(4)), str(p))
            paths[modality] = p
        return paths  # type: ignore[return-value]

    raise FileNotFoundError(
        f"No NIfTI (.nii, .nii.gz) or DICOM (.dcm) files found in {input_path}"
    )


def build_transforms() -> Compose:
    """Return MONAI Compose pipeline: load → channel → orient → resample → normalise."""
    keys = ["t1", "t1ce", "t2", "flair"]
    return Compose(
        [
            LoadImaged(keys=keys, image_only=False, ensure_channel_first=False),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=TARGET_SPACING,
                mode=("bilinear", "bilinear", "bilinear", "bilinear"),
            ),
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
            EnsureTyped(keys=keys, dtype=torch.float32),
        ]
    )


def preprocess(modality_paths: dict[str, Path]) -> torch.Tensor:
    """Apply MONAI transforms and stack channels into a (1, 4, H, W, D) tensor.

    Returns a batch-of-one tensor ready to pass directly to the inference model.
    The spatial size is padded/cropped to SPATIAL_SIZE so the model's
    sliding-window logic always sees the expected dimensions.
    """
    data = {k: str(v) for k, v in modality_paths.items()}
    transforms = build_transforms()
    result = transforms(data)

    # EnsureChannelFirstd gives each modality shape (1, H, W, D); squeeze the leading 1
    def _squeeze(t: torch.Tensor) -> torch.Tensor:
        return t.squeeze(0) if t.ndim == 4 else t

    channels = torch.stack(
        [_squeeze(result["t1"]), _squeeze(result["t1ce"]),
         _squeeze(result["t2"]), _squeeze(result["flair"])],
        dim=0,
    )  # (4, H, W, D)

    # Pad/crop to SPATIAL_SIZE so the MONAI bundle's spatial size constraint is met
    channels = _pad_or_crop(channels, SPATIAL_SIZE)

    return channels.unsqueeze(0)  # (1, 4, H, W, D)


def _pad_or_crop(tensor: torch.Tensor, target: tuple[int, int, int]) -> torch.Tensor:
    """Pad (reflect) or centre-crop tensor spatial dims to target (H, W, D)."""
    _, h, w, d = tensor.shape
    th, tw, td = target

    def _pad1d(size: int, target_size: int) -> tuple[int, int]:
        if size >= target_size:
            return 0, 0
        total = target_size - size
        return total // 2, total - total // 2

    ph = _pad1d(h, th)
    pw = _pad1d(w, tw)
    pd = _pad1d(d, td)

    if any(p > 0 for p in ph + pw + pd):
        tensor = torch.nn.functional.pad(
            tensor, (pd[0], pd[1], pw[0], pw[1], ph[0], ph[1]), mode="reflect"
        )

    _, h, w, d = tensor.shape
    sh = (h - th) // 2
    sw = (w - tw) // 2
    sd = (d - td) // 2
    return tensor[:, sh : sh + th, sw : sw + tw, sd : sd + td]
