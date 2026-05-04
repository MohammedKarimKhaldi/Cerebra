"""Generate a tiny 4-channel BraTS-format NIfTI study with a synthetic tumour.

Saves four files to tests/fixtures/synthetic/:
  BraTS_synthetic_t1.nii.gz
  BraTS_synthetic_t1ce.nii.gz
  BraTS_synthetic_t2.nii.gz
  BraTS_synthetic_flair.nii.gz

The synthetic lesion is a random ellipsoid placed in the right-frontal quadrant.
T1ce channel has hyperintensity inside the ellipsoid; other channels have mild
contrast. Used for smoke tests and demo only — not medically representative.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np


SHAPE = (120, 120, 80)   # x, y, z  (≈ 1mm³ voxels)
AFFINE = np.diag([1.0, 1.0, 1.0, 1.0])   # identity = 1mm³ isotropic


def _ellipsoid_mask(
    shape: tuple[int, int, int],
    centre: tuple[int, int, int],
    radii: tuple[int, int, int],
) -> np.ndarray:
    """Return a boolean 3-D mask with a filled ellipsoid."""
    x, y, z = np.ogrid[: shape[0], : shape[1], : shape[2]]
    mask = (
        ((x - centre[0]) / radii[0]) ** 2
        + ((y - centre[1]) / radii[1]) ** 2
        + ((z - centre[2]) / radii[2]) ** 2
    ) <= 1.0
    return mask.astype(bool)


def generate(output_dir: Path, seed: int = 42) -> dict[str, Path]:
    """Create synthetic 4-channel NIfTI study and return paths keyed by modality."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Background: uniform white-matter-like intensity + Gaussian noise
    base = rng.normal(loc=0.6, scale=0.05, size=SHAPE).astype(np.float32)

    # Lesion ellipsoid in right-frontal quadrant (x > mid, y > mid, z < mid)
    cx = int(SHAPE[0] * 0.65)
    cy = int(SHAPE[1] * 0.65)
    cz = int(SHAPE[2] * 0.40)
    rx, ry, rz = 12, 10, 8
    lesion = _ellipsoid_mask(SHAPE, (cx, cy, cz), (rx, ry, rz))

    # Skull / background
    skull_mask = _ellipsoid_mask(SHAPE, (SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2), (55, 55, 38))

    channels: dict[str, np.ndarray] = {}

    # T1: brain ~0.6, lesion slightly hypointense, outside skull = 0
    t1 = base.copy()
    t1[lesion] = rng.normal(0.4, 0.03, lesion.sum()).astype(np.float32)
    t1[~skull_mask] = 0.0
    channels["t1"] = t1

    # T1ce: lesion is markedly hyperintense (contrast-enhancing)
    t1ce = base.copy()
    t1ce[lesion] = rng.normal(0.95, 0.02, lesion.sum()).astype(np.float32)
    t1ce[~skull_mask] = 0.0
    channels["t1ce"] = t1ce

    # T2: brain ~0.6, lesion hyperintense
    t2 = base.copy()
    t2[lesion] = rng.normal(0.85, 0.03, lesion.sum()).astype(np.float32)
    t2[~skull_mask] = 0.0
    channels["t2"] = t2

    # FLAIR: similar to T2
    flair = base.copy()
    flair[lesion] = rng.normal(0.80, 0.03, lesion.sum()).astype(np.float32)
    flair[~skull_mask] = 0.0
    channels["flair"] = flair

    paths: dict[str, Path] = {}
    for modality, data in channels.items():
        path = output_dir / f"BraTS_synthetic_{modality}.nii.gz"
        nib.save(nib.Nifti1Image(data, AFFINE), str(path))
        paths[modality] = path
        print(f"Saved {path}")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "fixtures" / "synthetic",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
