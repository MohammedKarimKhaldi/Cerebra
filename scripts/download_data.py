"""Download & extract the public MSD Task01_BrainTumour dataset.

484 multimodal brain-MRI studies with expert tumour annotations — openly hosted
on the MONAI decathlon S3 mirror (no Synapse registration required). ~7.6 GB.

Usage:
    python scripts/download_data.py [--data-dir data] [--keep-tar]
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
TAR_NAME = "Task01_BrainTumour.tar"
EXTRACT_DIR = "Task01_BrainTumour"


def _report(count: int, block: int, total: int) -> None:
    done = count * block
    pct = min(100.0, 100.0 * done / total) if total > 0 else 0.0
    sys.stdout.write(f"\r  downloaded {done / 1e9:.2f} GB ({pct:.0f}%)")
    sys.stdout.flush()


def download(data_dir: Path, keep_tar: bool = False) -> Path:
    """Download and extract MSD Task01. Returns the extracted dataset directory."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / TAR_NAME
    extract_path = data_dir / EXTRACT_DIR

    if (extract_path / "dataset.json").exists():
        print(f"Dataset already present at {extract_path}")
        return extract_path

    if not tar_path.exists():
        print(f"Downloading MSD Task01_BrainTumour (~7.6 GB) → {tar_path}")
        urllib.request.urlretrieve(MSD_URL, tar_path, reporthook=_report)
        print()
    else:
        print(f"Using existing archive {tar_path}")

    print(f"Extracting → {extract_path} …")
    with tarfile.open(tar_path) as tf:
        tf.extractall(data_dir)

    if not keep_tar:
        tar_path.unlink(missing_ok=True)
        print("Removed tarball to reclaim disk.")

    n_studies = len(list((extract_path / "imagesTr").glob("*.nii.gz")))
    print(f"Done. {n_studies} studies at {extract_path}")
    return extract_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--keep-tar", action="store_true", help="keep the .tar after extraction")
    args = parser.parse_args()
    download(args.data_dir, args.keep_tar)


if __name__ == "__main__":
    main()
