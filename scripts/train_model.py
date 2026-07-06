"""Train the Cerebra whole-tumour model on real MSD data, then calibrate it.

Runs the full modern pipeline: train → pick best-Dice checkpoint → fit temperature
scaling for reliable probabilities → write the calibrated temperature back into the
checkpoint.

Examples
--------
Full GPU training (clinical-grade target):
    python scripts/train_model.py --max-epochs 300 --architecture swinunetr

Fast CPU proof-of-learning run (small subset, few steps):
    python scripts/train_model.py --limit-studies 24 --max-epochs 8 \
        --max-train-steps 20 --roi 96 96 96 --val-interval 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import structlog
import torch

from cerebra.calibrate import (
    collect_val_logits,
    expected_calibration_error,
    fit_temperature,
    apply_temperature,
)
from cerebra.data import build_loaders, read_dataset_manifest, split_train_val
from cerebra.model import build_model
from cerebra.train import DEFAULT_CHECKPOINT, TrainConfig, get_device, train

log = structlog.get_logger()


def calibrate_checkpoint(cfg: TrainConfig, checkpoint_path: Path) -> None:
    """Load the best checkpoint, fit temperature scaling on val, write T back."""
    device = get_device()
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    model = build_model(ckpt["architecture"], tuple(ckpt["roi_size"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    entries = read_dataset_manifest()
    _, val_files = split_train_val(
        entries, val_fraction=cfg.val_fraction, seed=cfg.seed, limit=cfg.limit_studies
    )
    _, val_loader = build_loaders(
        [], val_files, roi_size=tuple(ckpt["roi_size"]),
        num_workers=cfg.num_workers, cache_rate=0.0,
    )

    logits, targets = collect_val_logits(
        model, val_loader, device, tuple(ckpt["roi_size"])
    )
    if logits.numel() == 0:
        log.warning("calibrate.skip", reason="no validation voxels collected")
        return

    ece_before = expected_calibration_error(torch.sigmoid(logits), targets)
    temperature = fit_temperature(logits, targets)
    ece_after = expected_calibration_error(
        apply_temperature(logits, temperature), targets
    )

    ckpt["temperature"] = temperature
    ckpt["ece_before"] = ece_before
    ckpt["ece_after"] = ece_after
    torch.save(ckpt, str(checkpoint_path))
    log.info("calibrate.written", temperature=round(temperature, 4),
             ece_before=round(ece_before, 4), ece_after=round(ece_after, 4))
    print(f"Calibration: T={temperature:.3f}  ECE {ece_before:.4f} -> {ece_after:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--architecture", default="segresnet", choices=["segresnet", "swinunetr"])
    parser.add_argument("--roi", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-studies", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig(
        architecture=args.architecture,
        roi_size=tuple(args.roi),
        max_epochs=args.max_epochs,
        val_interval=args.val_interval,
        lr=args.lr,
        num_workers=args.num_workers,
        limit_studies=args.limit_studies,
        max_train_steps=args.max_train_steps,
        cache_rate=args.cache_rate,
        seed=args.seed,
    )

    result = train(cfg, args.checkpoint)
    print(f"\nBest val Dice: {result.best_dice:.4f} @ epoch {result.best_epoch}")
    print(f"Checkpoint: {result.checkpoint_path}")

    if not args.skip_calibration and result.best_dice > 0:
        print("\nFitting temperature scaling for calibrated probabilities …")
        calibrate_checkpoint(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
