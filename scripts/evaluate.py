"""Evaluate a trained Cerebra checkpoint on held-out MSD studies.

Reports the metrics that matter for a segmentation/triage model:
  - whole-tumour Dice (mean + per-case), using the SAME calibrated threshold the
    triage pipeline uses, so the number reflects real deployed behaviour
  - Expected Calibration Error before/after the checkpoint's temperature
  - triage confusion (FLAG/CLEAR vs. tumour-present ground truth) at the volume rule

Examples
--------
Evaluate on the default validation split (matches training seed/limit):
    python scripts/evaluate.py

Evaluate on the full dataset's validation fraction and write per-case CSV:
    python scripts/evaluate.py --limit-studies 0 --csv eval_percase.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import structlog
import torch
from monai.inferers import sliding_window_inference

from cerebra.calibrate import apply_temperature, expected_calibration_error
from cerebra.data import build_loaders, read_dataset_manifest, split_train_val
from cerebra.inference import OVERLAP, SW_BATCH_SIZE
from cerebra.model import build_model
from cerebra.postprocess import (
    VOLUME_THRESHOLD_MM3,
    compute_volume_mm3,
    largest_component,
    threshold_and_label,
)
from cerebra.train import DEFAULT_CHECKPOINT, get_device

log = structlog.get_logger()


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice between two boolean masks."""
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2.0 * inter / denom) if denom > 0 else 1.0


def evaluate(checkpoint: Path, limit_studies: int | None, val_fraction: float,
             seed: int, num_workers: int, csv_path: Path | None) -> dict:
    """Run full evaluation and return a metrics summary dict."""
    device = get_device()
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    temperature = float(ckpt.get("temperature", 1.0))

    model = build_model(ckpt["architecture"], tuple(ckpt["roi_size"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    roi = tuple(ckpt["roi_size"])

    entries = read_dataset_manifest()
    _, val_files = split_train_val(entries, val_fraction=val_fraction, seed=seed,
                                   limit=limit_studies)
    _, val_loader = build_loaders([], val_files, roi_size=roi,
                                  num_workers=num_workers, cache_rate=0.0)

    per_case: list[dict] = []
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = sliding_window_inference(
                images, roi, SW_BATCH_SIZE, model, overlap=OVERLAP, mode="gaussian",
            )
            prob = apply_temperature(logits, temperature)[0, 0].cpu().numpy()
            gt = labels[0, 0].cpu().numpy() > 0.5

            pred = prob >= 0.5
            d = dice_score(pred, gt)

            # triage decision reproduced from the pipeline rule
            comp = largest_component(threshold_and_label(prob))
            vol = compute_volume_mm3(comp)
            flagged = vol >= VOLUME_THRESHOLD_MM3
            gt_has_tumour = gt.sum() > 0
            if flagged and gt_has_tumour:
                tp += 1
            elif flagged and not gt_has_tumour:
                fp += 1
            elif not flagged and not gt_has_tumour:
                tn += 1
            else:
                fn += 1

            per_case.append({"case": i, "dice": round(d, 4),
                             "pred_volume_mm3": round(vol, 1),
                             "gt_voxels": int(gt.sum()), "flagged": flagged})
            log.info("evaluate.case", case=i, dice=round(d, 4), flagged=flagged)

            # subsample voxels for ECE (all positives + equal negatives)
            lg, tg = logits.flatten().cpu(), labels.flatten().cpu()
            pos = torch.nonzero(tg > 0.5).flatten()
            neg = torch.nonzero(tg <= 0.5).flatten()
            n = min(len(pos), 50_000)
            if n > 0:
                all_logits.append(lg[pos[torch.randperm(len(pos))[:n]]])
                all_targets.append(tg[pos[torch.randperm(len(pos))[:n]]])
                neg_sel = neg[torch.randperm(len(neg))[:n]]
                all_logits.append(lg[neg_sel])
                all_targets.append(tg[neg_sel])

    dices = [c["dice"] for c in per_case]
    logits_cat = torch.cat(all_logits) if all_logits else torch.zeros(0)
    targets_cat = torch.cat(all_targets) if all_targets else torch.zeros(0)
    ece_raw = expected_calibration_error(torch.sigmoid(logits_cat), targets_cat) if logits_cat.numel() else float("nan")
    ece_cal = expected_calibration_error(apply_temperature(logits_cat, temperature), targets_cat) if logits_cat.numel() else float("nan")

    summary = {
        "n_cases": len(per_case),
        "mean_dice": round(float(np.mean(dices)), 4) if dices else 0.0,
        "std_dice": round(float(np.std(dices)), 4) if dices else 0.0,
        "min_dice": round(float(np.min(dices)), 4) if dices else 0.0,
        "max_dice": round(float(np.max(dices)), 4) if dices else 0.0,
        "temperature": round(temperature, 4),
        "ece_uncalibrated": round(ece_raw, 4),
        "ece_calibrated": round(ece_cal, 4),
        "triage_tp": tp, "triage_fp": fp, "triage_tn": tn, "triage_fn": fn,
        "checkpoint_val_dice": round(float(ckpt.get("val_dice", 0.0)), 4),
    }

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_case[0].keys()))
            w.writeheader()
            w.writerows(per_case)
        print(f"Per-case metrics → {csv_path}")

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--limit-studies", type=int, default=60,
                   help="cap studies to match a training subset; pass 0 for the full dataset")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--csv", type=Path, default=None)
    args = p.parse_args()

    limit = None if args.limit_studies == 0 else args.limit_studies
    summary = evaluate(args.checkpoint, limit, args.val_fraction, args.seed,
                       args.num_workers, args.csv)

    print("\n===== Cerebra evaluation =====")
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")


if __name__ == "__main__":
    main()
