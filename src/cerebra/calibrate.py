"""Probability calibration via temperature scaling.

A segmentation network's raw sigmoid outputs are typically **over-confident** —
they do not match the true frequency of tumour at a given score. Temperature
scaling (Guo et al., 2017) learns a single scalar T that divides the logits
before the sigmoid, so that the resulting probabilities are *calibrated*:
among voxels the model reports at p≈0.7, roughly 70% are actually tumour.

This is what makes the triage "probability of it being a tumour" trustworthy
rather than just a monotonic score. T is fit on held-out validation data by
minimising binary cross-entropy, then stored in the checkpoint and applied at
inference. T > 1 softens over-confidence; T < 1 sharpens.
"""

from __future__ import annotations

import structlog
import torch
from monai.inferers import sliding_window_inference

log = structlog.get_logger()


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return calibrated probabilities: sigmoid(logits / T)."""
    t = max(temperature, 1e-3)
    return torch.sigmoid(logits / t)


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.01,
) -> float:
    """Fit a scalar temperature by minimising BCE on (logits, targets).

    Args:
        logits: flattened raw logits (before sigmoid).
        targets: matching binary ground-truth {0, 1}.

    Returns:
        The optimal temperature (a positive float).
    """
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T to keep T > 0
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    bce = torch.nn.BCEWithLogitsLoss()

    logits = logits.detach().float()
    targets = targets.detach().float()

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        t = torch.exp(log_t)
        loss = bce(logits / t, targets)
        loss.backward()
        return loss

    optimizer.step(_closure)
    temperature = float(torch.exp(log_t).item())
    log.info("calibrate.fit_temperature", temperature=round(temperature, 4))
    return temperature


def collect_val_logits(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    roi_size: tuple[int, int, int],
    max_voxels_per_case: int = 200_000,
    sw_overlap: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather a balanced sample of (logit, target) pairs across the validation set.

    To keep memory bounded we subsample voxels per case, deliberately keeping all
    tumour voxels (rare) plus a random background sample, so the fitted temperature
    is not dominated by the overwhelming background class.
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = sliding_window_inference(
                images, roi_size, 1, model, overlap=sw_overlap, mode="gaussian",
            )
            lg = logits.flatten()
            tg = labels.flatten()

            pos_mask = tg > 0.5
            pos_idx = torch.nonzero(pos_mask, as_tuple=False).flatten()
            neg_idx = torch.nonzero(~pos_mask, as_tuple=False).flatten()

            # keep all positives (capped), sample equal negatives
            n_pos = min(len(pos_idx), max_voxels_per_case // 2)
            n_neg = min(len(neg_idx), max(n_pos, max_voxels_per_case // 2))
            if n_pos > 0:
                pos_sel = pos_idx[torch.randperm(len(pos_idx))[:n_pos]]
                all_logits.append(lg[pos_sel].cpu())
                all_targets.append(tg[pos_sel].cpu())
            if n_neg > 0:
                neg_sel = neg_idx[torch.randperm(len(neg_idx))[:n_neg]]
                all_logits.append(lg[neg_sel].cpu())
                all_targets.append(tg[neg_sel].cpu())

    if not all_logits:
        log.warning("calibrate.collect_val_logits", status="no_voxels")
        return torch.zeros(0), torch.zeros(0)

    return torch.cat(all_logits), torch.cat(all_targets)


def expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) — lower is better-calibrated."""
    probs = probs.flatten()
    targets = targets.flatten().float()
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc = targets[in_bin].mean()
            conf = probs[in_bin].mean()
            ece += (acc - conf).abs() * prop
    return float(ece.item())
