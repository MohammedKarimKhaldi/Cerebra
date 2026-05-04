"""Render an annotated axial slice with segmentation overlay.

Finds the axial slice (z index) with the largest tumour cross-section and
draws a semi-transparent colour overlay on the T1ce channel.
Saves the result as a PNG to the specified output path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")   # headless — no display needed


def _find_best_axial_slice(mask: np.ndarray) -> int:
    """Return z-index of the axial slice with the most tumour voxels."""
    counts = mask.sum(axis=(0, 1))   # sum over x, y → (D,)
    if counts.max() == 0:
        return mask.shape[2] // 2    # fallback: mid-slice
    return int(np.argmax(counts))


def render_overlay(
    input_tensor: torch.Tensor,
    mask: np.ndarray,
    study_id: str,
    output_path: Path,
    confidence: float = 0.0,
) -> Path:
    """Save an axial PNG overlay of the tumour segmentation on the T1ce channel.

    Args:
        input_tensor: (1, 4, H, W, D) preprocessed scan tensor.
        mask: (H, W, D) boolean mask of the largest tumour component.
        study_id: used in the figure title.
        output_path: destination .png file path.
        confidence: segmentation confidence displayed in the title.

    Returns:
        The path of the saved PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # T1ce is channel index 1
    t1ce_vol = input_tensor[0, 1].numpy()   # (H, W, D)

    z = _find_best_axial_slice(mask)

    t1ce_slice = t1ce_vol[:, :, z]
    mask_slice = mask[:, :, z]

    # Normalise to [0, 1] for display
    vmin, vmax = t1ce_slice.min(), t1ce_slice.max()
    if vmax > vmin:
        display = (t1ce_slice - vmin) / (vmax - vmin)
    else:
        display = np.zeros_like(t1ce_slice)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.imshow(display.T, cmap="gray", origin="lower", aspect="equal")

    if mask_slice.any():
        # Semi-transparent red overlay
        overlay = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        overlay[mask_slice, 0] = 1.0   # R
        overlay[mask_slice, 3] = 0.45  # alpha
        ax.imshow(overlay.transpose(1, 0, 2), origin="lower", aspect="equal")

    ax.set_title(
        f"Cerebra Triage  |  {study_id}\n"
        f"Axial slice z={z}  |  confidence={confidence:.2f}",
        fontsize=8,
    )
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    return output_path
