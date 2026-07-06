"""Model factory for binary whole-tumour segmentation.

Two modern architectures are supported:
  - **segresnet**  – SegResNet (Myronenko 2018). Efficient residual encoder-decoder;
                     the pragmatic default that trains well even on CPU.
  - **swinunetr**  – Swin UNETR (Hatamizadeh 2022). Transformer-based, state of the
                     art on BraTS; heavier, use with a GPU.

Both output a single-channel logit map. A sigmoid turns it into P(tumour) per voxel.
Keeping one output channel (binary) rather than the 4-class BraTS head makes the
probability directly interpretable and cheaper to calibrate.
"""

from __future__ import annotations

import structlog
import torch
from torch import nn

log = structlog.get_logger()

IN_CHANNELS = 4   # T1, T1ce, T2, FLAIR
OUT_CHANNELS = 1  # binary whole-tumour logit


def build_model(
    architecture: str = "segresnet",
    img_size: tuple[int, int, int] = (128, 128, 128),
) -> nn.Module:
    """Construct a segmentation network for binary whole-tumour prediction.

    `img_size` is only consumed by Swin UNETR (it needs a fixed window size);
    SegResNet is fully convolutional and size-agnostic.
    """
    arch = architecture.lower()
    if arch == "segresnet":
        from monai.networks.nets import SegResNet

        model = SegResNet(
            spatial_dims=3,
            init_filters=16,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2,
        )
    elif arch == "swinunetr":
        from monai.networks.nets import SwinUNETR

        model = SwinUNETR(
            img_size=img_size,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            feature_size=48,
            use_checkpoint=True,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture!r} (use 'segresnet' or 'swinunetr')")

    n_params = sum(p.numel() for p in model.parameters())
    log.info("model.build", architecture=arch, params_m=round(n_params / 1e6, 2))
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
