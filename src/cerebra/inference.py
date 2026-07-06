"""Inference: load a segmentation model and produce a calibrated tumour probability map.

Model resolution order (first available wins):
  1. **Cerebra trained checkpoint** (`models/cerebra_whole_tumour.pt`) — our own binary
     whole-tumour model trained on real MSD data, with temperature-scaled calibration.
  2. **MONAI bundle** `brats_mri_segmentation` — a pretrained 4-class BraTS model.
  3. **Untrained fallback** — random-weight SegResNet, for structural smoke tests ONLY.
     This path is logged loudly and its outputs are not meaningful.

Whatever the source, `run_inference` returns a single-channel calibrated probability
map P(tumour) of shape (H, W, D), so all downstream code has one contract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import structlog
import torch
from monai.inferers import sliding_window_inference

from cerebra.calibrate import apply_temperature

log = structlog.get_logger()

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
TRAINED_CHECKPOINT = MODELS_DIR / "cerebra_whole_tumour.pt"
BUNDLE_NAME = "brats_mri_segmentation"

# Sliding-window parameters
ROI_SIZE = (128, 128, 128)
SW_BATCH_SIZE = 1
OVERLAP = 0.25
NUM_CLASSES = 4  # bundle path


@dataclass
class LoadedModel:
    """A ready-to-run model plus the metadata needed to interpret its output."""

    model: torch.nn.Module
    kind: str            # "trained" | "bundle" | "fallback"
    version: str
    temperature: float = 1.0
    roi_size: tuple[int, int, int] = ROI_SIZE
    val_dice: float | None = None


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_trained(device: torch.device) -> LoadedModel | None:
    """Load our own trained + calibrated whole-tumour checkpoint if present."""
    if not TRAINED_CHECKPOINT.exists():
        return None
    from cerebra.model import build_model

    ckpt = torch.load(str(TRAINED_CHECKPOINT), map_location=device, weights_only=False)
    roi = tuple(ckpt.get("roi_size", ROI_SIZE))
    model = build_model(ckpt["architecture"], roi).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    version = f"{ckpt['architecture']}-dice{ckpt.get('val_dice', 0):.3f}"
    log.info(
        "inference.load_trained",
        architecture=ckpt["architecture"],
        val_dice=ckpt.get("val_dice"),
        temperature=ckpt.get("temperature", 1.0),
        device=str(device),
    )
    return LoadedModel(
        model=model, kind="trained", version=version,
        temperature=float(ckpt.get("temperature", 1.0)),
        roi_size=roi, val_dice=ckpt.get("val_dice"),
    )


def _load_bundle(device: torch.device) -> LoadedModel | None:
    """Load the pretrained MONAI brats_mri_segmentation bundle if downloaded."""
    bundle_path = MODELS_DIR / BUNDLE_NAME
    config_path = bundle_path / "configs" / "inference.json"
    if not config_path.exists():
        config_path = bundle_path / "configs" / "inference.yaml"
    if not config_path.exists():
        return None
    try:
        from monai.bundle import ConfigParser

        parser = ConfigParser()
        parser.read_config(str(config_path))
        model = parser.get_parsed_content("network_def", instantiate=True)

        ckpt_dir = bundle_path / "models"
        ckpts = list(ckpt_dir.glob("model*.pt")) + list(ckpt_dir.glob("*.pth"))
        if ckpts:
            state = torch.load(str(ckpts[0]), map_location=device, weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)

        version = "unknown"
        meta_path = bundle_path / "configs" / "metadata.json"
        if meta_path.exists():
            import json
            version = json.load(open(meta_path)).get("version", version)

        model = model.to(device).eval()
        log.info("inference.load_bundle", version=version, device=str(device))
        return LoadedModel(model=model, kind="bundle", version=version, roi_size=(128, 128, 64))
    except Exception as exc:
        log.warning("inference.load_bundle", status="failed", error=str(exc))
        return None


def _load_fallback(device: torch.device) -> LoadedModel:
    """Untrained SegResNet — smoke-test structural path only. NOT clinically valid."""
    from monai.networks.nets import SegResNet

    model = SegResNet(
        blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=8,
        in_channels=4, out_channels=1, dropout_prob=0.0,
    ).to(device).eval()
    log.warning(
        "inference.load_fallback",
        status="UNTRAINED",
        note="random weights — outputs are NOT clinically valid, smoke-test only",
    )
    return LoadedModel(model=model, kind="fallback", version="fallback-untrained")


def load_model(device: torch.device | None = None, allow_fallback: bool = True) -> LoadedModel:
    """Resolve the best available model: trained → bundle → (optional) fallback."""
    if device is None:
        device = get_device()

    loaded = _load_trained(device) or _load_bundle(device)
    if loaded is not None:
        return loaded
    if allow_fallback:
        return _load_fallback(device)
    raise FileNotFoundError(
        "No trained checkpoint or MONAI bundle found. "
        "Train one with `python scripts/train_model.py` or download the bundle."
    )


def run_inference(
    loaded: LoadedModel,
    input_tensor: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int]:
    """Run sliding-window inference and return a calibrated P(tumour) map.

    Args:
        loaded: a LoadedModel from `load_model`.
        input_tensor: (1, 4, H, W, D) preprocessed scan.

    Returns:
        (tumour_prob_map, inference_time_ms) where tumour_prob_map is an (H, W, D)
        float tensor of calibrated P(tumour) in [0, 1].
    """
    if device is None:
        device = get_device()
    input_tensor = input_tensor.to(device)

    t0 = time.monotonic()
    with torch.no_grad():
        logits = sliding_window_inference(
            input_tensor, loaded.roi_size, SW_BATCH_SIZE, loaded.model,
            overlap=OVERLAP, mode="gaussian",
        )
        if loaded.kind == "bundle":
            # 4-class softmax → P(tumour) = 1 - P(background)
            probs = torch.softmax(logits, dim=1)
            tumour = 1.0 - probs[0, 0]
        else:
            # binary head → temperature-scaled sigmoid
            tumour = apply_temperature(logits, loaded.temperature)[0, 0]
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    log.info(
        "inference.run", kind=loaded.kind, temperature=loaded.temperature,
        inference_time_ms=elapsed_ms, device=str(device),
        out_shape=list(tumour.shape),
    )
    return tumour.cpu(), elapsed_ms
