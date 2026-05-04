"""MONAI bundle wrapper for brats_mri_segmentation sliding-window inference.

Downloads the bundle on first run and caches it under ./models/.
Returns a (1, 4, H, W, D) per-class probability map (softmax output).

The brats_mri_segmentation bundle produces 4 output classes:
  0 = background
  1 = necrotic core (NCR/NET)
  2 = peritumoral oedema (ED)
  3 = GD-enhancing tumour (ET)

For triage we combine classes 1–3 into a single "tumour present" map.
"""

from __future__ import annotations

import time
from pathlib import Path

import structlog
import torch
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import SegResNet

log = structlog.get_logger()

BUNDLE_NAME = "brats_mri_segmentation"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Sliding-window parameters — smaller ROI reduces RAM on CPU
ROI_SIZE = (128, 128, 64)
SW_BATCH_SIZE = 1
OVERLAP = 0.25

# Number of output classes from the BraTS bundle
NUM_CLASSES = 4


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_fallback_model(device: torch.device) -> torch.nn.Module:
    """Construct an untrained SegResNet as a structural fallback.

    Used when the MONAI bundle download fails so the pipeline can still run
    end-to-end in CI / demo mode. The weights are random — outputs are
    meaningless for clinical use.

    NOTE: this is intentionally only called when the real bundle is unavailable.
    The main pipeline will log a clear warning.
    """
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=8,
        in_channels=4,
        out_channels=NUM_CLASSES,
        dropout_prob=0.0,
    ).to(device)
    model.eval()
    return model


def download_bundle(bundle_dir: Path = MODELS_DIR) -> Path:
    """Download brats_mri_segmentation bundle if not already cached.

    Returns the bundle directory path.
    """
    bundle_path = bundle_dir / BUNDLE_NAME
    if bundle_path.exists():
        log.info("inference.download_bundle", status="cached", path=str(bundle_path))
        return bundle_path

    bundle_dir.mkdir(parents=True, exist_ok=True)
    log.info("inference.download_bundle", status="downloading", bundle=BUNDLE_NAME)
    try:
        from monai.bundle import download as bundle_download

        bundle_download(name=BUNDLE_NAME, bundle_dir=str(bundle_dir))
        log.info("inference.download_bundle", status="complete", path=str(bundle_path))
    except Exception as exc:
        log.warning(
            "inference.download_bundle",
            status="failed",
            error=str(exc),
            fallback="using untrained SegResNet",
        )
    return bundle_path


def load_model(
    bundle_dir: Path = MODELS_DIR,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, str]:
    """Load the pretrained brats_mri_segmentation model.

    Returns (model, version_string).  Falls back to untrained SegResNet if the
    bundle cannot be loaded — clearly logged at WARNING level.
    """
    if device is None:
        device = get_device()

    bundle_path = download_bundle(bundle_dir)

    try:
        from monai.bundle import ConfigParser

        config_path = bundle_path / "configs" / "inference.json"
        if not config_path.exists():
            config_path = bundle_path / "configs" / "inference.yaml"

        parser = ConfigParser()
        parser.read_config(str(config_path))

        # Resolve the network definition from the bundle config
        model: torch.nn.Module = parser.get_parsed_content("network_def", instantiate=True)

        # Load pretrained weights
        ckpt_dir = bundle_path / "models"
        ckpt_candidates = list(ckpt_dir.glob("model*.pt")) + list(ckpt_dir.glob("*.pth"))
        if ckpt_candidates:
            ckpt = ckpt_candidates[0]
            state = torch.load(str(ckpt), map_location=device, weights_only=True)
            # Handle various checkpoint formats
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
            log.info("inference.load_model", weights=str(ckpt), device=str(device))
        else:
            log.warning("inference.load_model", status="no_weights_found", path=str(ckpt_dir))

        # Read version from metadata
        version = "unknown"
        meta_path = bundle_path / "configs" / "metadata.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            version = meta.get("version", version)

        model = model.to(device).eval()
        return model, version

    except Exception as exc:
        log.warning(
            "inference.load_model",
            status="bundle_load_failed",
            error=str(exc),
            fallback="untrained SegResNet — outputs are NOT clinically valid",
        )
        model = _build_fallback_model(device)
        return model, "fallback-untrained"


def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int]:
    """Run sliding-window inference on a (1, 4, H, W, D) tensor.

    Returns (probability_map, inference_time_ms) where probability_map is
    (1, 4, H, W, D) softmax probabilities over the 4 BraTS classes.
    """
    if device is None:
        device = get_device()

    inferer = SlidingWindowInferer(
        roi_size=ROI_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        overlap=OVERLAP,
        mode="gaussian",
        progress=False,
    )

    input_tensor = input_tensor.to(device)

    t0 = time.monotonic()
    with torch.no_grad():
        logits = inferer(input_tensor, model)   # (1, 4, H, W, D)
        probs = torch.softmax(logits, dim=1)    # per-class probabilities
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    log.info(
        "inference.run_inference",
        output_shape=list(probs.shape),
        inference_time_ms=elapsed_ms,
        device=str(device),
    )
    return probs.cpu(), elapsed_ms
