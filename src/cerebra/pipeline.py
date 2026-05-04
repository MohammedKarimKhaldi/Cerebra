"""Orchestrates the full triage pipeline: preprocess → infer → postprocess → report.

Designed to be called from both the CLI and the FastAPI handler.
Pass the already-loaded model in to avoid repeated bundle loading across requests.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import structlog
import torch

from cerebra.inference import get_device, load_model, run_inference
from cerebra.postprocess import postprocess
from cerebra.preprocess import load_study, preprocess
from cerebra.schemas import ModelMetadata, TriageReport
from cerebra.viz import render_overlay

log = structlog.get_logger()


def run_triage(
    input_path: Path,
    model: torch.nn.Module | None = None,
    model_version: str = "unknown",
    output_dir: Path | None = None,
    study_id: str | None = None,
) -> TriageReport:
    """End-to-end triage pipeline.

    Args:
        input_path: Directory containing BraTS NIfTI files or DICOM series.
        model: Pre-loaded PyTorch model.  Loaded from bundle on first call if None.
        model_version: Version string for the loaded model weights.
        output_dir: Where to write the PNG overlay.  Defaults to /tmp.
        study_id: Stable identifier for this study; auto-generated if None.

    Returns:
        TriageReport with structured findings and path to the overlay PNG.
    """
    study_id = study_id or str(uuid.uuid4())[:8]
    output_dir = output_dir or Path("/tmp")
    device = get_device()
    bound_log = log.bind(study_id=study_id)

    # --- Stage 1: load model (only if not passed in) ---
    if model is None:
        t0 = time.monotonic()
        model, model_version = load_model(device=device)
        bound_log.info("pipeline.load_model", latency_ms=int((time.monotonic() - t0) * 1000))

    # --- Stage 2: preprocess ---
    t0 = time.monotonic()
    bound_log.info("pipeline.preprocess", stage="start", path=str(input_path))
    modality_paths = load_study(input_path)
    input_tensor = preprocess(modality_paths)   # (1, 4, H, W, D)
    preprocess_ms = int((time.monotonic() - t0) * 1000)
    bound_log.info("pipeline.preprocess", stage="done", latency_ms=preprocess_ms,
                   shape=list(input_tensor.shape))

    # --- Stage 3: inference ---
    t0 = time.monotonic()
    bound_log.info("pipeline.inference", stage="start")
    probs, inference_ms = run_inference(model, input_tensor, device=device)
    bound_log.info("pipeline.inference", stage="done", latency_ms=inference_ms)

    # --- Stage 4: postprocess ---
    t0 = time.monotonic()
    volume_shape = tuple(input_tensor.shape[2:])    # (H, W, D)
    findings, decision, confidence, component_mask = postprocess(probs, volume_shape)
    postprocess_ms = int((time.monotonic() - t0) * 1000)
    bound_log.info(
        "pipeline.postprocess",
        stage="done",
        latency_ms=postprocess_ms,
        decision=decision,
        volume_mm3=findings.tumour_volume_mm3,
    )

    # --- Stage 5: visualise ---
    t0 = time.monotonic()
    viz_path = output_dir / f"cerebra_{study_id}_overlay.png"
    render_overlay(input_tensor, component_mask, study_id, viz_path, confidence)
    viz_ms = int((time.monotonic() - t0) * 1000)
    bound_log.info("pipeline.viz", stage="done", latency_ms=viz_ms, path=str(viz_path))

    report = TriageReport(
        study_id=study_id,
        triage_decision=decision,
        confidence=confidence,
        findings=findings,
        model_metadata=ModelMetadata(
            model_id="brats_mri_segmentation",
            model_version=model_version,
            inference_time_ms=inference_ms,
            device=str(device),
        ),
        visualization_path=str(viz_path),
        schema_version="0.1.0",
    )

    bound_log.info(
        "pipeline.complete",
        decision=report.triage_decision,
        confidence=report.confidence,
        volume_mm3=findings.tumour_volume_mm3,
        total_ms=preprocess_ms + inference_ms + postprocess_ms + viz_ms,
    )
    return report
