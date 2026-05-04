"""End-to-end pipeline tests using the synthetic study fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cerebra.schemas import TriageDecision, TriageReport


def _make_mock_model(output_shape: tuple) -> torch.nn.Module:
    """Return a trivial model that always emits the same logit tensor."""

    class _MockModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            # High probability for tumour class 1 in right-frontal region
            logits = torch.zeros(b, 4, *output_shape[2:])
            logits[:, 0] = -2.0   # background logit low
            logits[:, 1, 130:160, 130:160, 70:90] = 5.0  # strong tumour signal
            return logits

    return _MockModel().eval()


def test_pipeline_returns_triage_report(synthetic_study_dir, tmp_path):
    """Full pipeline smoke test: produces a TriageReport with valid fields."""
    from cerebra.pipeline import run_triage

    report = run_triage(
        input_path=synthetic_study_dir,
        model=None,   # will attempt bundle load / fallback
        output_dir=tmp_path,
        study_id="smoke01",
    )

    assert isinstance(report, TriageReport)
    assert report.study_id == "smoke01"
    assert report.triage_decision in (TriageDecision.FLAG, TriageDecision.CLEAR)
    assert 0.0 <= report.confidence <= 1.0
    assert report.findings.tumour_volume_mm3 >= 0.0
    assert report.findings.max_diameter_mm >= 0.0
    assert Path(report.visualization_path).exists()


def test_pipeline_schema_version(synthetic_study_dir, tmp_path):
    from cerebra.pipeline import run_triage

    report = run_triage(
        input_path=synthetic_study_dir,
        output_dir=tmp_path,
        study_id="schema01",
    )
    assert report.schema_version == "0.1.0"


def test_pipeline_model_metadata(synthetic_study_dir, tmp_path):
    from cerebra.pipeline import run_triage

    report = run_triage(
        input_path=synthetic_study_dir,
        output_dir=tmp_path,
        study_id="meta01",
    )
    assert report.model_metadata.model_id == "brats_mri_segmentation"
    assert report.model_metadata.inference_time_ms >= 0
    assert report.model_metadata.device in ("cpu", "cuda", "mps")


def test_pipeline_png_created(synthetic_study_dir, tmp_path):
    from cerebra.pipeline import run_triage

    report = run_triage(
        input_path=synthetic_study_dir,
        output_dir=tmp_path,
        study_id="png01",
    )
    png = Path(report.visualization_path)
    assert png.exists()
    assert png.suffix == ".png"
    assert png.stat().st_size > 0


def test_pipeline_bad_input_raises(tmp_path):
    from cerebra.pipeline import run_triage

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        run_triage(input_path=empty_dir, output_dir=tmp_path)
