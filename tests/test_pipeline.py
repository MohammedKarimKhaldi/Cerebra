"""End-to-end pipeline tests using the synthetic study fixture.

These exercise the full preprocess → infer → postprocess → report path. With no
trained checkpoint or bundle present they run on the untrained fallback model, so
they assert on structure and value ranges rather than clinical correctness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cerebra.schemas import TriageDecision, TriageReport


def test_pipeline_returns_triage_report(synthetic_study_dir, tmp_path):
    """Full pipeline smoke test: produces a TriageReport with valid fields."""
    from cerebra.pipeline import run_triage

    report = run_triage(
        input_path=synthetic_study_dir,
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

    report = run_triage(input_path=synthetic_study_dir, output_dir=tmp_path, study_id="schema01")
    assert report.schema_version == "0.1.0"


def test_pipeline_model_metadata(synthetic_study_dir, tmp_path):
    from cerebra.pipeline import run_triage

    report = run_triage(input_path=synthetic_study_dir, output_dir=tmp_path, study_id="meta01")
    assert "cerebra_whole_tumour" in report.model_metadata.model_id
    assert report.model_metadata.inference_time_ms >= 0
    assert report.model_metadata.device in ("cpu", "cuda", "mps")


def test_pipeline_png_created(synthetic_study_dir, tmp_path):
    from cerebra.pipeline import run_triage

    report = run_triage(input_path=synthetic_study_dir, output_dir=tmp_path, study_id="png01")
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


def test_pipeline_no_model_available_raises(tmp_path, monkeypatch, synthetic_study_dir):
    """With fallback disabled and no trained/bundle model, loading must fail loudly."""
    import cerebra.inference as inf
    from cerebra.pipeline import run_triage

    monkeypatch.setattr(inf, "TRAINED_CHECKPOINT", tmp_path / "does_not_exist.pt")
    monkeypatch.setattr(inf, "MODELS_DIR", tmp_path / "no_models")
    with pytest.raises(FileNotFoundError):
        run_triage(input_path=synthetic_study_dir, output_dir=tmp_path, allow_fallback=False)
