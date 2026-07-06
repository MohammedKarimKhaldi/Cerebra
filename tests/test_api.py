"""HTTP API tests using httpx AsyncClient."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def study_zip(synthetic_study_dir: Path) -> bytes:
    """Build an in-memory zip archive from the synthetic NIfTI fixture."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in synthetic_study_dir.iterdir():
            zf.write(f, arcname=f.name)
    return buf.getvalue()


@pytest.fixture(scope="session")
def study_files(synthetic_study_dir: Path) -> dict[str, bytes]:
    """Map each modality to the raw bytes of its synthetic NIfTI file."""
    out: dict[str, bytes] = {}
    for mod in ("t1", "t1ce", "t2", "flair"):
        path = synthetic_study_dir / f"BraTS_synthetic_{mod}.nii.gz"
        out[mod] = path.read_bytes()
    return out


@pytest.fixture(scope="session")
def client():
    """Synchronous TestClient for the FastAPI app."""
    from fastapi.testclient import TestClient
    from cerebra.api import app

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_index_serves_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Cerebra" in response.text
    assert "Run triage" in response.text


def test_triage_files_endpoint(client, study_files):
    """The 4-channel loose-upload endpoint returns a valid report + servable overlay."""
    response = client.post(
        "/triage/files",
        files={
            "t1": ("t1.nii.gz", study_files["t1"], "application/octet-stream"),
            "t1ce": ("t1ce.nii.gz", study_files["t1ce"], "application/octet-stream"),
            "t2": ("t2.nii.gz", study_files["t2"], "application/octet-stream"),
            "flair": ("flair.nii.gz", study_files["flair"], "application/octet-stream"),
        },
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["triage_decision"] in ("FLAG", "CLEAR")
    assert 0.0 <= data["confidence"] <= 1.0

    # overlay must be retrievable by the returned study_id
    overlay = client.get(f"/overlay/{data['study_id']}")
    assert overlay.status_code == 200
    assert overlay.headers["content-type"] == "image/png"


def test_triage_files_missing_channel_returns_422(client, study_files):
    response = client.post(
        "/triage/files",
        files={"t1": ("t1.nii.gz", study_files["t1"], "application/octet-stream")},
    )
    assert response.status_code == 422


def test_overlay_missing_returns_404(client):
    response = client.get("/overlay/deadbeef")
    assert response.status_code == 404


def test_triage_returns_json(client, study_zip):
    response = client.post(
        "/triage",
        files={"file": ("study.zip", study_zip, "application/zip")},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    # Required fields
    assert "study_id" in data
    assert data["triage_decision"] in ("FLAG", "CLEAR")
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["schema_version"] == "0.1.0"


def test_triage_findings_structure(client, study_zip):
    response = client.post(
        "/triage",
        files={"file": ("study.zip", study_zip, "application/zip")},
    )
    assert response.status_code == 200
    findings = response.json()["findings"]
    assert "tumour_suspected" in findings
    assert "tumour_volume_mm3" in findings
    assert findings["tumour_volume_mm3"] >= 0.0
    assert "anatomical_region" in findings
    assert "max_diameter_mm" in findings


def test_triage_model_metadata(client, study_zip):
    response = client.post(
        "/triage",
        files={"file": ("study.zip", study_zip, "application/zip")},
    )
    assert response.status_code == 200
    meta = response.json()["model_metadata"]
    assert "cerebra_whole_tumour" in meta["model_id"]
    assert meta["inference_time_ms"] >= 0


def test_triage_invalid_file_returns_422(client):
    response = client.post(
        "/triage",
        files={"file": ("bad.zip", b"not a zip file", "application/zip")},
    )
    assert response.status_code == 422


def test_triage_no_file_returns_422(client):
    response = client.post("/triage")
    assert response.status_code == 422
