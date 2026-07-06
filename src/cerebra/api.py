"""FastAPI service exposing POST /triage.

Accepts a multipart file upload (zip or tar.gz of the BraTS study directory).
Returns the TriageReport as JSON.  The overlay PNG is saved to a temporary
directory and its path is included in the response.

Model loading is done once at startup via a FastAPI lifespan context manager
to avoid re-loading weights on every request.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
import zipfile
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from cerebra.inference import get_device, load_model
from cerebra.pipeline import run_triage
from cerebra.schemas import TriageReport

log = structlog.get_logger()

# Module-level state populated in lifespan — no global mutation after startup
_model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model weights once at startup; release on shutdown."""
    device = get_device()
    log.info("api.lifespan", status="loading_model", device=str(device))
    loaded = load_model(device=device)
    _model_state["loaded"] = loaded
    _model_state["device"] = device
    log.info("api.lifespan", status="ready", kind=loaded.kind, model_version=loaded.version)
    yield
    _model_state.clear()
    log.info("api.lifespan", status="shutdown")


app = FastAPI(
    title="Cerebra Triage API",
    description="Stage 1 brain MRI triage — tumour detection POC",
    version="0.1.0",
    lifespan=lifespan,
)


def _extract_study(archive_path: Path, dest: Path) -> Path:
    """Extract a zip or tar.gz archive to dest, return the study root directory."""
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")

    # If there is a single top-level directory inside the extract, step into it
    children = [p for p in dest.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return dest


@app.post("/triage", response_model=TriageReport)
async def triage_endpoint(file: UploadFile = File(..., description="zip or tar.gz of BraTS study")) -> TriageReport:
    """Accept a zipped BraTS study, run triage, return structured JSON report.

    The response includes visualization_path pointing to the server-local PNG.
    Use GET /overlay/{study_id} to retrieve the image.
    """
    study_id = str(uuid.uuid4())[:8]
    work_dir = Path(tempfile.mkdtemp(prefix=f"cerebra_{study_id}_"))

    try:
        # Save upload to disk
        upload_path = work_dir / (file.filename or "study.zip")
        with open(upload_path, "wb") as fh:
            content = await file.read()
            fh.write(content)

        # Extract
        extract_dir = work_dir / "study"
        extract_dir.mkdir()
        try:
            study_dir = _extract_study(upload_path, extract_dir)
        except (ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
            raise HTTPException(status_code=422, detail=f"Could not extract archive: {exc}") from exc

        # Run pipeline
        output_dir = work_dir / "output"
        output_dir.mkdir()

        report = run_triage(
            input_path=study_dir,
            model=_model_state.get("loaded"),
            output_dir=output_dir,
            study_id=study_id,
        )
        return report

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("api.triage.error", study_id=study_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/overlay/{study_id}")
async def get_overlay(study_id: str) -> FileResponse:
    """Return the PNG overlay for a completed triage study.

    NOTE: files are stored in /tmp and may be cleaned up by the OS.
    Production would use object storage.
    """
    pattern = f"/tmp/cerebra_{study_id}*/output/cerebra_{study_id}_overlay.png"
    import glob
    matches = glob.glob(pattern)
    if not matches:
        raise HTTPException(status_code=404, detail=f"Overlay not found for study_id={study_id}")
    return FileResponse(matches[0], media_type="image/png")


@app.get("/health")
async def health() -> dict:
    """Liveness check — reports which model kind is loaded."""
    loaded = _model_state.get("loaded")
    return {
        "status": "ok",
        "model_loaded": loaded is not None,
        "model_kind": loaded.kind if loaded else None,
        "model_version": loaded.version if loaded else None,
    }
