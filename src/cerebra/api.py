"""FastAPI service: web UI + triage endpoints.

Routes:
  GET  /                – single-page web UI (upload 4 channels, view result)
  POST /triage/files    – 4 loose NIfTI uploads (t1, t1ce, t2, flair) → TriageReport
  POST /triage          – a zip/tar.gz of a study directory → TriageReport
  GET  /overlay/{id}    – the rendered segmentation overlay PNG for a study
  GET  /health          – liveness + which model kind is loaded

Model weights load once at startup via a lifespan context manager. Rendered
overlays are written to a single server-owned directory so they can be served
back by exact path (no fragile globbing).
"""

from __future__ import annotations

import tarfile
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from cerebra.inference import get_device, load_model
from cerebra.pipeline import run_triage
from cerebra.schemas import TriageReport

log = structlog.get_logger()

STATIC_DIR = Path(__file__).parent / "static"

# Module-level state populated in lifespan — no global mutation after startup
_model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model weights once at startup; prepare the overlay output directory."""
    device = get_device()
    log.info("api.lifespan", status="loading_model", device=str(device))
    loaded = load_model(device=device)
    _model_state["loaded"] = loaded
    _model_state["device"] = device
    _model_state["overlay_dir"] = Path(tempfile.mkdtemp(prefix="cerebra_overlays_"))
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


# ------------------------------------------------------------------ helpers

def _overlay_dir() -> Path:
    return _model_state["overlay_dir"]


def _run(study_dir: Path, study_id: str) -> TriageReport:
    """Run the pipeline, writing the overlay into the shared overlay directory."""
    return run_triage(
        input_path=study_dir,
        model=_model_state.get("loaded"),
        output_dir=_overlay_dir(),
        study_id=study_id,
    )


def _suffix(filename: str | None) -> str:
    """Return '.nii.gz' or '.nii' based on an uploaded filename (default .nii.gz)."""
    name = (filename or "").lower()
    return ".nii" if name.endswith(".nii") else ".nii.gz"


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

    children = list(dest.iterdir())
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return dest


# ------------------------------------------------------------------ routes

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Serve the single-page triage UI."""
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.post("/triage/files", response_model=TriageReport)
async def triage_files(
    t1: UploadFile = File(..., description="T1-weighted NIfTI"),
    t1ce: UploadFile = File(..., description="T1 contrast-enhanced NIfTI"),
    t2: UploadFile = File(..., description="T2-weighted NIfTI"),
    flair: UploadFile = File(..., description="FLAIR NIfTI"),
) -> TriageReport:
    """Accept the four MRI channels as separate uploads and run triage.

    Files are saved under canonical `study_<modality>` names so the loader can
    identify each channel unambiguously regardless of the original filenames.
    """
    study_id = str(uuid.uuid4())[:8]
    work_dir = Path(tempfile.mkdtemp(prefix=f"cerebra_{study_id}_"))
    study_dir = work_dir / "study"
    study_dir.mkdir()

    try:
        for modality, upload in (("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair)):
            dest = study_dir / f"study_{modality}{_suffix(upload.filename)}"
            dest.write_bytes(await upload.read())
        return _run(study_dir, study_id)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("api.triage_files.error", study_id=study_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/triage", response_model=TriageReport)
async def triage_endpoint(
    file: UploadFile = File(..., description="zip or tar.gz of a study directory"),
) -> TriageReport:
    """Accept a zipped/tarred study directory and run triage."""
    study_id = str(uuid.uuid4())[:8]
    work_dir = Path(tempfile.mkdtemp(prefix=f"cerebra_{study_id}_"))

    try:
        upload_path = work_dir / (file.filename or "study.zip")
        upload_path.write_bytes(await file.read())

        extract_dir = work_dir / "study"
        extract_dir.mkdir()
        try:
            study_dir = _extract_study(upload_path, extract_dir)
        except (ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
            raise HTTPException(status_code=422, detail=f"Could not extract archive: {exc}") from exc

        return _run(study_dir, study_id)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("api.triage.error", study_id=study_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/overlay/{study_id}")
async def get_overlay(study_id: str) -> FileResponse:
    """Return the PNG overlay for a completed triage study."""
    # study_id is server-generated (uuid4 hex slice); guard against traversal anyway
    if not study_id.isalnum():
        raise HTTPException(status_code=400, detail="invalid study_id")
    path = _overlay_dir() / f"cerebra_{study_id}_overlay.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Overlay not found for study_id={study_id}")
    return FileResponse(str(path), media_type="image/png")


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
