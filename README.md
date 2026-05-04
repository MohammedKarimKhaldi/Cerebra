# Cerebra — Stage 1 Brain MRI Triage POC

A proof of concept demonstrating that a pretrained MONAI model can flag suspected intracranial tumours from multimodal brain MRI (BraTS format) within minutes, with a calibrated confidence score and structured JSON output.

> **This is a POC, not a product.** It is not validated for clinical use and must not be used for patient diagnosis or triage decisions.

---

## What it does

1. Accepts a brain MRI study (4 NIfTI files: T1, T1ce, T2, FLAIR) or DICOM directory
2. Preprocesses: resample → 1 mm³ isotropic, z-score normalise, pad/crop to 240×240×155
3. Runs sliding-window inference using the `brats_mri_segmentation` MONAI bundle
4. Computes: tumour volume, max axial diameter, coarse anatomical region, confidence
5. Applies triage rule: **FLAG** if volume ≥ 250 mm³ **and** max probability ≥ 0.5
6. Saves a 2D axial PNG overlay and returns a structured JSON report

### Output schema

```json
{
  "study_id": "demo01",
  "triage_decision": "FLAG",
  "confidence": 0.87,
  "findings": {
    "tumour_suspected": true,
    "tumour_volume_mm3": 12345.6,
    "anatomical_region": "right frontal",
    "max_diameter_mm": 24.3
  },
  "model_metadata": {
    "model_id": "brats_mri_segmentation",
    "model_version": "0.4.x",
    "inference_time_ms": 4321,
    "device": "cpu"
  },
  "visualization_path": "/tmp/cerebra_demo01_overlay.png",
  "schema_version": "0.1.0"
}
```

---

## Install

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

### Setup

```bash
git clone https://github.com/mohammedkarimkhaldi/cerebra.git
cd cerebra

# Create virtual environment and install all dependencies (CPU PyTorch)
uv venv --python 3.11
uv pip install -e "." --extra-index-url https://download.pytorch.org/whl/cpu

# Install dev dependencies
uv pip install pytest pytest-cov pytest-asyncio
```

---

## Download the MONAI model bundle

The `brats_mri_segmentation` bundle is downloaded automatically on first run and cached under `./models/`. To pre-download:

```bash
python -m monai.bundle download \
    --name brats_mri_segmentation \
    --bundle_dir ./models
```

> **Note:** If the download fails (e.g., no internet access), the pipeline falls back to an **untrained** SegResNet. Outputs in this mode are structurally valid but not clinically meaningful. This is clearly logged at WARNING level.

### Using real BraTS data

The BraTS dataset requires free registration at [Synapse (synapse.org)](https://www.synapse.org/#!Synapse:syn51514105). Once downloaded, point the CLI at the study directory:

```bash
cerebra-triage /path/to/BraTS_TCGA_GBM_0001 --output-dir /tmp/cerebra_out
```

The tool expects these files inside the directory (case-insensitive suffix matching):
- `*_t1.nii.gz`
- `*_t1ce.nii.gz` or `*_t1c.nii.gz`
- `*_t2.nii.gz`
- `*_flair.nii.gz`

---

## Quick demo (synthetic data)

```bash
# Generate synthetic fixture and run the full pipeline
python scripts/run_demo.py --output-dir /tmp/cerebra_demo
```

Expected output: a JSON report printed to stdout and a PNG overlay saved to `/tmp/cerebra_demo/`, completing in under 60 s on CPU.

---

## CLI

```bash
# Triage a study — prints JSON to stdout, writes PNG to /tmp
cerebra-triage /path/to/study_dir

# Custom output directory
cerebra-triage /path/to/study_dir --output-dir /tmp/my_output

# With a stable study identifier
cerebra-triage /path/to/study_dir --study-id patient_001
```

---

## HTTP API

Start the server:

```bash
uvicorn cerebra.api:app --host 0.0.0.0 --port 8000
```

Submit a study (zip the NIfTI directory first):

```bash
cd /path/to/study_dir && zip -r /tmp/study.zip .
curl -F "file=@/tmp/study.zip" http://localhost:8000/triage
```

Other endpoints:
- `GET /health` — liveness check
- `GET /overlay/{study_id}` — retrieve the PNG overlay

Interactive docs: http://localhost:8000/docs

---

## Run tests

```bash
PYTHONPATH=src pytest tests/ -v --cov=src/cerebra
```

Coverage targets (all met):

| Module | Coverage |
|---|---|
| `postprocess.py` | 100% |
| `pipeline.py` | 100% |
| `schemas.py` | 100% |

---

## Docker

Build and run:

```bash
docker build -f docker/Dockerfile -t cerebra:latest .

# Run the API
docker run -p 8000:8000 cerebra:latest

# Smoke test inside container
docker run --rm cerebra:latest python scripts/run_demo.py
```

---

## Project layout

```
cerebra/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/cerebra/
│   ├── __init__.py
│   ├── schemas.py         # Pydantic models for the triage report
│   ├── preprocess.py      # DICOM/NIfTI loading + MONAI transforms
│   ├── inference.py       # MONAI bundle wrapper, sliding-window inference
│   ├── postprocess.py     # threshold, volume, region mapping, confidence
│   ├── viz.py             # axial-slice overlay rendering with matplotlib
│   ├── pipeline.py        # orchestrates preprocess → infer → postprocess → report
│   ├── cli.py             # cerebra-triage entry point (Typer)
│   └── api.py             # FastAPI app with POST /triage
├── tests/
│   ├── conftest.py        # builds synthetic fixture
│   ├── test_postprocess.py
│   ├── test_pipeline.py
│   └── test_api.py
├── scripts/
│   ├── make_synthetic_sample.py
│   └── run_demo.py        # end-to-end demo against synthetic sample
└── docker/
    └── Dockerfile
```

---

## Known limitations / production TODOs

- **Anatomical region mapping** uses a bounding-box heuristic on volume centre. Production would register to MNI152 and use an atlas (AAL or Brodmann areas).
- **DICOM modality identification** maps series alphabetically. Production should parse `SeriesDescription` or `ProtocolName` DICOM tags.
- **Model fallback** to untrained SegResNet when the MONAI bundle is unavailable produces structurally valid but clinically meaningless outputs.
- **No authentication** — the API is open. Production requires AuthN/AuthZ.
- **Outputs written to /tmp** — not persistent. Production would use object storage.
- **CPU-only Docker image** — add `nvidia/cuda` base for GPU-accelerated inference.

---

## Disclaimer

This software is a research prototype only. It has not been validated for clinical use, is not a medical device, and must not be used to make patient care decisions.