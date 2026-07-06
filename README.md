# Cerebra — Stage 1 Brain MRI Triage POC

An AI brain-MRI triage system that flags suspected intracranial tumours from multimodal MRI within minutes, with a **temperature-calibrated** tumour probability and a structured JSON output.

Cerebra trains **its own** binary whole-tumour segmentation model on the **publicly accessible** Medical Segmentation Decathlon **Task01_BrainTumour** dataset (484 expert-annotated multimodal studies — the same imaging cohort as BraTS, but openly hosted with no registration). The trained model's probabilities are then **calibrated by temperature scaling** so the reported "probability of tumour" is trustworthy, not just a monotonic score.

> **This is a POC, not a product.** It is not validated for clinical use and must not be used for patient diagnosis or triage decisions.

---

## What it does

1. Accepts a brain MRI study (4 NIfTI files: T1, T1ce, T2, FLAIR) or a DICOM directory
2. Preprocesses: reorder to canonical channels → resample to 1 mm³ isotropic → z-score normalise
3. Runs sliding-window inference with the **trained Cerebra whole-tumour model** (SegResNet by default, Swin UNETR optional)
4. Converts logits to a **calibrated** P(tumour) map via temperature scaling
5. Computes: tumour volume, max axial diameter, coarse anatomical region, study-level probability
6. Applies triage rule: **FLAG** if volume ≥ 250 mm³ **and** probability ≥ 0.5
7. Saves a 2D axial PNG overlay and returns a structured JSON report

### Model resolution order

At inference the best available model is used automatically:

1. **Trained Cerebra checkpoint** — `models/cerebra_whole_tumour.pt` (our own model, calibrated)
2. **MONAI bundle** — pretrained `brats_mri_segmentation` (4-class BraTS), if downloaded
3. **Untrained fallback** — random-weight SegResNet, **smoke-test only**, logged loudly

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
    "model_id": "cerebra_whole_tumour[trained]",
    "model_version": "segresnet-dice0.812",
    "inference_time_ms": 4321,
    "device": "cpu"
  },
  "visualization_path": "/tmp/cerebra_demo01_overlay.png",
  "schema_version": "0.1.0"
}
```

`confidence` is the calibrated study-level tumour probability (mean of the most-confident decile of voxels in the flagged region).

---

## Real data + training

### 1. Download the public dataset (~7.6 GB, no registration)

```bash
python scripts/download_data.py            # → data/Task01_BrainTumour/
```

The dataset is the **MSD Task01_BrainTumour** cohort, openly mirrored on S3. It ships 484 studies, each with 4 co-registered MRI channels and an expert tumour annotation. Cerebra reframes the labels as **binary whole-tumour** (any tumour vs. background) — exactly the Stage-1 triage question, and far faster to train reliably than the 3-class problem.

### 2. Train + calibrate

```bash
# Full training (GPU strongly recommended — targets clinical-grade Dice)
python scripts/train_model.py --architecture swinunetr --max-epochs 300

# Fast CPU proof-of-learning run (small subset, capped steps)
python scripts/train_model.py \
    --limit-studies 24 --max-epochs 8 --max-train-steps 20 --roi 96 96 96
```

Training does DiceCE loss, AdamW + cosine LR, patch-based sampling, sliding-window validation with a real Dice metric, and best-Dice checkpointing. After the best checkpoint is chosen, **temperature scaling** is fit on the validation set and written back into the checkpoint (`temperature`, plus `ece_before`/`ece_after`).

> **Honest note on compute.** A 3-D segmentation network reaches clinical-grade whole-tumour Dice (~0.85+) only with a GPU and many epochs. On CPU the same pipeline still learns genuine tumour features (validation Dice climbs well above zero) and produces a real, non-random, calibrated checkpoint — enough to demonstrate the end-to-end claim — but it is **not** a converged clinical model. The `--limit-studies` / `--max-train-steps` flags exist for exactly this CPU demonstration.

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

## Running on a study directory

Point the CLI at any directory containing the four MRI channels (case-insensitive suffix matching):
- `*_t1.nii.gz`
- `*_t1ce.nii.gz` or `*_t1c.nii.gz`
- `*_t2.nii.gz`
- `*_flair.nii.gz`

```bash
cerebra-triage /path/to/study_dir --output-dir /tmp/cerebra_out
```

Studies from the downloaded MSD dataset live under `data/Task01_BrainTumour/imagesTr/` as single 4-channel NIfTI files; the demo and tests use a synthetic fixture so you can run everything without the full download.

### Optional: pretrained MONAI bundle

If you have not trained a Cerebra model, the pipeline can fall back to the pretrained 4-class `brats_mri_segmentation` bundle:

```bash
python -m monai.bundle download --name brats_mri_segmentation --bundle_dir ./models
```

If neither a trained checkpoint nor the bundle is available, an **untrained** SegResNet is used for structural smoke tests only — outputs are not meaningful and this is logged loudly at WARNING level.

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
pytest tests/ -v --cov=src/cerebra
```

Tests use a synthetic fixture and do **not** require the dataset download or a
trained model (they exercise the untrained fallback for the end-to-end path).
Coverage targets (all met) on the pure-logic modules:

| Module | Coverage |
|---|---|
| `postprocess.py` | 100% |
| `schemas.py` | 100% |
| `calibrate.py` | high |
| `model.py` | high |

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
│   ├── data.py            # real MSD dataset ingestion + transforms (binary whole-tumour)
│   ├── model.py           # model factory: SegResNet (default) / Swin UNETR
│   ├── train.py           # modern training pipeline (DiceCE, cosine LR, best-Dice ckpt)
│   ├── calibrate.py       # temperature scaling for reliable probabilities
│   ├── preprocess.py      # DICOM/NIfTI loading + MONAI transforms
│   ├── inference.py       # trained-ckpt / bundle / fallback resolution + calibrated infer
│   ├── postprocess.py     # threshold, volume, region mapping, study probability
│   ├── viz.py             # axial-slice overlay rendering with matplotlib
│   ├── pipeline.py        # orchestrates preprocess → infer → postprocess → report
│   ├── cli.py             # cerebra-triage entry point (Typer)
│   └── api.py             # FastAPI app with POST /triage
├── tests/
│   ├── conftest.py        # builds synthetic fixture
│   ├── test_postprocess.py
│   ├── test_model_calibrate.py
│   ├── test_pipeline.py
│   └── test_api.py
├── scripts/
│   ├── make_synthetic_sample.py
│   ├── download_data.py   # fetch + extract public MSD Task01_BrainTumour
│   ├── train_model.py     # train + calibrate on real data
│   └── run_demo.py        # end-to-end demo against synthetic sample
└── docker/
    └── Dockerfile
```

---

## Known limitations / production TODOs

- **Compute** — the checkpoint shipped from a CPU run is a real, calibrated model that learns tumour features but is not converged to clinical Dice. Retrain on GPU (`scripts/train_model.py --architecture swinunetr --max-epochs 300`) for clinical-grade performance.
- **Anatomical region mapping** uses a bounding-box heuristic on volume centre. Production would register to MNI152 and use an atlas (AAL or Brodmann areas).
- **DICOM modality identification** maps series alphabetically. Production should parse `SeriesDescription` or `ProtocolName` DICOM tags.
- **No authentication** — the API is open. Production requires AuthN/AuthZ.
- **Outputs written to /tmp** — not persistent. Production would use object storage.
- **CPU-only Docker image** — add `nvidia/cuda` base for GPU-accelerated training/inference.

---

## Disclaimer

This software is a research prototype only. It has not been validated for clinical use, is not a medical device, and must not be used to make patient care decisions.