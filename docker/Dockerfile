# Cerebra Triage POC — CPU-only image
# Multi-stage: build installs deps; runtime image is lean.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install uv
RUN pip install uv==0.8.17

WORKDIR /app

# --- dependency layer (cached unless pyproject.toml changes) ---
FROM base AS deps

COPY pyproject.toml README.md ./
RUN uv venv /opt/venv --python 3.11 && \
    uv pip install --python /opt/venv/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -e "." && \
    /opt/venv/bin/python -c "import cerebra; print('import ok')"

# --- runtime image ---
FROM base AS runtime

COPY --from=deps /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md pyproject.toml ./

# Pre-generate synthetic fixture so the smoke test works offline
RUN python scripts/make_synthetic_sample.py \
      --output-dir /app/tests/fixtures/synthetic

# Create model cache dir
RUN mkdir -p /app/models

# Download the MONAI bundle at build time so the first request is fast.
# If the download fails (no internet at build time), the container still starts
# and will use the fallback model.
RUN python -m monai.bundle download \
      --name brats_mri_segmentation \
      --bundle_dir /app/models \
    || echo "WARNING: bundle download failed — will use fallback model"

EXPOSE 8000

CMD ["uvicorn", "cerebra.api:app", "--host", "0.0.0.0", "--port", "8000"]
