# Cerebra Triage POC — CPU-only image
# Multi-stage: build installs deps; runtime image is lean.
#
# The trained checkpoint (models/cerebra_whole_tumour.pt) is NOT baked in — it is
# large and produced by `scripts/train_model.py`. Mount it at runtime:
#   docker run -v $PWD/models:/app/models -p 8000:8000 cerebra:latest
# Without a checkpoint or the MONAI bundle, the service runs on the untrained
# fallback (smoke-test only), which is logged loudly.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

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

# Pre-generate synthetic fixture so the smoke demo works offline
RUN python scripts/make_synthetic_sample.py \
      --output-dir /app/tests/fixtures/synthetic

# Model cache dir — mount a trained checkpoint here at runtime
RUN mkdir -p /app/models /app/data

EXPOSE 8000

CMD ["uvicorn", "cerebra.api:app", "--host", "0.0.0.0", "--port", "8000"]
