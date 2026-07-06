#!/usr/bin/env bash
# Cerebra — one command to update, install, clean up, and launch the web UI.
#
#   ./run.sh
#
# Environment overrides (all optional):
#   PORT=8000            port for the web UI
#   HOST=0.0.0.0         bind address
#   CEREBRA_CPU=1        install CPU-only PyTorch wheels (default: GPU/CUDA wheels)
#   CEREBRA_NO_SERVE=1   do the update + clean steps but don't start the server
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# Clean transient temp dirs the app/pipeline scatter in /tmp, on entry and on exit.
cleanup_tmp() { rm -rf /tmp/cerebra_* /tmp/cerebra_overlays_* /tmp/cerebra_dcm_* 2>/dev/null || true; }
trap cleanup_tmp EXIT

say "Updating repository"
git pull --ff-only 2>/dev/null || echo "   (no update — detached/offline, continuing)"

say "Ensuring uv is installed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

say "Installing / updating dependencies"
[ -d .venv ] || uv venv --python 3.11
if [ "${CEREBRA_CPU:-0}" = "1" ]; then
  uv pip install -e "." --extra-index-url https://download.pytorch.org/whl/cpu
else
  uv pip install -e "."
fi

say "Cleaning caches and stale artifacts"
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find . -type d -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf htmlcov .coverage coverage.xml 2>/dev/null || true
cleanup_tmp

# Report which model will be served so it's obvious in the terminal.
if [ -f models/cerebra_whole_tumour.pt ]; then
  echo "   model: trained checkpoint (models/cerebra_whole_tumour.pt)"
else
  echo "   model: none trained yet — will use MONAI bundle or untrained fallback"
  echo "   (train one with: uv run python scripts/train_model.py --help)"
fi

if [ "${CEREBRA_NO_SERVE:-0}" = "1" ]; then
  say "Setup complete (CEREBRA_NO_SERVE=1, not starting server)"
  exit 0
fi

say "Launching Cerebra UI  →  http://localhost:${PORT}"
echo "   (press Ctrl-C to stop; temp files are cleaned on exit)"
uv run uvicorn cerebra.api:app --host "$HOST" --port "$PORT"
