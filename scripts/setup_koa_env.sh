#!/usr/bin/env bash
# Setup script for preparing the koa-ml Python environment on KOA.
# Usage:
#   source scripts/setup_koa_env.sh            # default settings
#   PYTHON_MODULE=lang/Python/3.10.4-GCCcore-11.3.0 source scripts/setup_koa_env.sh
#
# The script assumes you are running on KOA (login or compute node) with the
# repository synced to ~/koa-ml (or $PROJECT_ROOT). It will:
#   1. Load the requested Python module (defaults to KOA's Python 3.11.5 build)
#   2. Create/replace .venv
#   3. Upgrade pip tooling
#   4. Install an appropriate PyTorch wheel (defaults to CUDA 12.1 build)
#   5. Install koa-ml in editable mode with the [ml] extras

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_MODULE="${PYTHON_MODULE:-lang/Python/3.11.5-GCCcore-13.2.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

log() {
  printf '[setup-koa] %s\n' "$*"
}

die() {
  log "ERROR: $*"
  return 1
}

if ! command -v module >/dev/null 2>&1; then
  die "Lmod 'module' command not found. Run this script on KOA where modules are available."
fi

log "Loading Python module: ${PYTHON_MODULE}"
module purge >/dev/null 2>&1 || true
if ! module load "${PYTHON_MODULE}" >/dev/null 2>&1; then
  die "Failed to load module '${PYTHON_MODULE}'. Check module availability with 'module avail python'."
fi

if [[ -n "${CUDA_MODULE}" ]]; then
  if module avail "${CUDA_MODULE}" >/dev/null 2>&1; then
    log "Loading CUDA module: ${CUDA_MODULE}"
    if ! module load "${CUDA_MODULE}" >/dev/null 2>&1; then
      log "WARNING: Unable to load CUDA module '${CUDA_MODULE}'. flash-attn build may fail (nvcc required)."
    fi
  else
    log "WARNING: CUDA module '${CUDA_MODULE}' not found in module tree; skipping."
  fi
fi

PYTHON_BIN=""
if [[ -n "${EBROOTPYTHON:-}" ]]; then
  if [[ -x "${EBROOTPYTHON}/bin/python3" ]]; then
    PYTHON_BIN="${EBROOTPYTHON}/bin/python3"
  elif [[ -x "${EBROOTPYTHON}/bin/python" ]]; then
    PYTHON_BIN="${EBROOTPYTHON}/bin/python"
  else
    log "Module '${PYTHON_MODULE}' reports EBROOTPYTHON=${EBROOTPYTHON}, but no python executable was found there."
  fi
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  die "Unable to locate a python interpreter after loading '${PYTHON_MODULE}'. On KOA you may need to run this script inside an interactive 'srun --pty' session so the module binaries are available."
fi

log "Using python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  die "Project root '${PROJECT_ROOT}' not found. Ensure you are running from inside the koa-ml repository."
fi

log "Creating virtual environment at ${VENV_DIR}"
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

log "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

log "Ensuring numpy is installed early (flash-attn expects it)"
python -m pip install --upgrade numpy

if ! python -c "import torch" >/dev/null 2>&1; then
  log "Installing torch ${TORCH_VERSION} (index: ${TORCH_INDEX_URL})"
  python -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
else
  log "Torch already present; skipping torch install"
fi

log "Installing koa-ml and ML extras"
if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  if ! python -m pip install -e ".[ml]"; then
    log "WARNING: 'pip install -e \".[ml]\"' failed (likely flash-attn build). Retrying without flash-attn."
    python -m pip install -e .
    python -m pip install \
      "transformers>=4.40.0" \
      "accelerate>=0.28.0" \
      "peft>=0.11.1" \
      "trl>=0.9.6" \
      "datasets>=2.18.0" \
      "lm-eval[wandb]>=0.4.2" \
      "bitsandbytes>=0.42.0"
  fi
else
  log "INSTALL_FLASH_ATTN=0; installing ML stack without flash-attn"
  python -m pip install -e .
  python -m pip install \
    "transformers>=4.40.0" \
    "accelerate>=0.28.0" \
    "peft>=0.11.1" \
    "trl>=0.9.6" \
    "datasets>=2.18.0" \
    "lm-eval[wandb]>=0.4.2" \
    "bitsandbytes>=0.42.0"
fi

log "Environment ready."
log "To use it in future sessions: source '${VENV_DIR}/bin/activate'"
