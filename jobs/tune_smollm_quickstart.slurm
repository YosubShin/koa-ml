#!/bin/bash
#SBATCH --job-name=tune-smollm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/tune-smollm-%j.out

set -euo pipefail

REMOTE_WORKDIR="${KOA_ML_WORKDIR:-$HOME/koa-ml}"
if [ ! -d "$REMOTE_WORKDIR" ]; then
  echo "Remote workdir not found at $REMOTE_WORKDIR" >&2
  exit 1
fi
cd "$REMOTE_WORKDIR"


echo "======================================"
echo "Fine-tuning SmolLM 135M (Quick Test)"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Load required modules (set KOA_PYTHON_MODULE/KOA_CUDA_MODULE to override)
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  if [ -n "${KOA_PYTHON_MODULE:-}" ]; then
    module load "${KOA_PYTHON_MODULE}"
  else
    module load lang/Python/3.11.5-GCCcore-13.2.0
  fi
  if [ -n "${KOA_CUDA_MODULE:-}" ]; then
    module load "${KOA_CUDA_MODULE}"
  fi
fi

# Activate virtual environment
VENV_PATH="${KOA_ML_VENV:-$HOME/koa-ml/.venv}"
if [ ! -d "$VENV_PATH" ]; then
  echo "Virtualenv not found at $VENV_PATH" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

# Print GPU info
echo "==== GPU Info ===="
nvidia-smi
echo ""

# Print Python environment
echo "==== Python Environment ===="
which python
python --version
echo ""

# Run training
echo "==== Starting Training ===="
python tune/train.py \
    --config tune/configs/models/smollm_135m_lora.yaml \
    --output_dir ./output/smollm_quickstart_${SLURM_JOB_ID}

echo ""
echo "======================================"
echo "Training complete!"
echo "Ended: $(date)"
echo "======================================"
