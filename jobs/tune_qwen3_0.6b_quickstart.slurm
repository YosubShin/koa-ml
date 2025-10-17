#!/bin/bash
#SBATCH --job-name=tune-qwen3-0.6b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/tune-qwen3-0.6b-%j.out

set -euo pipefail

REMOTE_WORKDIR="${KOA_ML_WORKDIR:-$HOME/koa-ml}"
if [ ! -d "$REMOTE_WORKDIR" ]; then
  echo "Remote workdir not found at $REMOTE_WORKDIR" >&2
  exit 1
fi
cd "$REMOTE_WORKDIR"


echo "======================================"
echo "Fine-tuning Qwen3 0.6B (Quick Test)"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Load modules (adjust based on KOA's available modules)
# module load python/3.11
# module load cuda/12.1

# Activate virtual environment
VENV_PATH="${KOA_ML_VENV:-$HOME/koa-ml/.venv}"
if [ ! -d "$VENV_PATH" ]; then
  echo "Virtualenv not found at $VENV_PATH" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

# GPU info
echo "==== GPU Info ===="
nvidia-smi
echo ""

# Python environment
echo "==== Python Environment ===="
which python
python --version
echo ""

# Training
echo "==== Starting Training ===="
python tune/train.py \
    --config tune/configs/models/qwen3_0.6b_lora.yaml \
    --output_dir ./output/qwen3_0.6b_quickstart_${SLURM_JOB_ID}

echo ""
echo "======================================"
echo "Training complete!"
echo "Ended: $(date)"
echo "======================================"
