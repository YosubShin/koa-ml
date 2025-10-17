#!/bin/bash
#SBATCH --job-name=tune-qwen3-8b-qlora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/tune-qwen3-8b-qlora-%j.out

set -euo pipefail

REMOTE_WORKDIR="${KOA_ML_WORKDIR:-$HOME/koa-ml}"
if [ ! -d "$REMOTE_WORKDIR" ]; then
  echo "Remote workdir not found at $REMOTE_WORKDIR" >&2
  exit 1
fi
cd "$REMOTE_WORKDIR"


echo "======================================"
echo "Fine-tuning Qwen3 8B with QLoRA"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Load modules
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

# Training
echo "==== Starting Training ===="
python tune/train.py \
    --config tune/configs/models/qwen3_8b_qlora.yaml \
    --output_dir ./output/qwen3_8b_qlora_${SLURM_JOB_ID}

echo ""
echo "======================================"
echo "Training complete!"
echo "Ended: $(date)"
echo "======================================"
