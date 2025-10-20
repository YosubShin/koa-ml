#!/bin/bash
#SBATCH --job-name=eval-qwen3-vl-8b-instruct-m2sv
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/eval-qwen3-vl-8b-instruct-m2sv-%j.out

set -euo pipefail

REMOTE_WORKDIR="${KOA_ML_WORKDIR:-$HOME/koa-ml}"
if [ ! -d "$REMOTE_WORKDIR" ]; then
  echo "Remote workdir not found at $REMOTE_WORKDIR" >&2
  exit 1
fi
cd "$REMOTE_WORKDIR"

echo "======================================"
echo "Evaluating Qwen3-VL 8B Instruct on M2SV"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Load required modules (override with KOA_PYTHON_MODULE / KOA_CUDA_MODULE)
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
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_TELEMETRY=1
VENV_PATH="${KOA_ML_VENV:-$HOME/koa-ml/.venv}"
if [ ! -d "$VENV_PATH" ]; then
  echo "Virtualenv not found at $VENV_PATH" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "==== GPU Info ===="
nvidia-smi
echo ""

echo "==== Python Environment ===="
which python
python --version
echo ""

echo "==== Starting vLLM server (background) ===="
PORT=${VLLM_PORT:-8000}
# Use dedicated vLLM venv if available
export KOA_VLLM_VENV="${KOA_VLLM_VENV:-$HOME/koa-ml/.venv-vllm}"
MODEL="Qwen/Qwen3-VL-8B-Instruct"
scripts/run_vllm_server.sh --model "${MODEL}" --port "${PORT}" --tp ${VLLM_TP:-1} --gpu-mem-util ${VLLM_GPU_MEM_UTIL:-0.92} &
VLLM_PID=$!

echo "Waiting for vLLM to become ready on port ${PORT}..."
RETRIES=300
until curl -sSf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  sleep 2
  RETRIES=$((RETRIES-1))
  if [ ${RETRIES} -le 0 ]; then
    echo "vLLM did not become ready in time." >&2
    kill ${VLLM_PID} || true
    exit 1
  fi
done

export KOA_EVAL_BACKEND=vllm
export VLLM_API_BASE="http://127.0.0.1:${PORT}"
export VLLM_MODEL="${MODEL}"

echo "==== Starting Evaluation (vLLM) ===="
python eval/qwen3_vl_eval.py \
  --config eval/configs/qwen3_vl_8b_instruct_m2sv.yaml || STATUS=$?

echo "Stopping vLLM server (PID ${VLLM_PID})..."
kill ${VLLM_PID} || true
wait ${VLLM_PID} || true
exit ${STATUS:-0}

echo ""
echo "======================================"
echo "Evaluation Qwen3-VL 8B Instruct on M2SV complete!"
echo "Ended: $(date)"
echo "======================================"
