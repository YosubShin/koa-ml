#!/bin/bash
#SBATCH --job-name=koa-long-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=koa-long-test-%j.out

set -euo pipefail

REMOTE_WORKDIR="${KOA_ML_WORKDIR:-$HOME/koa-ml}"
if [ ! -d "$REMOTE_WORKDIR" ]; then
  echo "Remote workdir not found at $REMOTE_WORKDIR" >&2
  exit 1
fi
cd "$REMOTE_WORKDIR"


echo "KOA long test job running on $(hostname) at $(date)"
echo "==== GPU info ===="
nvidia-smi || echo "nvidia-smi not available"

echo "Sleeping for 20 minutes to keep the allocation active..."
sleep 1200

echo "Long test job complete at $(date)"