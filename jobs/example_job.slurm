#!/bin/bash
#SBATCH --job-name=koa-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=koa-test-%j.out

set -euo pipefail

REMOTE_WORKDIR="${KOA_ML_WORKDIR:-$HOME/koa-ml}"
if [ ! -d "$REMOTE_WORKDIR" ]; then
  echo "Remote workdir not found at $REMOTE_WORKDIR" >&2
  exit 1
fi
cd "$REMOTE_WORKDIR"


echo "KOA test job running on $(hostname)"
echo "==== GPU info ===="
nvidia-smi || echo "nvidia-smi not available"

echo "Sleeping for 60 seconds to keep the allocation."
sleep 60

echo "Job complete."