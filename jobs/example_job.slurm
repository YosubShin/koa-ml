#!/bin/bash
#SBATCH --job-name=koa-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=koa-test-%j.out

set -euo pipefail

echo "KOA test job running on $(hostname)"
echo "==== GPU info ===="
nvidia-smi || echo "nvidia-smi not available"

echo "Sleeping for 60 seconds to keep the allocation."
sleep 60

echo "Job complete."