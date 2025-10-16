#!/bin/bash
#SBATCH --job-name=koa-long-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=koa-long-test-%j.out

set -euo pipefail

echo "KOA long test job running on $(hostname) at $(date)"
echo "==== GPU info ===="
nvidia-smi || echo "nvidia-smi not available"

echo "Sleeping for 20 minutes to keep the allocation active..."
sleep 1200

echo "Long test job complete at $(date)"