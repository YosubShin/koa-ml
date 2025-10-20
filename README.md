# koa-ml

---

Utilities for managing KOA HPC GPU jobs and fine-tuning/evaluating language models from your local workstation.

## Features

- **Job Management**: Submit, monitor, and cancel jobs on KOA
- **Fine-Tuning**: LoRA, QLoRA, and full fine-tuning with config files
- **Evaluation**: Standard benchmarks (MMLU, GSM8K, HellaSwag, etc.)
- **SLURM Integration**: Ready-to-use job templates
- **HuggingFace Integration**: Seamless model and dataset loading

## Quick Start

### 1. Setup KOA Connection

In order to avoid getting prompted with MFA authorization every time, add this to `~/.ssh/config`.
```
Host koa koa.its.hawaii.edu
  HostName koa.its.hawaii.edu
  User <user_id>
  IdentitiesOnly yes
  IdentityFile ~/.ssh/<ssh_key_path>
  ControlMaster auto
  ControlPath ~/.ssh/cm-%C
  ControlPersist 1h
  ServerAliveInterval 60
  ServerAliveCountMax 3
  TCPKeepAlive yes
  StrictHostKeyChecking accept-new
```

```bash
# Create Python environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .

# Configure KOA credentials
cp .koa-config.example.yaml ~/.config/koa-ml/config.yaml
# Edit config.yaml with your KOA username, host, and SSH key

# Verify connectivity
koa-ml check
```

### KOA Environment Setup

When preparing the environment directly on KOA, start an interactive session if needed (e.g. `srun -p gpu-sandbox --gres=gpu:1 --mem=8G -t 0-00:30 --pty /bin/bash`) and run:

```bash
cd ~/koa-ml
source scripts/setup_koa_env.sh              # defaults to Python 3.11.5 + cuda/12.1
# Optional overrides:
#   PYTHON_MODULE=lang/Python/3.10.8-GCCcore-12.2.0 source scripts/setup_koa_env.sh
#   CUDA_MODULE=cuda/12.4 source scripts/setup_koa_env.sh
```

The script automatically loads a KOA Python module (3.11 by default with fallbacks), attempts to load `system/CUDA/12.2.0`, recreates `.venv`, installs PyTorch (CUDA 12.x build), and installs the project with `.[ml]` extras.

If `flash-attn` fails to build (missing `nvcc`/CUDA headers), the script retries automatically without that dependency. To skip it entirely, run `INSTALL_FLASH_ATTN=0 source scripts/setup_koa_env.sh`.

### 2. Install ML Dependencies (Optional)

```bash
# For fine-tuning and evaluation
pip install -e ".[ml]"

# Or install separately:
# pip install -e ".[train]"  # Just training
# pip install -e ".[eval]"   # Just evaluation

# Note: GPU-only extras (`bitsandbytes`, `flash-attn`) are available on Linux.
# On macOS these optional packages are skipped automatically.
```

### 3. Try It Out

```bash
# Submit a quick GPU test
koa-ml submit jobs/example_long_job.slurm

# Fine-tune a model (requires ML dependencies)
koa-ml submit jobs/tune_smollm_quickstart.slurm

# Evaluate on MMLU benchmark (requires ML dependencies)
koa-ml submit jobs/eval_quickstart.slurm

# Check job status
koa-ml jobs
```

## Documentation

### Getting Started
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - 5-minute guide to fine-tuning & evaluation
- **[docs/QWEN3_QUICKREF.md](docs/QWEN3_QUICKREF.md)** - One-page Qwen3 reference

### Detailed Guides
- **[docs/QWEN3_GUIDE.md](docs/QWEN3_GUIDE.md)** - Complete Qwen3 guide
- **[docs/ML_GUIDE.md](docs/ML_GUIDE.md)** - Complete ML workflow documentation
- **[tune/README.md](tune/README.md)** - Fine-tuning details
- **[eval/README.md](eval/README.md)** - Evaluation details

### Development
- **[docs/TESTING.md](docs/TESTING.md)** - Testing instructions

## CLI usage

After installation you can run the CLI as `koa-ml <command>` (the `koa_ml` alias also works if you prefer underscores).

- `koa-ml check` — run a lightweight KOA health check (`hostname`, `sinfo`)
- `koa-ml submit jobs/example_job.slurm --partition gpu --gpus 1` — copy the job script to KOA, submit it with `sbatch`, and print the job id
- `koa-ml jobs` — list your active jobs using `squeue`
- `koa-ml cancel <job_id>` — cancel an active job (`scancel`)
- `koa-ml refresh` — rsync the current directory to the remote workdir (defaults to `~/koa-ml`)

Each command accepts `--config` if you want to point at a custom configuration path.

## Examples

### Fine-Tuning

```bash
# Quick test with SmolLM (30 min)
koa-ml submit jobs/tune_smollm_quickstart.slurm

# Llama 8B with LoRA (12 hours)
koa-ml submit jobs/tune_llama8b_lora.slurm

# Memory-efficient QLoRA
koa-ml submit jobs/tune_llama8b_qlora.slurm
```

### Evaluation

```bash
# Quick test
koa-ml submit jobs/eval_quickstart.slurm

# Full MMLU benchmark
koa-ml submit jobs/eval_mmlu.slurm

# Evaluate your fine-tuned model
python eval/evaluate.py \
  --model ./output/llama8b_lora \
  --tasks mmlu,gsm8k,hellaswag

# Multimodal M2SV evaluation (Qwen3-VL)
koa-ml submit jobs/eval_qwen3_vl_m2sv.slurm
```

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run unit tests
pytest tests/ -m "not integration" -v
```

See [docs/TESTING.md](docs/TESTING.md) for details.

## Project Structure

```
koa-ml/
├── docs/                      # Documentation
│   ├── QUICKSTART.md         # 5-minute start guide
│   ├── QWEN3_QUICKREF.md     # Qwen3 quick reference
│   ├── QWEN3_GUIDE.md        # Complete Qwen3 guide
│   ├── ML_GUIDE.md           # ML workflow guide
│   └── TESTING.md            # Testing guide
├── tune/                      # Fine-tuning system
│   ├── configs/models/       # Training configs (Qwen3, Llama, etc.)
│   ├── train.py              # Training script
│   └── README.md             # Fine-tuning guide
├── eval/                      # Evaluation system
│   ├── configs/benchmarks/   # Benchmark configs (MMLU, GSM8K, etc.)
│   ├── evaluate.py           # Evaluation script
│   └── README.md             # Evaluation guide
├── jobs/                      # SLURM job templates
│   ├── tune_*.slurm          # Training jobs
│   └── eval_*.slurm          # Evaluation jobs
└── src/koa_ml/               # Job management CLI
```

## KOA quick reference

- Login node: `koa.its.hawaii.edu` (SSH/MFA required); Open OnDemand is available at https://koa.its.hawaii.edu
- Storage: `/home/<user>` (50 GB, snapshotted daily) and `/mnt/lustre/koa/scratch/<user>` (90-day purge); copy results off scratch promptly
- Common partitions: `gpu` (up to 3 days), `gpu-sandbox` (4 h, unique job slot), `sandbox` (CPU, 4 h), `shared` (3 d), `shared-long` (7 d)
- Slurm commands of interest: `sinfo`, `squeue -u <user>`, `sbatch`, `srun`, `scancel`
- Interactive test session: `srun -I30 -p sandbox -N 1 -c 1 --mem=6G -t 0-01:00:00 --pty /bin/bash`
- Support: uh-hpc-help@lists.hawaii.edu (include job id, script path, error/output files when reporting issues)

## Getting Help

- **Documentation**: See guides in [docs/](docs/)
- **Quick Reference**: [docs/QWEN3_QUICKREF.md](docs/QWEN3_QUICKREF.md)
- **KOA Support**: uh-hpc-help@lists.hawaii.edu
- **Issues**: Open an issue in this repository

## Acknowledgments

ML features inspired by:
- [oumi-ai/oumi](https://github.com/oumi-ai/oumi) - Training framework
- [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) - Evaluation patterns
- [ThinkingMachinesLab/tinker-cookbook](https://github.com/ThinkingMachinesLab/tinker-cookbook) - Recipe structure
# Notes for KOA jobs
- Job scripts automatically activate the KOA virtualenv at `$HOME/koa-ml/.venv`. Override with `KOA_ML_VENV=/path/to/venv koa-ml submit ...` if you keep the env elsewhere.
- Jobs default to loading `lang/Python/3.11.5-GCCcore-13.2.0` (and no CUDA module). Override with `KOA_PYTHON_MODULE=` / `KOA_CUDA_MODULE=`.
- Scripts `cd` into `$HOME/koa-ml` automatically; change with `KOA_ML_WORKDIR=/path/on/koa`.
