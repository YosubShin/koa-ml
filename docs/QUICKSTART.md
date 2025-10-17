# Quick Start: Fine-Tuning & Evaluation on KOA

Get started with fine-tuning and evaluating models on KOA in 5 minutes.

## Setup (One Time)

```bash
# 1. Install dependencies
source .venv/bin/activate
pip install -e ".[ml]"
```

> Setting up on KOA? Start an interactive session (`srun -p gpu-sandbox --gres=gpu:1 --mem=8G -t 0-00:30 --pty /bin/bash`), then run
> `source scripts/setup_koa_env.sh` inside the repo. Override modules if needed (e.g.
> `PYTHON_MODULE=lang/Python/3.10.8-GCCcore-12.2.0 CUDA_MODULE=cuda/12.4 source scripts/setup_koa_env.sh`).
> If `flash-attn` fails to compile, the script automatically retries without it.
> The SLURM jobs expect the venv at `$HOME/koa-ml/.venv` (set `KOA_ML_VENV` to override), load `lang/Python/3.11.5-GCCcore-13.2.0` by default (`KOA_PYTHON_MODULE`), and execute from `$HOME/koa-ml` unless `KOA_ML_WORKDIR` is provided.

This installs:
- PyTorch, Transformers, Accelerate (core ML)
- PEFT, TRL (fine-tuning)
- lm-eval (benchmarking)
- bitsandbytes, flash-attn (optimizations; Linux only extras and skipped on macOS)

## Test Locally (Optional)

```bash
# Quick training test (requires GPU)
python tune/train.py \
  --config tune/configs/models/smollm_135m_lora.yaml \
  --max_steps 10

# Quick evaluation test (requires GPU)
python eval/evaluate.py \
  --config eval/configs/quickstart.yaml \
  --limit 10
```

## Run on KOA

### Fine-Tuning

```bash
# Quick test (30 min)
koa-ml submit jobs/tune_smollm_quickstart.slurm

# Llama 8B LoRA (12 hours)
koa-ml submit jobs/tune_llama8b_lora.slurm

# Llama 8B QLoRA - memory efficient (12 hours)
koa-ml submit jobs/tune_llama8b_qlora.slurm
```

### Evaluation

```bash
# Quick test (30 min)
koa-ml submit jobs/eval_quickstart.slurm

# Full MMLU benchmark (2 hours)
koa-ml submit jobs/eval_mmlu.slurm
```

### Monitor Jobs

```bash
# Check job status
koa-ml jobs

# Cancel a job
koa-ml cancel <job_id>
```

## What Gets Created

After training:
```
output/
└── smollm_quickstart_<job_id>/
    ├── adapter_model.safetensors  # LoRA weights
    ├── adapter_config.json        # LoRA config
    └── tokenizer files...
```

After evaluation:
```
eval_results/
└── mmlu/
    └── mmlu_results.json          # Benchmark scores
```

Job logs:
```
logs/
├── tune-smollm-<job_id>.out      # Training logs
└── eval-quickstart-<job_id>.out  # Evaluation logs
```

## Next Steps

### Customize Training

Edit config files in [tune/configs/models/](tune/configs/models/):

```yaml
# Change dataset
data:
  train_dataset:
    dataset_name: "your-username/your-dataset"

# Adjust training
training:
  learning_rate: 2.0e-04
  num_train_epochs: 3

# Tweak LoRA
peft:
  lora_r: 16      # Higher = more parameters
  lora_alpha: 32
```

### Evaluate Your Model

```bash
# Option 1: Edit config file
# Edit eval/configs/benchmarks/mmlu.yaml
# Change model_name to "./output/your_model"

# Option 2: Use CLI
python eval/evaluate.py \
  --model ./output/llama8b_lora \
  --tasks mmlu,gsm8k,hellaswag
```

### Try Different Models

SmolLM (testing):
```yaml
model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
```

Llama 3.1 (production):
```yaml
model_name: "meta-llama/Llama-3.1-8B-Instruct"
```

Qwen (alternative):
```yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"
```

## Available Configs

### Training Configs ([tune/configs/models/](tune/configs/models/))
- `smollm_135m_lora.yaml` - Quick testing (10 min)
- `llama_8b_lora.yaml` - Standard LoRA
- `llama_8b_qlora.yaml` - Memory-efficient
- `llama_8b_full.yaml` - Full fine-tuning

### Evaluation Configs ([eval/configs/benchmarks/](eval/configs/benchmarks/))
- `quickstart.yaml` - Quick test
- `mmlu.yaml` - Knowledge (57 subjects)
- `gsm8k.yaml` - Math reasoning
- `hellaswag.yaml` - Commonsense
- `truthfulqa.yaml` - Truthfulness
- `arc.yaml` - Science reasoning

## Detailed Guides

- [ML_GUIDE.md](ML_GUIDE.md) - Complete workflow guide
- [tune/README.md](tune/README.md) - Fine-tuning details
- [eval/README.md](eval/README.md) - Evaluation details

## Common Commands

```bash
# Training
python tune/train.py --config tune/configs/models/llama_8b_lora.yaml
python tune/train.py --config <config> --output_dir ./my_output
python tune/train.py --config <config> --max_steps 100  # Quick test

# Evaluation
python eval/evaluate.py --config eval/configs/benchmarks/mmlu.yaml
python eval/evaluate.py --model ./output/my_model --tasks mmlu,gsm8k
python eval/evaluate.py --config <config> --limit 10  # Quick test

# KOA job management
koa-ml submit <job.slurm>
koa-ml jobs
koa-ml cancel <job_id>
koa-ml check
```

## Tips

1. **Start small**: Test with SmolLM before expensive runs
2. **Use QLoRA**: If you hit memory issues
3. **Check logs**: SSH to KOA and check `logs/` directory
4. **Save configs**: Commit your configs to git for reproducibility
5. **Monitor training**: Look for steady loss decrease in logs

## Troubleshooting

**Out of memory**: Switch to QLoRA or reduce batch size

**Job killed**: Increase time limit in SLURM script

**Dataset error**: Check dataset format matches expected columns

**Slow training**: Ensure flash attention is enabled

For more help, see the detailed guides or open an issue!
