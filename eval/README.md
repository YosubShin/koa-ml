# Evaluating Models on KOA

This directory contains configurations and scripts for evaluating language models on standard benchmarks using the KOA HPC cluster.

## Quick Start

### 1. Install Dependencies

```bash
# On your local machine
pip install -e ".[ml]"

# Or install only evaluation dependencies
pip install -e ".[eval]"
```

### 2. Test Locally (Optional)

```bash
# Quick test with SmolLM
python eval/evaluate.py --config eval/configs/quickstart.yaml
```

### 3. Submit to KOA

```bash
# Quick test evaluation
koa-ml submit jobs/eval_quickstart.slurm

# Full MMLU evaluation
koa-ml submit jobs/eval_mmlu.slurm
```

## Available Benchmarks

All benchmark configs are in `eval/configs/benchmarks/`:

### MMLU (Massive Multitask Language Understanding)
- **File**: [mmlu.yaml](configs/benchmarks/mmlu.yaml)
- **What it tests**: General knowledge across 57 academic subjects
- **Time**: ~2 hours for full MMLU
- **Standard setup**: 5-shot evaluation
- **Use case**: Overall model knowledge assessment

### GSM8K (Grade School Math)
- **File**: [gsm8k.yaml](configs/benchmarks/gsm8k.yaml)
- **What it tests**: Mathematical reasoning and arithmetic
- **Time**: ~30 minutes
- **Standard setup**: 8-shot evaluation
- **Use case**: Math ability assessment

### HellaSwag
- **File**: [hellaswag.yaml](configs/benchmarks/hellaswag.yaml)
- **What it tests**: Commonsense reasoning
- **Time**: ~1 hour
- **Standard setup**: 10-shot evaluation
- **Use case**: Common sense understanding

### TruthfulQA
- **File**: [truthfulqa.yaml](configs/benchmarks/truthfulqa.yaml)
- **What it tests**: Truthfulness and avoiding misconceptions
- **Time**: ~30 minutes
- **Standard setup**: 0-shot evaluation
- **Use case**: Factuality assessment

### ARC (AI2 Reasoning Challenge)
- **File**: [arc.yaml](configs/benchmarks/arc.yaml)
- **What it tests**: Science reasoning
- **Time**: ~1 hour
- **Standard setup**: 25-shot evaluation
- **Use case**: Scientific reasoning

### Quickstart (Testing)
- **File**: [quickstart.yaml](configs/quickstart.yaml)
- **What it tests**: Single MMLU subtask with SmolLM
- **Time**: ~5 minutes
- **Use case**: Pipeline validation

## Config Structure

Each benchmark config has 3 sections:

### 1. Model Section
```yaml
model:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  # Or evaluate your fine-tuned model:
  # model_name: "./output/llama8b_lora"
  model_max_length: 2048
  dtype: "bfloat16"
  attn_implementation: "flash_attention_2"
```

### 2. Generation Section
```yaml
generation:
  per_device_batch_size: 4
  temperature: 0.0      # Greedy decoding for evaluation
  max_new_tokens: 512
```

### 3. Tasks Section
```yaml
tasks:
  - backend: "lm_harness"
    task: "mmlu"
    num_fewshot: 5
    output_path: "./eval_results/mmlu"
```

## Evaluating Your Fine-Tuned Models

### Option 1: Edit the config file

```yaml
model:
  model_name: "./output/llama8b_lora"  # Path to your checkpoint
```

### Option 2: Use CLI override

```bash
python eval/evaluate.py \
  --config eval/configs/benchmarks/mmlu.yaml \
  --model ./output/llama8b_lora
```

### Option 3: Direct CLI evaluation

```bash
python eval/evaluate.py \
  --model ./output/llama8b_lora \
  --tasks mmlu,gsm8k,hellaswag \
  --num_fewshot 5
```

## CLI Options

```bash
# Basic usage
python eval/evaluate.py --config <config_file>

# Override model
python eval/evaluate.py --config <config_file> --model ./my_model

# Override tasks
python eval/evaluate.py --config <config_file> --tasks mmlu,gsm8k

# Quick test (limit examples)
python eval/evaluate.py --config <config_file> --limit 10

# Change few-shot count
python eval/evaluate.py --config <config_file> --num_fewshot 3

# Custom output path
python eval/evaluate.py --config <config_file> --output_path ./my_results
```

## Understanding Results

Results are saved as JSON files in `eval_results/`:

```json
{
  "results": {
    "mmlu": {
      "acc": 0.5234,
      "acc_stderr": 0.0123,
      "acc_norm": 0.5456,
      "acc_norm_stderr": 0.0115
    }
  },
  "config": {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "num_fewshot": 5
  }
}
```

### Key Metrics

- **acc**: Accuracy (main metric)
- **acc_stderr**: Standard error of accuracy
- **acc_norm**: Normalized accuracy (for multiple choice)
- **exact_match**: Exact string match (for generation tasks)

### Benchmark Scores Reference

Typical scores for reference (approximate):

| Model | MMLU (5-shot) | GSM8K (8-shot) | HellaSwag (10-shot) |
|-------|---------------|----------------|---------------------|
| GPT-4 | ~86% | ~92% | ~95% |
| Llama 3.1 70B | ~79% | ~89% | ~89% |
| Llama 3.1 8B | ~66% | ~79% | ~82% |
| SmolLM 135M | ~25% | ~3% | ~30% |

## Monitoring Jobs on KOA

```bash
# Check job status
koa-ml jobs

# View job output
# SSH to KOA and check logs/eval-*-<job_id>.out
```

## Comparing Models

### Before and After Fine-Tuning

```bash
# Evaluate base model
python eval/evaluate.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tasks mmlu,gsm8k \
  --output_path ./results/base_model

# Evaluate fine-tuned model
python eval/evaluate.py \
  --model ./output/llama8b_lora \
  --tasks mmlu,gsm8k \
  --output_path ./results/finetuned_model

# Compare results manually or with a script
```

## Available Tasks in lm-evaluation-harness

You can evaluate on any task from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

**Reasoning & Knowledge**:
- `mmlu`, `mmlu_abstract_algebra`, `mmlu_anatomy`, etc. (57 subtasks)
- `arc_easy`, `arc_challenge`
- `hellaswag`, `winogrande`, `piqa`

**Math**:
- `gsm8k`, `math`

**Truthfulness**:
- `truthfulqa_mc1`, `truthfulqa_mc2`

**Code**:
- `humaneval`, `mbpp`

**Multilingual**:
- `xcopa`, `xnli`, `xwinograd`

To see all available tasks:
```bash
python -c "from lm_eval import tasks; print(tasks.ALL_TASKS)"
```

## Creating Custom Benchmark Configs

Copy an existing config and modify:

```bash
cp eval/configs/benchmarks/mmlu.yaml eval/configs/benchmarks/my_benchmark.yaml
```

Edit the tasks section:
```yaml
tasks:
  - backend: "lm_harness"
    task: "my_custom_task"
    num_fewshot: 5
    output_path: "./eval_results/my_benchmark"
```

## Tips for Evaluation

1. **Start small**: Use quickstart config to validate setup
2. **Use temperature=0**: For reproducible evaluation
3. **Match few-shot counts**: Use standard counts for comparability
4. **Batch size**: Increase if you have memory headroom
5. **Save results**: Results are saved automatically as JSON

## Troubleshooting

### Out of Memory

1. Reduce `per_device_batch_size`
2. Reduce `model_max_length`
3. Use a smaller model for testing

### Slow Evaluation

1. Increase batch size if memory allows
2. Ensure flash attention is enabled
3. Use `--limit` for quick tests

### Task Not Found

Check available tasks:
```bash
python -c "from lm_eval import tasks; print([t for t in tasks.ALL_TASKS if 'mmlu' in t])"
```

## Evaluation on KOA

### Typical Resource Requirements

| Benchmark | Time | GPU Memory | Recommended GPU |
|-----------|------|------------|-----------------|
| Quickstart | 5-10 min | 8GB | Any |
| MMLU (full) | 2 hours | 16GB | RTX A4000+ |
| GSM8K | 30 min | 16GB | RTX A4000+ |
| HellaSwag | 1 hour | 16GB | RTX A4000+ |

### Creating Custom SLURM Jobs

Copy an existing job script:
```bash
cp jobs/eval_mmlu.slurm jobs/eval_custom.slurm
```

Edit the evaluation command:
```bash
python eval/evaluate.py \
    --model ./output/my_model \
    --tasks mmlu,gsm8k,hellaswag
```

Submit to KOA:
```bash
koa-ml submit jobs/eval_custom.slurm
```

## Next Steps

- Evaluate your fine-tuned models
- Compare before/after fine-tuning
- Try different benchmarks
- Create custom evaluation suites
- Track results over training iterations
