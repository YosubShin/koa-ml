#!/usr/bin/env python3
"""
Evaluation script for language models using lm-evaluation-harness.
Supports standard HuggingFace benchmarks (MMLU, GSM8K, HellaSwag, etc.)

Usage:
    python eval/evaluate.py --config eval/configs/quickstart.yaml
    python eval/evaluate.py --config eval/configs/benchmarks/mmlu.yaml
    python eval/evaluate.py --model ./output/llama8b_lora --tasks mmlu,gsm8k
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from lm_eval import evaluator, tasks as lm_eval_tasks
from lm_eval.models.huggingface import HFLM


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict):
    """Create model wrapper for lm-evaluation-harness."""
    model_config = config['model']
    gen_config = config.get('generation', {})

    model_args = {
        'pretrained': model_config['model_name'],
        'dtype': model_config.get('dtype', 'bfloat16'),
        'batch_size': gen_config.get('per_device_batch_size', 'auto'),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'trust_remote_code': True,
    }

    # Add attention implementation if specified
    if model_config.get('attn_implementation'):
        model_args['attn_implementation'] = model_config['attn_implementation']

    # Create model wrapper
    model = HFLM(**model_args)

    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate language models')
    parser.add_argument('-c', '--config', help='Path to config YAML file')
    parser.add_argument('--model', help='Override model name or path')
    parser.add_argument('--tasks', help='Comma-separated task names (overrides config)')
    parser.add_argument('--num_fewshot', type=int, help='Number of few-shot examples')
    parser.add_argument('--output_path', help='Override output path')
    parser.add_argument('--limit', type=int, help='Limit number of examples (for testing)')
    args = parser.parse_args()

    # Load config if provided
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    else:
        # Minimal config if using CLI args only
        config = {
            'model': {'model_name': args.model or 'meta-llama/Llama-3.1-8B-Instruct'},
            'generation': {},
            'tasks': []
        }

    # Apply CLI overrides
    if args.model:
        config['model']['model_name'] = args.model

    # Build task list
    if args.tasks:
        # Override with CLI tasks
        task_names = args.tasks.split(',')
        tasks_to_run = []
        for task_name in task_names:
            task_dict = {
                'backend': 'lm_harness',
                'task': task_name.strip(),
                'num_fewshot': args.num_fewshot or 0,
            }
            tasks_to_run.append(task_dict)
    else:
        # Use tasks from config
        tasks_to_run = config.get('tasks', [])

    if not tasks_to_run:
        print("Error: No tasks specified. Use --tasks or provide a config with tasks.")
        return 1

    # Create model
    print("\nLoading model...")
    print(f"Model: {config['model']['model_name']}")
    model = create_model(config)

    # Run evaluation for each task
    all_results = {}

    for task_config in tasks_to_run:
        task_name = task_config['task']
        num_fewshot = args.num_fewshot if args.num_fewshot is not None else task_config.get('num_fewshot', 0)
        output_path = args.output_path or task_config.get('output_path', './eval_results')

        print("\n" + "="*50)
        print(f"Evaluating: {task_name}")
        print(f"Few-shot: {num_fewshot}")
        print("="*50 + "\n")

        # Run evaluation
        results = evaluator.simple_evaluate(
            model=model,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            limit=args.limit,
            bootstrap_iters=100,
        )

        # Store results
        all_results[task_name] = results

        # Print results
        print("\n" + "-"*50)
        print(f"Results for {task_name}:")
        print("-"*50)

        if 'results' in results:
            task_results = results['results'].get(task_name, {})
            for metric, value in task_results.items():
                if not metric.endswith('_stderr'):
                    stderr_key = f"{metric}_stderr"
                    stderr = task_results.get(stderr_key, 0)
                    print(f"  {metric}: {value:.4f} ± {stderr:.4f}")

        # Save results to file
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path) / f"{task_name}_results.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    # Save combined results
    if len(all_results) > 1:
        combined_output = Path(args.output_path or './eval_results') / 'combined_results.json'
        with open(combined_output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Combined results saved to: {combined_output}")

    print("\n" + "="*50)
    print("Evaluation complete!")
    print("="*50)


if __name__ == '__main__':
    main()
