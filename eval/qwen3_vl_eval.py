#!/usr/bin/env python3
"""
Evaluation script for Qwen3-VL models on image + text multiple-choice datasets.

Usage:
    python eval/qwen3_vl_eval.py --config eval/configs/qwen3_vl_m2sv.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import asyncio
import base64
import io
import time

import pandas as pd
import torch
import yaml
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
import re

Image.MAX_IMAGE_PIXELS = None

# Disable HF Transfer fallback (not reliable on KOA)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_HF_TRANSFER", "1")

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

@dataclass
class EvalConfig:
    model_name: str
    dtype: str = "float16"
    device_map: str = "auto"
    dataset_name: str = "yosubshin/m2sv"
    dataset_split: str = "train"
    generation_max_new_tokens: int = 128
    generation_temperature: float = 0.1
    limit: Optional[int] = None
    output_dir: str = "./eval_results/qwen3_vl_m2sv"
    save_predictions: bool = True
    # Inference backend
    backend: str = "hf"  # "hf" | "vllm"
    # vLLM settings (OpenAI-compatible API)
    vllm_api_base: Optional[str] = None
    vllm_api_key: Optional[str] = None
    vllm_model: Optional[str] = None  # override model name for server if needed
    vllm_max_concurrency: int = 8
    vllm_request_timeout_s: float = 120.0
    vllm_max_retries: int = 3


def load_config(path: str) -> EvalConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    model_cfg = raw.get("model", {})
    dataset_cfg = raw.get("dataset", {})
    generation_cfg = raw.get("generation", {})
    output_cfg = raw.get("output", {})

    inference_cfg = raw.get("inference", {}) or {}
    vllm_cfg = inference_cfg.get("vllm", {}) or {}

    cfg = EvalConfig(
        model_name=model_cfg["model_name"],
        dtype=model_cfg.get("dtype", "float16"),
        device_map=model_cfg.get("device_map", "auto"),
        dataset_name=dataset_cfg.get("name", "yosubshin/m2sv"),
        dataset_split=dataset_cfg.get("split", "train"),
        generation_max_new_tokens=generation_cfg.get("max_new_tokens", 128),
        generation_temperature=generation_cfg.get("temperature", 0.1),
        limit=generation_cfg.get("limit"),
        output_dir=output_cfg.get("dir", "./eval_results/qwen3_vl_m2sv"),
        save_predictions=output_cfg.get("save_predictions", True),
        backend=(inference_cfg.get("backend") or "hf").lower(),
        vllm_api_base=vllm_cfg.get("api_base"),
        vllm_api_key=vllm_cfg.get("api_key"),
        vllm_model=vllm_cfg.get("model"),
        vllm_max_concurrency=int(vllm_cfg.get("max_concurrency", 8) or 8),
        vllm_request_timeout_s=float(vllm_cfg.get("request_timeout_s", 120.0) or 120.0),
        vllm_max_retries=int(vllm_cfg.get("max_retries", 3) or 3),
    )
    return cfg


def to_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = mapping.get(name.lower())
    if dtype is None:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {list(mapping.keys())}")
    return dtype


def safe_load_image(image_obj: Any) -> Optional[Image.Image]:
    if image_obj is None:
        return None
    if isinstance(image_obj, Image.Image):
        img_copy = image_obj.copy()
        image_obj.close()
        return img_copy.convert("RGB")
    return image_obj


def format_prompt(question: str, options: List[str]) -> str:
    system_prompt = "You will be given two images concatenated side by side: (1) a north-up overhead map with arrows labeled A, B, C, ... and (2) a street-view photo.\nRules:\n- The camera location is the same for all options: the center of the intersection.\n- Each letter corresponds to facing outward from that center along the arrow of that label.\n- The small circles near labels are markers only; they are not camera locations.\n- The map and photo may be captured years apart. Ignore transient objects (cars, people).\nOn the final line, output only: Final answer: \\boxed{X} where X is a single letter (A, B, C, ...)."
    prompt_lines = [system_prompt, question]
    return "\n".join(prompt_lines)


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    # Use PNG to avoid JPEG artifacts and ensure broad compatibility
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def normalize_letter(text: str, num_options: int) -> str:
    """Return a single option letter if confidently present.

    Priority:
    1) Exact single-letter response (ignoring surrounding whitespace).
    2) Letter inside \boxed{X} (case-insensitive).
    3) Explicit conclusion phrases like "answer is X" or "final answer: X" (also supports "is:").
    4) Last non-empty line is effectively just a styled single letter (e.g., **B**, (C), `A`, "C.").
    5) As a weaker fallback, accept phrases like "choose X", "option X", "arrow X" unless preceded by elimination/negation context.
    Otherwise returns empty string to avoid false positives from prose.
    """
    if text is None:
        return ""
    t = text.strip()
    if not t:
        return ""

    def is_valid_letter(ch: str) -> str:
        if not ch:
            return ""
        ch_u = ch.upper()
        idx = ord(ch_u) - ord("A")
        return ch_u if 0 <= idx < num_options else ""

    # 1) Exact single letter
    m = re.fullmatch(r"\s*([A-Za-z])\s*", t)
    if m:
        ch = is_valid_letter(m.group(1))
        if ch:
            return ch

    # 2) \boxed{X}
    m = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", t, flags=re.IGNORECASE)
    if m:
        ch = is_valid_letter(m.group(1))
        if ch:
            return ch

    # 2b) Repeated-letter outputs like "C. C" or "B B" as the entire response
    m = re.fullmatch(r"\s*([A-Za-z])\s*[\.-:;,]?\s*\1\s*\.?\s*", t)
    if m:
        ch = is_valid_letter(m.group(1))
        if ch:
            return ch

    # 3) Prefer explicit conclusion phrases anywhere in text (prefer the last such mention)
    explicit_answer_patterns = [
        r"(?:\bthe\s+answer\b|\banswer\b)\s*(?:is\s*[:=]?|[:=])\s*([A-Za-z])\b",
        r"\bfinal\s*(?:answer)?\s*(?:is\s*[:=]?|[:=])\s*([A-Za-z])\b",
    ]
    explicit_candidates: list[str] = []
    for pat in explicit_answer_patterns:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            explicit_candidates.append(m.group(1))
    for raw in reversed(explicit_candidates):
        ch = is_valid_letter(raw)
        if ch:
            return ch

    # 4) Last non-empty line: accept if it's effectively just a single styled letter
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        # If the last line itself contains an explicit phrase, re-use explicit logic on it for precision
        for pat in explicit_answer_patterns:
            m2 = re.search(pat, last, flags=re.IGNORECASE)
            if m2:
                ch = is_valid_letter(m2.group(1))
                if ch:
                    return ch
        # Repeated-letter on last line like "C. C"
        mrep = re.fullmatch(r"\s*([A-Za-z])\s*[\.-:;,]?\s*\1\s*\.?\s*", last)
        if mrep:
            ch = is_valid_letter(mrep.group(1))
            if ch:
                return ch
        # Strip typical wrappers and styling around a lone letter
        stripped = re.sub(r"[\s\*`_~\-–—\(\)\[\]\{\}\"'.:;,!]+", "", last)
        # If what's left is a single letter, accept it
        if re.fullmatch(r"[A-Za-z]", stripped):
            ch = is_valid_letter(stripped)
            if ch:
                return ch

    # 5) Weaker fallback: ambiguous phrases choose/option/arrow X, but avoid elimination contexts
    ambiguous_patterns = [
        r"\bchoose\s*([A-Za-z])\b",
        r"\b(?:option|choice|arrow)\s*([A-Za-z])\b",
    ]
    last_ch = ""
    for pat in ambiguous_patterns:
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            start = m.start()
            context = t[max(0, start-50):start].lower()
            if any(neg in context for neg in ["eliminate", "eliminates", "eliminated", "eliminating", "not ", "isn't", "is not", "avoid", "eliminates option", "eliminate option"]):
                continue
            ch = is_valid_letter(m.group(1))
            if ch:
                last_ch = ch
    if last_ch:
        return last_ch

    return ""


def evaluate_hf(cfg: EvalConfig) -> Dict[str, Any]:
    print("Configuration:")
    print(f"  Model: {cfg.model_name}")
    print(f"  Dataset: {cfg.dataset_name} ({cfg.dataset_split})")
    print(f"  Output dir: {cfg.output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    dtype = to_torch_dtype(cfg.dtype)
    print("\n[1/4] Loading Qwen3-VL model (HF backend)...")
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_name,
        dtype=dtype,
        device_map=cfg.device_map,
    )
    processor = AutoProcessor.from_pretrained(cfg.model_name)

    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters: {params_b:.2f}B")

    print("\n[2/4] Loading dataset...")
    dataset: Dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.limit is not None:
        dataset = dataset.select(range(min(cfg.limit, len(dataset))))
    total = len(dataset)
    print(f"  Samples: {total}")

    results: List[Dict[str, Any]] = []
    correct = 0

    pbar = tqdm(dataset, desc="Evaluating", total=total)

    for idx, item in enumerate(pbar):
        try:
            prompt = format_prompt(item["question"], item["options"])
            image_content: List[Dict[str, Any]] = []

            for key in ("image_sv", "image_map"):
                image = safe_load_image(item.get(key))
                if image is not None:
                    image_content.append({"type": "image", "image": image})

            if not image_content:
                raise ValueError("Sample missing both scene and map images.")

            image_content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": image_content}]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=cfg.generation_max_new_tokens,
                    temperature=cfg.generation_temperature,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], outputs)
            ]

            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            prediction = normalize_letter(response, len(item["options"]))
            ground_truth = item["answer"]
            is_correct = prediction == ground_truth

            if is_correct:
                correct += 1

            results.append(
                {
                    "id": item.get("id", idx),
                    "question": item["question"],
                    "options": item["options"],
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "raw_response": response,
                    "correct": is_correct,
                }
            )

            accuracy = correct / (idx + 1)
            pbar.set_postfix({"accuracy": f"{accuracy:.2%}"})

            if (idx + 1) % 10 == 0:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

        except Exception as exc:
            print(f"\nError on sample {idx}: {exc}")
            results.append(
                {
                    "id": item.get("id", idx),
                    "question": item.get("question"),
                    "options": item.get("options"),
                    "ground_truth": item.get("answer"),
                    "prediction": "ERROR",
                    "raw_response": str(exc),
                    "correct": False,
                }
            )

    pbar.close()

    accuracy = correct / total if total else 0.0
    summary = {
        "model": cfg.model_name,
        "dataset": cfg.dataset_name,
        "split": cfg.dataset_split,
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": float(accuracy),
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "dtype": cfg.dtype,
            "device_map": cfg.device_map,
            "max_new_tokens": cfg.generation_max_new_tokens,
            "temperature": cfg.generation_temperature,
            "limit": cfg.limit,
        },
    }

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    for key, value in summary.items():
        if key != "config":
            print(f"{key.title().replace('_', ' ')}: {value}")
    print("=" * 80)

    if cfg.save_predictions:
        os.makedirs(cfg.output_dir, exist_ok=True)
        results_path = os.path.join(cfg.output_dir, "predictions.csv")
        summary_path = os.path.join(cfg.output_dir, "summary.json")

        pd.DataFrame(results).to_csv(results_path, index=False)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved predictions to: {results_path}")
        print(f"Saved summary to: {summary_path}")

    del model
    del processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return summary


async def _vllm_eval_one(
    client,
    semaphore: asyncio.Semaphore,
    cfg: EvalConfig,
    idx: int,
    item: Dict[str, Any],
    model_name: str,
):
    prompt = format_prompt(item["question"], item["options"])
    image_contents: List[Dict[str, Any]] = []
    for key in ("image_sv", "image_map"):
        image = safe_load_image(item.get(key))
        if image is not None:
            data_url = image_to_data_url(image)
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })
    if not image_contents:
        raise ValueError("Sample missing both scene and map images.")
    content: List[Dict[str, Any]] = []
    content.extend(image_contents)
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": cfg.generation_temperature,
        "max_tokens": cfg.generation_max_new_tokens,
    }

    backoff = 1.0
    for attempt in range(cfg.vllm_max_retries + 1):
        try:
            async with semaphore:
                resp = await client.post(
                    "/v1/chat/completions",
                    json=payload,
                    timeout=cfg.vllm_request_timeout_s,
                )
            resp.raise_for_status()
            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]
            prediction = normalize_letter(response_text, len(item["options"]))
            ground_truth = item["answer"]
            is_correct = prediction == ground_truth
            record = {
                "id": item.get("id", idx),
                "question": item.get("question"),
                "options": item.get("options"),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "raw_response": response_text,
                "correct": is_correct,
            }
            return record
        except Exception as exc:
            if attempt >= cfg.vllm_max_retries:
                return {
                    "id": item.get("id", idx),
                    "question": item.get("question"),
                    "options": item.get("options"),
                    "ground_truth": item.get("answer"),
                    "prediction": "ERROR",
                    "raw_response": str(exc),
                    "correct": False,
                }
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 8.0)


def evaluate_vllm(cfg: EvalConfig) -> Dict[str, Any]:
    print("Configuration:")
    print(f"  Model: {cfg.model_name}")
    print(f"  Dataset: {cfg.dataset_name} ({cfg.dataset_split})")
    print(f"  Output dir: {cfg.output_dir}")
    print("\n[1/3] Using vLLM OpenAI-compatible endpoint...")

    api_base = cfg.vllm_api_base or os.environ.get("VLLM_API_BASE")
    if not api_base:
        raise ValueError("vLLM backend selected but no api_base provided (in config.inference.vllm.api_base or VLLM_API_BASE env)")
    api_key = cfg.vllm_api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
    model_name = cfg.vllm_model or cfg.model_name

    print("\n[2/3] Loading dataset...")
    dataset: Dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.limit is not None:
        dataset = dataset.select(range(min(cfg.limit, len(dataset))))
    total = len(dataset)
    print(f"  Samples: {total}")

    results: List[Dict[str, Any]] = []

    async def runner():
        import httpx  # local import to avoid dependency if vLLM isn't used

        headers = {"Authorization": f"Bearer {api_key}"}
        limits = httpx.Limits(max_connections=cfg.vllm_max_concurrency, max_keepalive_connections=cfg.vllm_max_concurrency)
        async with httpx.AsyncClient(base_url=api_base, headers=headers, limits=limits) as client:
            semaphore = asyncio.Semaphore(cfg.vllm_max_concurrency)
            tasks = [
                _vllm_eval_one(client, semaphore, cfg, idx, item, model_name)
                for idx, item in enumerate(dataset)
            ]
            for coro in asyncio.as_completed(tasks):
                rec = await coro
                results.append(rec)

    asyncio.run(runner())

    # compute metrics
    correct = sum(1 for r in results if r.get("correct"))
    accuracy = correct / total if total else 0.0
    summary = {
        "model": cfg.model_name,
        "dataset": cfg.dataset_name,
        "split": cfg.dataset_split,
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": float(accuracy),
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "backend": cfg.backend,
            "vllm_api_base": api_base,
            "vllm_model": model_name,
            "max_new_tokens": cfg.generation_max_new_tokens,
            "temperature": cfg.generation_temperature,
            "limit": cfg.limit,
        },
    }

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    for key, value in summary.items():
        if key != "config":
            print(f"{key.title().replace('_', ' ')}: {value}")
    print("=" * 80)

    if cfg.save_predictions:
        os.makedirs(cfg.output_dir, exist_ok=True)
        results_path = os.path.join(cfg.output_dir, "predictions.csv")
        summary_path = os.path.join(cfg.output_dir, "summary.json")
        pd.DataFrame(results).to_csv(results_path, index=False)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved predictions to: {results_path}")
        print(f"Saved summary to: {summary_path}")

    return summary


def evaluate(cfg: EvalConfig) -> Dict[str, Any]:
    # Environment overrides (prefer env if set)
    backend_env = (os.environ.get("KOA_EVAL_BACKEND") or os.environ.get("KOA_EVALUATOR_BACKEND") or "").strip().lower()
    if backend_env:
        cfg.backend = backend_env
    if os.environ.get("VLLM_API_BASE"):
        cfg.vllm_api_base = os.environ.get("VLLM_API_BASE")
        if cfg.backend != "vllm":
            cfg.backend = "vllm"
    if os.environ.get("VLLM_API_KEY"):
        cfg.vllm_api_key = os.environ.get("VLLM_API_KEY")
    if os.environ.get("VLLM_MODEL"):
        cfg.vllm_model = os.environ.get("VLLM_MODEL")
    if os.environ.get("VLLM_MAX_CONCURRENCY"):
        try:
            cfg.vllm_max_concurrency = int(os.environ.get("VLLM_MAX_CONCURRENCY"))
        except Exception:
            pass
    if os.environ.get("VLLM_REQUEST_TIMEOUT_S"):
        try:
            cfg.vllm_request_timeout_s = float(os.environ.get("VLLM_REQUEST_TIMEOUT_S"))
        except Exception:
            pass
    if os.environ.get("VLLM_MAX_RETRIES"):
        try:
            cfg.vllm_max_retries = int(os.environ.get("VLLM_MAX_RETRIES"))
        except Exception:
            pass

    if cfg.backend == "vllm":
        return evaluate_vllm(cfg)
    return evaluate_hf(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL models.")
    parser.add_argument("-c", "--config", required=True, help="Path to config YAML file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
