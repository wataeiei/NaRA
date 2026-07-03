"""
Inference quality evaluation for NaRA/full vs NaRA-freeze-mid checkpoints.

Run from the NaRA repository root.

Example for full:
    WANDB_MODE=disabled python nara_freeze_mid_validation/inference_quality_eval.py \
      --config config/nara/qcheck_full.yaml \
      --checkpoint outputs/qcheck_full \
      --data data/test.jsonl \
      --out logs/pred_full.jsonl \
      --max-samples 200

Example for freeze-mid:
    NARA_SKIP_LAYERS=8-23 WANDB_MODE=disabled python nara_freeze_mid_validation/inference_quality_eval.py \
      --config config/nara/qcheck_freeze_8_23.yaml \
      --checkpoint outputs/qcheck_freeze_8_23 \
      --data data/test.jsonl \
      --out logs/pred_freeze_8_23.jsonl \
      --max-samples 200

Input JSONL should contain one prompt field and one reference field. The script
tries common names automatically:
    prompt/question/instruction/input/query
    answer/output/response/target/reference

The script is intentionally defensive because NaRA/LLaDA repos often have custom
model wrappers. It tries:
    1. get_model_by_config(config)
    2. model.load_lora_only(checkpoint)
    3. model.load_state_dict(checkpoint)
    4. model.generate(...) or model.model.generate(...)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PROMPT_KEYS = ("prompt", "question", "instruction", "input", "query")
REF_KEYS = ("answer", "output", "response", "target", "reference", "label")


def gpu_metrics() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "gpu_mem_allocated_gb": 0.0,
            "gpu_mem_reserved_gb": 0.0,
            "gpu_peak_allocated_gb": 0.0,
            "gpu_peak_reserved_gb": 0.0,
        }
    return {
        "gpu_mem_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 4),
        "gpu_mem_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 4),
        "gpu_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 4),
        "gpu_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 4),
    }


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def read_config(path: Path) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.load(path)


def read_jsonl(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def pick_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"Could not find any of keys {keys} in row keys={list(row.keys())}")


def format_prompt(text: str) -> str:
    # Keep this plain. If your training used a special chat template, pass data
    # with the final prompt already formatted.
    return text


def unwrap_model(model):
    return getattr(model, "module", model)


def find_checkpoint_file(path: Path) -> Path:
    if not path.exists() and path.name == "latest_final":
        marker = path.parent / "LATEST_FINAL.txt"
        if marker.exists():
            target = marker.read_text(encoding="utf-8").strip()
            if target:
                path = path.parent / target

    if path.is_file():
        return path
    candidates = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
        "checkpoint.pt",
        "checkpoint.pth",
    ]
    for name in candidates:
        p = path / name
        if p.exists():
            return p

    marker = path / "LATEST_FINAL.txt"
    if marker.exists():
        target = marker.read_text(encoding="utf-8").strip()
        if target:
            try:
                return find_checkpoint_file(path / target)
            except FileNotFoundError:
                pass

    latest_link = path / "latest_final"
    if latest_link.exists():
        try:
            return find_checkpoint_file(latest_link)
        except FileNotFoundError:
            pass

    nested = []
    for pattern in ("**/adapter_model.safetensors", "**/adapter_model.bin", "**/pytorch_model.bin", "**/checkpoint.pt", "**/checkpoint.pth"):
        nested.extend(path.glob(pattern))
    if nested:
        priority = {"STOP_UPDATE": 0, "FINAL": 1, "BEST": 2, "UPDATE": 3}

        def rank(p: Path):
            parts = set(p.parts)
            folder_rank = min((v for k, v in priority.items() if any(part.startswith(k) for part in parts)), default=9)
            return (folder_rank, -p.stat().st_mtime)

        return sorted(nested, key=rank)[0]
    raise FileNotFoundError(f"No checkpoint file found under {path}")


def load_checkpoint_if_given(model, checkpoint: str | None, adapter_name: str) -> None:
    if not checkpoint:
        return
    ckpt_path = find_checkpoint_file(Path(checkpoint))
    model = unwrap_model(model)

    if hasattr(model, "load_lora_only"):
        print(f"[eval] loading LoRA/NARA weights via load_lora_only: {ckpt_path}")
        model.load_lora_only(str(ckpt_path), adapter_name=adapter_name)
        return

    print(f"[eval] loading state_dict: {ckpt_path}")
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(ckpt_path), device="cpu")
    else:
        state = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[eval] missing={len(missing)} unexpected={len(unexpected)}")


def set_nara_context(model, noise_level: float, noise_density: float | None) -> None:
    model = unwrap_model(model)
    if hasattr(model, "set_context_state"):
        model.set_context_state(noise_level=noise_level, noise_density=noise_density)
    elif hasattr(model, "set_noise_level"):
        model.set_noise_level(noise_level)


def get_generate_owner(model):
    model = unwrap_model(model)
    if hasattr(model, "generate"):
        return model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "generate"):
        return inner
    raise AttributeError("Neither model nor model.model has generate(). Use your repo's inference function here.")


def infer_finetuning_type(config):
    from config import FINETUNING_TYPE, get_type

    return get_type(FINETUNING_TYPE, config.get("finetuning_method", None))


def get_eval_defaults(config) -> dict[str, Any]:
    train_eval = config.get("train", {}).get("eval", {})
    return {
        "steps": int(train_eval.get("steps", 128)),
        "gen_length": int(train_eval.get("gen_length", 128)),
        "block_length": int(train_eval.get("block_length", 128)),
        "cfg_scale": float(train_eval.get("cfg_scale", 0.0)),
        "remasking": str(train_eval.get("remasking", "low_confidence")),
    }


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    *,
    backend: str,
    finetuning_type: Any,
    direct_noise: bool,
    steps: int,
    block_length: int,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
    random_noise: bool,
    till_eos: bool,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(unwrap_model(model).parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if backend in ("llada", "auto"):
        try:
            from eval.llada_generate import generate as llada_generate

            prompt_ids = inputs["input_ids"]
            gen_length = max_new_tokens
            if gen_length % block_length != 0:
                gen_length = ((gen_length + block_length - 1) // block_length) * block_length
            steps = max(1, steps)
            num_blocks = max(1, gen_length // block_length)
            if steps % num_blocks != 0:
                steps = ((steps + num_blocks - 1) // num_blocks) * num_blocks

            output_ids = llada_generate(
                model,
                tokenizer,
                finetuning_type,
                direct_noise,
                prompt_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
                is_main_process=True,
                random_noise=random_noise,
                till_eos=till_eos,
            )
            gen_ids = output_ids[0][prompt_ids.shape[1] :]
            return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        except Exception:
            if backend == "llada":
                raise

    owner = get_generate_owner(model)

    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "top_p": top_p,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    output_ids = owner.generate(**inputs, **kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[0][prompt_len:] if output_ids.shape[-1] > prompt_len else output_ids[0]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def simple_metrics(preds: list[str], refs: list[str]) -> dict[str, float]:
    exact = 0
    contains = 0
    pred_contains_ref = 0
    for pred, ref in zip(preds, refs):
        p = normalize_text(pred)
        r = normalize_text(ref)
        exact += int(p == r)
        contains += int(p in r or r in p)
        pred_contains_ref += int(r and r in p)
    n = max(1, len(preds))
    return {
        "exact_match": exact / n,
        "substring_match": contains / n,
        "pred_contains_ref": pred_contains_ref / n,
    }


def bertscore_metrics(preds: list[str], refs: list[str], lang: str) -> dict[str, float]:
    try:
        from bert_score import score
    except ImportError:
        print("[eval] bert_score is not installed; skipping BERTScore.")
        return {}
    p, r, f1 = score(preds, refs, lang=lang, verbose=False)
    return {
        "bertscore_precision": float(p.mean().item()),
        "bertscore_recall": float(r.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter-name", default="default")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--backend", choices=("auto", "llada", "hf"), default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--remasking", default=None)
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--direct-noise", action="store_true")
    parser.add_argument("--random-noise", action="store_true")
    parser.add_argument("--till-eos", action="store_true")
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--noise-density", type=float, default=None)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-lang", default="zh")
    parser.add_argument("--timing-out", default=None)
    args = parser.parse_args()

    from model.get_model import get_model_by_config

    total_start = time.perf_counter()
    config = read_config(Path(args.config))
    eval_defaults = get_eval_defaults(config)
    finetuning_type = infer_finetuning_type(config)
    model, tokenizer = get_model_by_config(config)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    load_checkpoint_if_given(model, args.checkpoint, args.adapter_name)
    set_nara_context(model, args.noise_level, args.noise_density)
    cuda_sync()
    model_ready_sec = time.perf_counter() - total_start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rows = read_jsonl(Path(args.data), args.max_samples)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    timing_path = Path(args.timing_out) if args.timing_out else Path(args.out).with_suffix(".timing.jsonl")
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    preds: list[str] = []
    refs: list[str] = []
    total_generation_sec = 0.0
    total_prompt_tokens = 0
    total_prediction_tokens = 0
    with Path(args.out).open("w", encoding="utf-8") as f, timing_path.open("w", encoding="utf-8") as tf:
        for idx, row in enumerate(rows):
            prompt = format_prompt(pick_text(row, PROMPT_KEYS))
            ref = pick_text(row, REF_KEYS)
            prompt_tokens = len(tokenizer(prompt)["input_ids"])
            cuda_sync()
            sample_start = time.perf_counter()
            pred = generate_one(
                model,
                tokenizer,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                backend=args.backend,
                finetuning_type=finetuning_type,
                direct_noise=args.direct_noise,
                steps=args.steps or eval_defaults["steps"],
                block_length=args.block_length or eval_defaults["block_length"],
                cfg_scale=args.cfg_scale if args.cfg_scale is not None else eval_defaults["cfg_scale"],
                remasking=args.remasking or eval_defaults["remasking"],
                mask_id=args.mask_id,
                random_noise=args.random_noise,
                till_eos=args.till_eos,
            )
            cuda_sync()
            generation_sec = time.perf_counter() - sample_start
            pred_tokens = len(tokenizer(pred)["input_ids"]) if pred else 0
            total_generation_sec += generation_sec
            total_prompt_tokens += prompt_tokens
            total_prediction_tokens += pred_tokens
            preds.append(pred)
            refs.append(ref)
            out_row = dict(row)
            out_row["prediction"] = pred
            out_row["generation_sec"] = generation_sec
            out_row["prompt_tokens"] = prompt_tokens
            out_row["prediction_tokens"] = pred_tokens
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            timing_row = {
                "idx": idx,
                "generation_sec": generation_sec,
                "prompt_tokens": prompt_tokens,
                "prediction_tokens": pred_tokens,
                "prediction_tokens_per_sec": pred_tokens / max(generation_sec, 1e-9),
                **gpu_metrics(),
            }
            tf.write(json.dumps(timing_row, ensure_ascii=False) + "\n")
            print(f"[{idx + 1}/{len(rows)}] pred={pred[:80]!r} ref={ref[:80]!r}")

    metrics = simple_metrics(preds, refs)
    bertscore_start = time.perf_counter()
    if args.bertscore:
        metrics.update(bertscore_metrics(preds, refs, args.bertscore_lang))
    bertscore_sec = time.perf_counter() - bertscore_start if args.bertscore else 0.0
    total_sec = time.perf_counter() - total_start
    metrics.update(
        {
            "num_samples": len(rows),
            "model_ready_sec": model_ready_sec,
            "total_wall_sec": total_sec,
            "total_generation_sec": total_generation_sec,
            "avg_generation_sec": total_generation_sec / max(len(rows), 1),
            "bertscore_sec": bertscore_sec,
            "total_prompt_tokens": total_prompt_tokens,
            "total_prediction_tokens": total_prediction_tokens,
            "prediction_tokens_per_sec": total_prediction_tokens / max(total_generation_sec, 1e-9),
            "samples_per_sec_generation": len(rows) / max(total_generation_sec, 1e-9),
            "config": args.config,
            "checkpoint": args.checkpoint,
            "backend": args.backend,
            **gpu_metrics(),
        }
    )

    metrics_path = Path(args.out).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[eval] predictions: {args.out}")
    print(f"[eval] timing: {timing_path}")
    print(f"[eval] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
