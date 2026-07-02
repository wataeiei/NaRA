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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml


PROMPT_KEYS = ("prompt", "question", "instruction", "input", "query")
REF_KEYS = ("answer", "output", "response", "target", "reference", "label")


def to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(x) for x in obj]
    return obj


def read_config(path: Path) -> Any:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return to_namespace(raw)


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
    nested = sorted(path.glob("**/adapter_model.safetensors")) + sorted(path.glob("**/adapter_model.bin"))
    if nested:
        return nested[-1]
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


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    owner = get_generate_owner(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(owner.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

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
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--noise-density", type=float, default=None)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-lang", default="zh")
    args = parser.parse_args()

    from model.get_model import get_model_by_config

    config = read_config(Path(args.config))
    model, tokenizer = get_model_by_config(config)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    load_checkpoint_if_given(model, args.checkpoint, args.adapter_name)
    set_nara_context(model, args.noise_level, args.noise_density)

    rows = read_jsonl(Path(args.data), args.max_samples)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    preds: list[str] = []
    refs: list[str] = []
    with Path(args.out).open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            prompt = format_prompt(pick_text(row, PROMPT_KEYS))
            ref = pick_text(row, REF_KEYS)
            pred = generate_one(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p)
            preds.append(pred)
            refs.append(ref)
            out_row = dict(row)
            out_row["prediction"] = pred
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            print(f"[{idx + 1}/{len(rows)}] pred={pred[:80]!r} ref={ref[:80]!r}")

    metrics = simple_metrics(preds, refs)
    if args.bertscore:
        metrics.update(bertscore_metrics(preds, refs, args.bertscore_lang))

    metrics_path = Path(args.out).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[eval] predictions: {args.out}")
    print(f"[eval] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
