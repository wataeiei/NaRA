#!/usr/bin/env python
"""Build reproducible train/val/test splits with question-level deduplication.

Example:
  python nara_freeze_mid_validation/build_dataset_splits.py \
    --source data/llm_adapt/math/math_14k.json \
    --out-dir data/splits/math14k_seed42 \
    --dataset-name math14k \
    --val-size 1000 \
    --test-size 1000 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from pathlib import Path
from typing import Any


QUESTION_KEYS = ("instruction", "question", "prompt", "input", "query")
ANSWER_KEYS = ("output", "answer", "response", "target", "reference")
REFERENCE_KEYS = ("answer", "target", "reference", "ground_truth", "gt")


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "train", "examples", "records"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError(f"Unsupported dataset format: {path}")


def pick(row: dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default


def normalize_question(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def standardize_row(row: dict[str, Any], source_idx: int) -> dict[str, Any]:
    question = pick(row, QUESTION_KEYS)
    output = pick(row, ("output", "response", "answer", "target", "reference"))
    reference = pick(row, REFERENCE_KEYS, output)
    if not question or not output:
        raise ValueError(f"Missing question/output at source_idx={source_idx}")

    new_row = dict(row)
    new_row["instruction"] = question
    new_row["output"] = output
    new_row["answer"] = reference
    new_row["question"] = question
    new_row["source_idx"] = source_idx
    new_row["split_key"] = normalize_question(question)
    return new_row


def dedup_by_question(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        key = row["split_key"]
        if key in seen:
            duplicates.append(
                {
                    "source_idx": row["source_idx"],
                    "duplicate_of_source_idx": seen[key],
                    "question": row["question"],
                    "split_key": key,
                }
            )
            continue
        seen[key] = row["source_idx"]
        kept.append(row)
    return kept, duplicates


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if k != "split_key"}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def overlap_count(*splits: list[dict[str, Any]]) -> dict[str, int]:
    names = ["train", "val", "test"][: len(splits)]
    keys = [set(row["split_key"] for row in split) for split in splits]
    report: dict[str, int] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            report[f"{names[i]}_vs_{names[j]}"] = len(keys[i] & keys[j])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/llm_adapt/math/math_14k.json")
    parser.add_argument("--out-dir", default="data/splits/math14k_seed42")
    parser.add_argument("--dataset-name", default="math14k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--train-size", type=int, default=None)
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    raw_rows = read_records(source)
    standardized = [standardize_row(row, idx) for idx, row in enumerate(raw_rows)]
    deduped, duplicates = dedup_by_question(standardized)

    rng = random.Random(args.seed)
    shuffled = list(deduped)
    rng.shuffle(shuffled)

    if args.val_size + args.test_size >= len(shuffled):
        raise ValueError(
            f"val_size + test_size must be smaller than deduped dataset size. "
            f"Got val={args.val_size}, test={args.test_size}, deduped={len(shuffled)}"
        )

    val_rows = shuffled[: args.val_size]
    test_rows = shuffled[args.val_size : args.val_size + args.test_size]
    train_rows = shuffled[args.val_size + args.test_size :]
    if args.train_size is not None:
        train_rows = train_rows[: args.train_size]

    train_path = out_dir / f"{args.dataset_name}_train.jsonl"
    val_path = out_dir / f"{args.dataset_name}_val_{len(val_rows)}.jsonl"
    test_path = out_dir / f"{args.dataset_name}_test_{len(test_rows)}.jsonl"
    dup_path = out_dir / f"{args.dataset_name}_duplicates.jsonl"
    info_path = out_dir / "SPLIT_INFO.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)
    write_jsonl(dup_path, duplicates)

    overlaps = overlap_count(train_rows, val_rows, test_rows)
    info = {
        "dataset_name": args.dataset_name,
        "source": str(source),
        "seed": args.seed,
        "dedup": "normalized question: lowercase + punctuation removal + whitespace collapse",
        "raw_count": len(raw_rows),
        "deduped_count": len(deduped),
        "duplicate_count": len(duplicates),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "test_count": len(test_rows),
        "overlap_counts": overlaps,
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "duplicates": str(dup_path),
        },
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(info, ensure_ascii=False, indent=2))
    if any(overlaps.values()):
        raise SystemExit(f"Overlap check failed: {overlaps}")


if __name__ == "__main__":
    main()
