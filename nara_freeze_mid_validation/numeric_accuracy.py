"""
Mixed math accuracy for math-style inference prediction JSONL files.

The input files are produced by inference_quality_eval.py and contain rows with
reference fields such as answer/target/reference plus a prediction field.
It supports numeric answers and multiple-choice answers A/B/C/D/E.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


REF_KEYS = ("answer", "output", "response", "target", "reference", "label")
CHOICE_RE = re.compile(r"\b([A-E])\b")


def pick_ref(row: dict[str, Any]) -> str:
    for key in REF_KEYS:
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"Could not find reference key in row keys={list(row.keys())}")


def parse_numeric_token(token: str) -> float | None:
    token = token.strip().replace(",", "")
    if not token:
        return None

    percent = token.endswith("%")
    if percent:
        token = token[:-1].strip()

    if "/" in token:
        parts = token.split("/")
        if len(parts) == 2:
            try:
                denom = float(parts[1])
                if denom == 0:
                    return None
                value = float(parts[0]) / denom
            except ValueError:
                return None
        else:
            return None
    else:
        try:
            value = float(token)
        except ValueError:
            return None

    return value / 100.0 if percent else value


def extract_number(text: str) -> float | None:
    if text is None:
        return None
    text = str(text)

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    search_text = boxed[-1] if boxed else text

    # Prefer the last numeric expression, matching eval/eval.py behavior.
    pattern = r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*-?\d[\d,]*(?:\.\d+)?)?%?"
    matches = re.findall(pattern, search_text)
    for token in reversed(matches):
        value = parse_numeric_token(re.sub(r"\s+", "", token))
        if value is not None and math.isfinite(value):
            return value
    return None


def extract_choice(text: str) -> str | None:
    if text is None:
        return None
    text = str(text)
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    search_text = boxed[-1] if boxed else text

    # Prefer explicit answer phrases, then fall back to the last standalone option.
    explicit_patterns = [
        r"(?:the\s+answer\s+is|answer\s*[:：]|option\s*[:：])\s*\(?\s*([A-E])\s*\)?",
        r"therefore\s*,?\s*\(?\s*([A-E])\s*\)?",
        r"conclusively\s*[:：]?\s*\(?\s*([A-E])\s*\)?",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, search_text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    matches = CHOICE_RE.findall(search_text)
    return matches[-1].upper() if matches else None


def reference_kind(ref_text: str) -> str:
    stripped = str(ref_text).strip()
    if re.fullmatch(r"[A-Ea-e]", stripped):
        return "choice"
    if extract_number(stripped) is not None:
        return "numeric"
    return "unsupported"


def close_enough(pred: float | None, ref: float | None, abs_tol: float, rel_tol: float) -> bool:
    if pred is None or ref is None:
        return False
    return math.isclose(pred, ref, abs_tol=abs_tol, rel_tol=rel_tol)


def evaluate_file(path: Path, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    rows = []
    total = 0
    math_correct = 0
    numeric_total = 0
    numeric_correct = 0
    choice_total = 0
    choice_correct = 0
    unsupported_total = 0
    pred_missing = 0
    ref_missing = 0
    errors = []

    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pred_text = str(row.get("prediction", ""))
            ref_text = pick_ref(row)
            kind = reference_kind(ref_text)
            pred_num = extract_number(pred_text)
            ref_num = extract_number(ref_text)
            pred_choice = extract_choice(pred_text)
            ref_choice = extract_choice(ref_text)

            if kind == "choice":
                ok = pred_choice == ref_choice and ref_choice is not None
                choice_total += 1
                choice_correct += int(ok)
                pred_missing += int(pred_choice is None)
                ref_missing += int(ref_choice is None)
            elif kind == "numeric":
                ok = close_enough(pred_num, ref_num, abs_tol=abs_tol, rel_tol=rel_tol)
                numeric_total += 1
                numeric_correct += int(ok)
                pred_missing += int(pred_num is None)
                ref_missing += int(ref_num is None)
            else:
                ok = False
                unsupported_total += 1
                pred_missing += 1
                ref_missing += 1

            total += 1
            math_correct += int(ok)

            out_row = dict(row)
            out_row.update(
                {
                    "idx": idx,
                    "answer_kind": kind,
                    "prediction_number": pred_num,
                    "reference_number": ref_num,
                    "prediction_choice": pred_choice,
                    "reference_choice": ref_choice,
                    "numeric_correct": ok if kind == "numeric" else None,
                    "choice_correct": ok if kind == "choice" else None,
                    "math_correct": ok,
                }
            )
            rows.append(out_row)
            if not ok:
                errors.append(out_row)

    metrics = {
        "file": str(path),
        "num_samples": total,
        "math_correct": math_correct,
        "math_accuracy": math_correct / max(total, 1),
        "numeric_total": numeric_total,
        "numeric_correct": numeric_correct,
        "numeric_accuracy": numeric_correct / max(numeric_total, 1),
        "choice_total": choice_total,
        "choice_correct": choice_correct,
        "choice_accuracy": choice_correct / max(choice_total, 1),
        "unsupported_total": unsupported_total,
        "prediction_answer_missing": pred_missing,
        "reference_answer_missing": ref_missing,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
    }

    metrics_path = path.with_suffix(".math_metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    scored_path = path.with_suffix(".math_scored.jsonl")
    with scored_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    errors_path = path.with_suffix(".math_errors.jsonl")
    with errors_path.open("w", encoding="utf-8") as f:
        for row in errors:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics["metrics_path"] = str(metrics_path)
    metrics["scored_path"] = str(scored_path)
    metrics["errors_path"] = str(errors_path)
    return metrics


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", nargs="+", required=True)
    parser.add_argument("--abs-tol", type=float, default=1e-6)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    args = parser.parse_args()

    rows = [evaluate_file(Path(p), args.abs_tol, args.rel_tol) for p in args.preds]
    columns = [
        "file",
        "num_samples",
        "math_correct",
        "math_accuracy",
        "numeric_total",
        "numeric_correct",
        "numeric_accuracy",
        "choice_total",
        "choice_correct",
        "choice_accuracy",
        "unsupported_total",
        "prediction_answer_missing",
        "reference_answer_missing",
        "metrics_path",
        "errors_path",
    ]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(fmt(row.get(col)) for col in columns))

    if len(rows) >= 2:
        base = rows[0]
        print()
        print("Relative to first prediction file:")
        for row in rows[1:]:
            print(
                f"  {Path(row['file']).name} | "
                f"math_accuracy_delta={row['math_accuracy'] - base['math_accuracy']:+.4f} | "
                f"math_correct_delta={row['math_correct'] - base['math_correct']:+d} | "
                f"numeric_accuracy_delta={row['numeric_accuracy'] - base['numeric_accuracy']:+.4f} | "
                f"choice_accuracy_delta={row['choice_accuracy'] - base['choice_accuracy']:+.4f}"
            )


if __name__ == "__main__":
    main()
