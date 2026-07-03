"""
Numeric accuracy for math-style inference prediction JSONL files.

The input files are produced by inference_quality_eval.py and contain rows with
reference fields such as answer/target/reference plus a prediction field.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


REF_KEYS = ("answer", "output", "response", "target", "reference", "label")


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


def close_enough(pred: float | None, ref: float | None, abs_tol: float, rel_tol: float) -> bool:
    if pred is None or ref is None:
        return False
    return math.isclose(pred, ref, abs_tol=abs_tol, rel_tol=rel_tol)


def evaluate_file(path: Path, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    rows = []
    total = 0
    correct = 0
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
            pred_num = extract_number(pred_text)
            ref_num = extract_number(ref_text)
            ok = close_enough(pred_num, ref_num, abs_tol=abs_tol, rel_tol=rel_tol)

            total += 1
            correct += int(ok)
            pred_missing += int(pred_num is None)
            ref_missing += int(ref_num is None)

            out_row = dict(row)
            out_row.update(
                {
                    "idx": idx,
                    "prediction_number": pred_num,
                    "reference_number": ref_num,
                    "numeric_correct": ok,
                }
            )
            rows.append(out_row)
            if not ok:
                errors.append(out_row)

    metrics = {
        "file": str(path),
        "num_samples": total,
        "numeric_correct": correct,
        "numeric_accuracy": correct / max(total, 1),
        "prediction_number_missing": pred_missing,
        "reference_number_missing": ref_missing,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
    }

    metrics_path = path.with_suffix(".numeric_metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    scored_path = path.with_suffix(".numeric_scored.jsonl")
    with scored_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    errors_path = path.with_suffix(".numeric_errors.jsonl")
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
        "numeric_correct",
        "numeric_accuracy",
        "prediction_number_missing",
        "reference_number_missing",
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
                f"numeric_accuracy_delta={row['numeric_accuracy'] - base['numeric_accuracy']:+.4f} | "
                f"numeric_correct_delta={row['numeric_correct'] - base['numeric_correct']:+d}"
            )


if __name__ == "__main__":
    main()
