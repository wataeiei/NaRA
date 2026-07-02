"""
Compare inference metrics JSON files produced by inference_quality_eval.py.

Example:
    python nara_freeze_mid_validation/compare_inference_metrics.py \
      --metrics logs/pred_full.metrics.json logs/pred_freeze_8_23.metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", nargs="+", required=True)
    args = parser.parse_args()

    rows = []
    keys = set()
    for path_str in args.metrics:
        path = Path(path_str)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["file"] = str(path)
        rows.append(data)
        keys.update(data.keys())

    ordered = ["file", "bertscore_f1", "bertscore_precision", "bertscore_recall", "exact_match", "substring_match", "pred_contains_ref"]
    columns = ordered + sorted(k for k in keys if k not in set(ordered))
    print("\t".join(columns))
    for row in rows:
        print("\t".join(fmt(row.get(col)) for col in columns))

    if len(rows) >= 2:
        base = rows[0]
        print()
        print("Relative to first metrics file:")
        for row in rows[1:]:
            name = Path(row["file"]).name
            parts = [name]
            for key in columns:
                if key == "file":
                    continue
                if isinstance(base.get(key), (int, float)) and isinstance(row.get(key), (int, float)):
                    parts.append(f"{key}_delta={row[key] - base[key]:+.4f}")
            print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
