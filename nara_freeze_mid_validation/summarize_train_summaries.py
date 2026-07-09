"""
Print a compact TSV table from train_summary.json files.

Default:
    python nara_freeze_mid_validation/summarize_train_summaries.py

Custom runs:
    python nara_freeze_mid_validation/summarize_train_summaries.py \
      --item "Full=outputs/qcheck_full_200/metrics/train_summary.json" \
      --item "Freeze 4-27=outputs/qcheck_freeze_4_27_200/metrics/train_summary.json"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_ITEMS = [
    ("Full", "outputs/qcheck_full_200/metrics/train_summary.json"),
    ("Freeze 8-23", "outputs/qcheck_freeze_8_23_200/metrics/train_summary.json"),
    ("Freeze 4-27", "outputs/qcheck_freeze_4_27_200/metrics/train_summary.json"),
    ("Freeze 12-19", "outputs/qcheck_freeze_12_19_200/metrics/train_summary.json"),
]

DEFAULT_KEYS = [
    "trainable_params_M",
    "trainable_ratio_percent",
    "completed_updates",
    "train_time_min",
    "updates_per_min",
    "tokens_per_sec",
    "peak_vram_gb",
    "best_loss",
    "best_step",
]


def parse_item(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = value
        return Path(path).parent.parent.name, path
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", action="append", default=[], help="Format: name=path/to/train_summary.json")
    parser.add_argument("--key", action="append", default=[], help="Extra summary key to print, repeatable.")
    args = parser.parse_args()

    items = [parse_item(item) for item in args.item] if args.item else DEFAULT_ITEMS
    keys = DEFAULT_KEYS + [key for key in args.key if key not in DEFAULT_KEYS]

    print("method\t" + "\t".join(keys))
    for name, path in items:
        summary_path = Path(path)
        if not summary_path.exists():
            print(name + "\t" + "\t".join("-" for _ in keys))
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        print(name + "\t" + "\t".join(str(data.get(key, "-")) for key in keys))


if __name__ == "__main__":
    main()
