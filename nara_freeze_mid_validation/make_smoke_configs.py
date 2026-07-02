"""
Create two short-run configs for checking whether NARA middle-layer skipping speeds up training.

Run from the NaRA repository root after applying patch_nara_skip_layers.py:

    python path/to/make_smoke_configs.py \
      --base-config config/nara/llada_instruct_nara_math14k.yaml \
      --steps 200

It writes:
    config/nara/smoke_nara_full.yaml
    config/nara/smoke_nara_freeze_mid.yaml

Then run both configs with the normal NaRA training entry:
    python train.py --config config/nara/smoke_nara_full.yaml --seed 1234
    python train.py --config config/nara/smoke_nara_freeze_mid.yaml --seed 1234
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


MID_8_23_REGEX = r"model\.layers\.(?:8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23)\."


def set_if_present_or_add(cfg: dict, key: str, value) -> None:
    cfg[key] = value


def make_smoke_config(base: dict, steps: int, skip_regex: str | None) -> dict:
    cfg = copy.deepcopy(base)

    # Common names used by Trainer/Accelerate-style configs. Unknown keys are harmless
    # only if train.py ignores them; keep this file small and easy to adjust.
    for key in ("num_train_epochs", "epochs", "num_epochs"):
        if key in cfg:
            cfg[key] = 1

    for key in ("max_steps", "max_train_steps", "train_steps"):
        if key in cfg:
            cfg[key] = steps

    if not any(k in cfg for k in ("max_steps", "max_train_steps", "train_steps")):
        cfg["max_steps"] = steps

    for key in ("save_steps", "eval_steps", "logging_steps"):
        if key in cfg:
            cfg[key] = min(int(cfg[key]), max(10, steps // 5))

    if "output_dir" in cfg:
        suffix = "freeze_mid" if skip_regex else "full"
        cfg["output_dir"] = str(Path(cfg["output_dir"]) / f"smoke_{suffix}")

    if skip_regex:
        set_if_present_or_add(cfg, "skip_layer_regex", skip_regex)
    else:
        cfg.pop("skip_layer_regex", None)

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--skip-regex", default=MID_8_23_REGEX)
    parser.add_argument("--out-dir", default="config/nara")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError(f"{base_path} did not parse as a yaml mapping")

    full = make_smoke_config(base, args.steps, None)
    freeze_mid = make_smoke_config(base, args.steps, args.skip_regex)

    full_path = out_dir / "smoke_nara_full.yaml"
    freeze_path = out_dir / "smoke_nara_freeze_mid.yaml"
    full_path.write_text(yaml.safe_dump(full, sort_keys=False, allow_unicode=True), encoding="utf-8")
    freeze_path.write_text(yaml.safe_dump(freeze_mid, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print("Wrote configs:")
    print(f"  {full_path}")
    print(f"  {freeze_path}")
    print()
    print("Run:")
    print(f"  python train.py --config {full_path.as_posix()} --seed 1234")
    print(f"  python train.py --config {freeze_path.as_posix()} --seed 1234")
    print()
    print("Freeze-mid regex:")
    print(json.dumps(args.skip_regex))


if __name__ == "__main__":
    main()
