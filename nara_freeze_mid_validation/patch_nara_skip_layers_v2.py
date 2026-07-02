"""
Robust NaRA middle-layer skip patch.

Run from the NaRA repository root:

    python nara_freeze_mid_validation/patch_nara_skip_layers_v2.py
    python -m py_compile nara/tuners/nara.py

Then test with:

    NARA_SKIP_LAYERS=8-23 NARA_DEBUG_SKIP_LAYERS=1 WANDB_MODE=disabled \
    python train.py --config config/nara/smoke_nara_freeze_mid.yaml --seed 1234

This patch does not rely on yaml propagation. It can skip by layer index parsed
from target module names, and can also use regex as a fallback.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


HELPERS = r'''

def _nara_parse_skip_layers(spec: Optional[str]) -> set[int]:
    """Parse a spec like "8-23,30" into a set of layer indexes."""
    if not spec:
        return set()
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(part))
    return out


def _nara_layer_index_from_key(key: str) -> Optional[int]:
    """Extract a transformer layer index from common module-name styles."""
    patterns = [
        r"(?:^|\.)layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)layer\.(\d+)(?:\.|$)",
        r"(?:^|\.)blocks\.(\d+)(?:\.|$)",
        r"(?:^|\.)block\.(\d+)(?:\.|$)",
        r"(?:^|\.)h\.(\d+)(?:\.|$)",
        r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)encoder\.layers\.(\d+)(?:\.|$)",
    ]
    for pat in patterns:
        m = re.search(pat, key)
        if m:
            return int(m.group(1))
    return None


def _nara_should_skip_key(key: str, lcfg) -> bool:
    """Return True when this target module should not receive a NARA adapter."""
    regex = getattr(lcfg, "skip_layer_regex", None) or os.environ.get("NARA_SKIP_LAYER_REGEX")
    if regex and re.search(regex, key):
        return True

    skip_layers = _nara_parse_skip_layers(os.environ.get("NARA_SKIP_LAYERS"))
    if not skip_layers:
        skip_layers = _nara_parse_skip_layers(getattr(lcfg, "skip_layers", None))
    if skip_layers:
        layer_idx = _nara_layer_index_from_key(key)
        return layer_idx in skip_layers

    return False
'''


def ensure_import(text: str, module: str) -> str:
    if re.search(rf"(?m)^import {re.escape(module)}$", text):
        return text
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, f"import {module}\n")
    return "".join(lines)


def ensure_config_fields(text: str) -> str:
    if "skip_layers:" not in text:
        m = re.search(
            r"(?m)^(?P<indent>\s*)skip_layer_regex:\s*Optional\[str\]\s*=\s*field\(\n"
            r"(?P<body>.*?\n)"
            r"(?P=indent)\)\n",
            text,
            flags=re.S,
        )
        if m:
            indent = m.group("indent")
            insert = (
                f'{indent}skip_layers: Optional[str] = field(\n'
                f'{indent}    default=None,\n'
                f'{indent}    metadata={{"help": "Layer indexes/ranges to skip, e.g. 8-23,30."}},\n'
                f'{indent})\n'
            )
            text = text[: m.end()] + insert + text[m.end() :]
        else:
            m = re.search(r"(?m)^(?P<indent>\s*)target_modules:.*$", text)
            if not m:
                raise RuntimeError("Could not find NARAConfig target_modules field.")
            indent = m.group("indent")
            insert_at = m.end() + 1
            insert = (
                f'{indent}skip_layer_regex: Optional[str] = field(\n'
                f'{indent}    default=None,\n'
                f'{indent}    metadata={{"help": "Regex for module names that should NOT receive NARA adapters."}},\n'
                f'{indent})\n'
                f'{indent}skip_layers: Optional[str] = field(\n'
                f'{indent}    default=None,\n'
                f'{indent}    metadata={{"help": "Layer indexes/ranges to skip, e.g. 8-23,30."}},\n'
                f'{indent})\n'
            )
            text = text[:insert_at] + insert + text[insert_at:]
    return text


def ensure_helpers(text: str) -> str:
    if "def _nara_should_skip_key" in text:
        return text
    marker = "# ---------------------------------------------------------------------\n# Model wrapper"
    if marker not in text:
        raise RuntimeError("Could not find Model wrapper marker.")
    return text.replace(marker, HELPERS + "\n" + marker, 1)


def patch_find_and_replace(text: str) -> str:
    # Remove older skip blocks added by previous patch attempts.
    text = re.sub(
        r"\n\s*skip_layer_regex = getattr\(self\.peft_config\[adapter_name\], \"skip_layer_regex\", None\)"
        r"(?: or os\.environ\.get\(\"NARA_SKIP_LAYER_REGEX\"\))?\n"
        r"\s*\n?\s*if skip_layer_regex and re\.search\(skip_layer_regex, key\):\n"
        r"\s*\n?\s*continue\n",
        "\n",
        text,
    )

    target_block = (
        "            if not target_module_found:\n"
        "                continue\n"
    )
    if "_nara_should_skip_key(key, lcfg)" in text:
        return text
    if target_block not in text:
        raise RuntimeError("Could not find target_module_found continue block.")

    insert = (
        target_block
        + "\n"
        + "            if _nara_should_skip_key(key, lcfg):\n"
        + "                if os.environ.get(\"NARA_DEBUG_SKIP_LAYERS\"):\n"
        + "                    print(f\"[NARA skip] {key}\")\n"
        + "                continue\n"
        + "\n"
        + "            if os.environ.get(\"NARA_DEBUG_TARGETS\"):\n"
        + "                print(f\"[NARA target] {key}\")\n"
    )
    return text.replace(target_block, insert, 1)


def patch_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    text = original
    text = ensure_import(text, "re")
    text = ensure_import(text, "os")
    text = ensure_config_fields(text)
    text = ensure_helpers(text)
    text = patch_find_and_replace(text)

    if text == original:
        return False

    backup = path.with_suffix(path.suffix + ".before_skip_layer_v2_patch")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nara-py", default="nara/tuners/nara.py")
    args = parser.parse_args()
    path = Path(args.nara_py)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Run this from the NaRA repo root.")
    changed = patch_file(path)
    print(f"{'Patched' if changed else 'Already patched'} {path}")


if __name__ == "__main__":
    main()
