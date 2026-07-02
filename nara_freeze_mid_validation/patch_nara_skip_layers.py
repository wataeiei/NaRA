"""
Patch NaRA so a config can skip adapter injection for selected transformer layers.

Run this from the NaRA repository root:

    python path/to/patch_nara_skip_layers.py

Then add this field to a NaRA yaml config:

    skip_layer_regex: "model\\.layers\\.(?:8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23)\\."

This is meant for a quick speed smoke test. It skips creating NARA modules in the
matched layers, which is more meaningful for speed than creating adapters and
only setting requires_grad=False.

If the NaRA training code does not pass skip_layer_regex from yaml into
NARAConfig, you can still enable skipping with:

    NARA_SKIP_LAYER_REGEX='(?:^|\\.)layers\\.(?:8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23)\\.' python train.py ...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def make_config_insert(indent: str) -> str:
    continuation = indent + "    "
    return (
        f'{indent}skip_layer_regex: Optional[str] = field(\n'
        f'{continuation}default=None,\n'
        f'{continuation}metadata={{"help": "Regex for module names that should NOT receive NARA adapters."}},\n'
        f'{indent})\n'
    )


def infer_child_indent(text: str, parent_indent: str) -> str:
    lines = text.splitlines()
    for line in lines:
        stripped = line.lstrip(" \t")
        if not stripped or stripped.startswith("#"):
            continue
        indent = line[: len(line) - len(stripped)]
        if indent.startswith(parent_indent) and len(indent) > len(parent_indent):
            return indent
    return parent_indent + "    "


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False

    if "import re" not in text:
        if "import math\n" in text:
            text = text.replace("import math\n", "import math\nimport re\n", 1)
        else:
            text = "import re\n" + text
        changed = True

    if "import os" not in text:
        if "import re\n" in text:
            text = text.replace("import re\n", "import re\nimport os\n", 1)
        elif "import math\n" in text:
            text = text.replace("import math\n", "import math\nimport os\n", 1)
        else:
            text = "import os\n" + text
        changed = True

    if "skip_layer_regex" not in text:
        config_pattern = re.compile(
            r"(?m)^(?P<indent>\s*)target_modules:\s*Optional\[Union\[List\[str\],\s*str\]\]\s*=\s*field\(default=None\).*$"
        )
        match = config_pattern.search(text)
        if not match:
            raise RuntimeError("Could not find NARAConfig target_modules anchor.")
        insert_at = match.end() + 1
        text = text[:insert_at] + make_config_insert(match.group("indent")) + text[insert_at:]
        changed = True

    if 'os.environ.get("NARA_SKIP_LAYER_REGEX")' not in text:
        text = text.replace(
            'getattr(self.peft_config[adapter_name], "skip_layer_regex", None)',
            'getattr(self.peft_config[adapter_name], "skip_layer_regex", None) or os.environ.get("NARA_SKIP_LAYER_REGEX")',
        )
        changed = True

    if 're.search(skip_layer_regex, key)' not in text:
        true_pattern = re.compile(r"(?m)^(?P<indent>\s*)is_target_modules_in_base_model\s*=\s*True\s*$")
        match = true_pattern.search(text)
        if not match:
            raise RuntimeError("Could not find _find_and_replace loop anchor.")

        indent = match.group("indent")
        child_indent = infer_child_indent(text, indent)
        skip_block = (
            f'{indent}skip_layer_regex = getattr(self.peft_config[adapter_name], "skip_layer_regex", None)\n'
            f'{indent}if skip_layer_regex and re.search(skip_layer_regex, key):\n'
            f'{child_indent}continue\n'
            "\n"
        )
        text = text[: match.start()] + skip_block + text[match.start() :]
        changed = True

    if changed:
        backup = path.with_suffix(path.suffix + ".before_skip_layer_patch")
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(text, encoding="utf-8")

    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nara-py",
        default="nara/tuners/nara.py",
        help="Path to NaRA's nara/tuners/nara.py",
    )
    args = parser.parse_args()

    path = Path(args.nara_py)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Run this from the NaRA repo root.")

    changed = patch_file(path)
    if changed:
        print(f"Patched {path}")
        print(f"Backup written beside it as {path.name}.before_skip_layer_patch")
    else:
        print(f"{path} already has skip_layer_regex support")


if __name__ == "__main__":
    main()
