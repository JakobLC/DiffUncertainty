#!/usr/bin/env python3
"""Scan saved experiment `hparams.yaml` files and replace HPC import paths with local ones.

Find files matching `values/saves/*/*/hparams.yaml` and if they contain
- data_input_dir: /work3/jloch/DiffUncertainty/values_datasets/<...>
- save_dir: /work3/jloch/DiffUncertainty/saves

then replace them with
- data_input_dir: /home/jloch/Desktop/diff/luzern/values_datasets/${dataset}
- save_dir: /home/jloch/Desktop/diff/luzern/values/saves

The script lists the files that would be changed (relative to `values/saves/`),
prompts for confirmation, and applies the modifications if the user confirms.
"""

import argparse
import fnmatch
import re
import sys
from pathlib import Path
try:
    import torch
except Exception:
    torch = None
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]  # project root (two levels up from values/global_utils)
SAVES_DIR = ROOT / "values" / "saves"

HPC_DATASETS_PREFIX = "/work3/jloch/DiffUncertainty/values_datasets/"
HPC_SAVES_PREFIX = "/work3/jloch/DiffUncertainty/saves"

LOCAL_DATA_INPUT = "/home/jloch/Desktop/diff/luzern/values_datasets/${data.name}"
LOCAL_SAVE_DIR = "/home/jloch/Desktop/diff/luzern/values/saves"

# Patterns to detect YAML lines like: data_input_dir: <value>  (with optional quotes)
LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[^:]+):\s*(?P<val>.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force_override",
        type=str,
        default=None,
        help=(
            "Glob pattern relative to values/saves used to force processing for matching experiments. "
            "Example: --force_override 'ood_aug/ens*'"
        ),
    )
    return parser.parse_args()


def find_hparams_files() -> list[Path]:
    # match files at values/saves/*/*/hparams.yaml
    pattern = SAVES_DIR.glob("*/*/hparams.yaml")
    return [p for p in pattern if p.is_file()]


def normalize_val(val: str) -> str:
    # strip quotes and whitespace
    val = val.strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


def preview_changes(filepath: Path, force: bool = False):
    """Return modified text (or None if no changes required).

    When ``force`` is True, replace recognized keys regardless of their current value.
    """
    text = filepath.read_text(encoding="utf-8")
    changed = False
    out_lines = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            out_lines.append(line)
            continue
        key = m.group('key').strip()
        val = normalize_val(m.group('val'))
        indent = m.group('indent')
        replace_data_dir = force or val.startswith(HPC_DATASETS_PREFIX)
        replace_save_dir = force or val.startswith(HPC_SAVES_PREFIX)
        if key == 'data_input_dir' and replace_data_dir:
            out_lines.append(f"{indent}{key}: {LOCAL_DATA_INPUT}")
            changed = True
        elif key == 'save_dir' and replace_save_dir:
            out_lines.append(f"{indent}{key}: {LOCAL_SAVE_DIR}")
            changed = True
        else:
            out_lines.append(line)
    if changed:
        return "\n".join(out_lines) + "\n"
    return None


def main():
    args = parse_args()
    force_pattern = args.force_override
    files = find_hparams_files()
    candidates = []
    for f in files:
        try:
            rel_path = f.relative_to(SAVES_DIR)
        except Exception:
            rel_path = f
        rel_posix = rel_path.as_posix()
        matches_force = bool(force_pattern and fnmatch.fnmatch(rel_posix, force_pattern))
        new_text = preview_changes(f, force=matches_force)
        if new_text is not None or matches_force:
            candidates.append((f, new_text, matches_force))

    if not candidates:
        print("No hparams.yaml files found that need updating.")
        return 0

    print("The following files would be processed (relative to values/saves/):")
    for f, new_text, forced in candidates:
        rel = f.relative_to(SAVES_DIR)
        suffix = " (forced)" if forced else ""
        print(f" - {rel.as_posix()}{suffix}")

    ans = input("Proceed and modify these files? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        print("Aborted — no files changed.")
        return 0

    # Apply changes
    for f, new_text, forced in candidates:
        rel = f.relative_to(SAVES_DIR)
        if new_text is not None:
            try:
                f.write_text(new_text, encoding='utf-8')
                print(f"Updated {rel}")
            except Exception as exc:
                print(f"Failed to update {f}: {exc}")
                continue
        else:
            print(f"No textual changes applied to {rel} (processed due to {'force override' if forced else 'preview detection'}).")

        # Now update any checkpoints under the same directory (only if hparams required replacement)
        parent = f.parent
        ckpt_files = list(parent.rglob("*.ckpt"))
        if not ckpt_files:
            continue
        if torch is None:
            print("torch not available: skipping checkpoint modifications for", f.relative_to(SAVES_DIR))
            continue
        if tqdm is not None:
            iter_ckpts = tqdm(ckpt_files, desc=f"Processing ckpts in {parent.name}", unit="ckpt")
        else:
            iter_ckpts = ckpt_files
        for ckpt_path in iter_ckpts:
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            except Exception as exc:
                print(f"Failed to load checkpoint {ckpt_path}: {exc}")
                continue
            modified = False
            if 'hyper_parameters' in ckpt:
                hp_obj = ckpt['hyper_parameters']
                # try to get a dict view
                hp = None
                if isinstance(hp_obj, dict):
                    hp = hp_obj
                else:
                    try:
                        hp = dict(hp_obj)
                    except Exception:
                        hp = None
                if hp is not None:
                    if 'data_input_dir' in hp and isinstance(hp['data_input_dir'], str):
                        if forced or hp['data_input_dir'].startswith(HPC_DATASETS_PREFIX):
                            hp['data_input_dir'] = LOCAL_DATA_INPUT
                            modified = True
                    data_cfg = hp.get('data') if isinstance(hp.get('data'), dict) else None
                    if data_cfg is not None:
                        data_dir_val = data_cfg.get('data_input_dir')
                        if isinstance(data_dir_val, str):
                            if forced or data_dir_val.startswith(HPC_DATASETS_PREFIX):
                                data_cfg['data_input_dir'] = LOCAL_DATA_INPUT
                                modified = True
                    if 'save_dir' in hp and isinstance(hp['save_dir'], str):
                        if forced or hp['save_dir'].startswith(HPC_SAVES_PREFIX):
                            hp['save_dir'] = LOCAL_SAVE_DIR
                            modified = True
                    if modified:
                        # assign back (ensure we preserve type when possible)
                        try:
                            if isinstance(hp_obj, dict):
                                ckpt['hyper_parameters'] = hp
                            else:
                                ckpt['hyper_parameters'] = hp
                        except Exception:
                            ckpt['hyper_parameters'] = hp
            if modified:
                try:
                    torch.save(ckpt, ckpt_path)
                    print(f"Updated checkpoint {ckpt_path.relative_to(SAVES_DIR)}")
                except Exception as exc:
                    print(f"Failed to save modified checkpoint {ckpt_path}: {exc}")
    print("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
