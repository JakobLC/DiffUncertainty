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

from pathlib import Path
import re
import sys
import shutil
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

LOCAL_DATA_INPUT = "/home/jloch/Desktop/diff/luzern/values_datasets/${dataset}"
LOCAL_SAVE_DIR = "/home/jloch/Desktop/diff/luzern/values/saves"

# Patterns to detect YAML lines like: data_input_dir: <value>  (with optional quotes)
LINE_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[^:]+):\s*(?P<val>.*)$")


def find_hparams_files():
    # match files at values/saves/*/*/hparams.yaml
    pattern = SAVES_DIR.glob("*/*/hparams.yaml")
    return [p for p in pattern if p.is_file()]


def normalize_val(val: str) -> str:
    # strip quotes and whitespace
    val = val.strip()
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


def preview_changes(filepath: Path):
    """Return modified text (or None if no changes required)"""
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
        if key == 'data_input_dir' and val.startswith(HPC_DATASETS_PREFIX):
            out_lines.append(f"{indent}{key}: {LOCAL_DATA_INPUT}")
            changed = True
        elif key == 'save_dir' and val.startswith(HPC_SAVES_PREFIX):
            out_lines.append(f"{indent}{key}: {LOCAL_SAVE_DIR}")
            changed = True
        else:
            out_lines.append(line)
    if changed:
        return "\n".join(out_lines) + "\n"
    return None


def main():
    files = find_hparams_files()
    candidates = []
    for f in files:
        new_text = preview_changes(f)
        if new_text is not None:
            candidates.append((f, new_text))

    if not candidates:
        print("No hparams.yaml files found that need updating.")
        return 0

    print("The following files would be modified (relative to values/saves/):")
    for f, _ in candidates:
        try:
            rel = f.relative_to(SAVES_DIR)
        except Exception:
            rel = f
        print(" -", rel.as_posix())

    ans = input("Proceed and modify these files? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        print("Aborted — no files changed.")
        return 0

    # Apply changes
    for f, new_text in candidates:
        backup = f.with_suffix('.yaml.bak')
        try:
            # write backup if not exists
            if not backup.exists():
                backup.write_text(f.read_text(encoding='utf-8'), encoding='utf-8')
            f.write_text(new_text, encoding='utf-8')
            print(f"Updated {f.relative_to(SAVES_DIR)}")
        except Exception as exc:
            print(f"Failed to update {f}: {exc}")
            continue

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
                ckpt = torch.load(ckpt_path, map_location='cpu')
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
                    if 'data_input_dir' in hp and isinstance(hp['data_input_dir'], str) and hp['data_input_dir'].startswith(HPC_DATASETS_PREFIX):
                        hp['data_input_dir'] = LOCAL_DATA_INPUT
                        modified = True
                    if 'save_dir' in hp and isinstance(hp['save_dir'], str) and hp['save_dir'].startswith(HPC_SAVES_PREFIX):
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
                # backup ckpt if not exists
                bak = ckpt_path.with_suffix(ckpt_path.suffix + '.bak')
                try:
                    if not bak.exists():
                        shutil.copy2(ckpt_path, bak)
                    torch.save(ckpt, ckpt_path)
                    print(f"Updated checkpoint {ckpt_path.relative_to(SAVES_DIR)}")
                except Exception as exc:
                    print(f"Failed to save modified checkpoint {ckpt_path}: {exc}")
    print("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
