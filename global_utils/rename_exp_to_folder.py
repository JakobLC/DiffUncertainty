from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import torch


def _strip_quotes(value: str) -> str:
    if not value:
        return value
    value = value.split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _extract_top_level(lines: Sequence[str], key: str) -> str | None:
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(" "):
            continue
        stripped = line.strip()
        if stripped.startswith(prefix):
            raw_value = stripped[len(prefix):].strip()
            value = _strip_quotes(raw_value)
            if value.lower() == "null":
                return None
            return value
    return None


def _build_run_dir(global_save_dir: str | None, exp_name: str, version: str | None, yaml_path: Path) -> str:
    base_dir = global_save_dir or str(yaml_path.parent.parent.parent)
    version_name = version or yaml_path.parent.name
    base_dir = base_dir.rstrip("/\\")
    return os.path.join(base_dir, exp_name, version_name)


def _replace_line(lines: List[str], idx: int, key: str, new_value: str) -> None:
    line = lines[idx]
    stripped = line.lstrip()
    indent_len = len(line) - len(stripped)
    indent = line[:indent_len]
    lines[idx] = f"{indent}{key} {new_value}"


def _update_yaml(path: Path, new_exp_name: str, dry_run: bool) -> bool:
    text = path.read_text()
    lines = text.splitlines()
    save_dir_value = _extract_top_level(lines, "save_dir")
    version_value = _extract_top_level(lines, "version")
    run_dir = _build_run_dir(save_dir_value, new_exp_name, version_value, path)

    changed = False
    section_stack: List[Tuple[str, int]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "":
            continue
        indent = len(line) - len(line.lstrip(" "))
        while section_stack and indent <= section_stack[-1][1]:
            section_stack.pop()
        if stripped.endswith(":") and not stripped.startswith("- "):
            section_stack.append((stripped[:-1].strip(), indent))
            continue

        in_logger = any(name == "logger" for name, _ in section_stack)
        in_trainer = any(name == "trainer" for name, _ in section_stack)

        if stripped.startswith("exp_name:"):
            _replace_line(lines, idx, "exp_name:", new_exp_name)
            changed = True
        elif stripped.startswith("local_run_dir:"):
            _replace_line(lines, idx, "local_run_dir:", run_dir)
            changed = True
        elif stripped.startswith("default_root_dir:") and in_trainer:
            _replace_line(lines, idx, "default_root_dir:", run_dir)
            changed = True
        elif stripped.startswith("save_dir:") and in_logger:
            _replace_line(lines, idx, "save_dir:", run_dir)
            changed = True

    if not changed or dry_run:
        return changed

    ending = "\n" if text.endswith("\n") else ""
    path.write_text("\n".join(lines) + ending)
    return True


def _update_checkpoint(path: Path, new_exp_name: str, dry_run: bool) -> bool:
    data = torch.load(path, map_location="cpu", weights_only=False)
    hyper_params = data.get("hyper_parameters")
    if not isinstance(hyper_params, dict):
        return False
    if hyper_params.get("exp_name") == new_exp_name:
        return False
    if dry_run:
        return True
    hyper_params["exp_name"] = new_exp_name
    torch.save(data, path)
    return True


def _collect_files(run_root: Path) -> Tuple[List[Path], List[Path]]:
    yaml_files: List[Path] = []
    ckpt_files: List[Path] = []
    for file_path in run_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name == "hparams.yaml":
            yaml_files.append(file_path)
        elif file_path.suffix == ".ckpt":
            ckpt_files.append(file_path)
    return yaml_files, ckpt_files


def rename_exp_to_folder(folder: Path, dry_run: bool) -> None:
    folder = folder.resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")
    new_exp_name = folder.name
    yaml_files, ckpt_files = _collect_files(folder)
    if not yaml_files and not ckpt_files:
        print(f"No targets found under {folder}")
        return

    updated_yaml = 0
    updated_ckpt = 0
    for yaml_path in yaml_files:
        if _update_yaml(yaml_path, new_exp_name, dry_run):
            updated_yaml += 1
    for ckpt_path in ckpt_files:
        if _update_checkpoint(ckpt_path, new_exp_name, dry_run):
            updated_ckpt += 1

    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"[{mode}] Folder: {folder}")
    print(f"  YAML files updated: {updated_yaml}")
    print(f"  Checkpoints updated: {updated_ckpt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync experiment metadata with its enclosing folder name")
    parser.add_argument("--folder", required=True, help="Path to the experiment folder (e.g., values/saves/<exp_name>)")
    parser.add_argument("--dry", action="store_true", help="Show planned edits without writing changes")
    args = parser.parse_args()
    rename_exp_to_folder(Path(args.folder), args.dry)


if __name__ == "__main__":
    main()
