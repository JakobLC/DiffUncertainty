from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch


def _resolve_saves_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "saves").resolve()


def _collect_targets(base: Path) -> tuple[List[Path], List[Path]]:
    yaml_paths: List[Path] = []
    ckpt_paths: List[Path] = []
    for path in base.rglob("*"):
        if path.is_file():
            if path.name == "hparams.yaml":
                yaml_paths.append(path)
            elif path.suffix == ".ckpt":
                ckpt_paths.append(path)
    return yaml_paths, ckpt_paths


def _print_dry_run(old_path: Path, new_path: Path, yaml_files: Iterable[Path], ckpt_files: Iterable[Path]) -> None:
    print(str(old_path.resolve()))
    print(str(new_path.resolve()))
    print("yaml files:")
    for path in sorted(yaml_files):
        print(str(path.resolve()))
    print("ckpt files:")
    for path in sorted(ckpt_files):
        print(str(path.resolve()))


def _update_yaml(path: Path, new_exp_name: str) -> bool:
    text = path.read_text()
    lines = text.splitlines()
    replaced = False
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("exp_name:"):
            prefix_len = len(line) - len(stripped)
            lines[idx] = f"{line[:prefix_len]}exp_name: {new_exp_name}"
            replaced = True
            break
    if not replaced:
        return False
    ending = "\n" if text.endswith("\n") else ""
    path.write_text("\n".join(lines) + ending)
    return True


def _update_checkpoint(path: Path, new_exp_name: str) -> bool:
    data = torch.load(path, map_location="cpu", weights_only=False)
    updated = False
    hyper_params = data.get("hyper_parameters")
    if isinstance(hyper_params, dict):
        if hyper_params.get("exp_name") != new_exp_name:
            hyper_params["exp_name"] = new_exp_name
            updated = True
    if updated:
        torch.save(data, path)
    return updated


def rename_experiment(old_name: str, new_name: str, saves_dir: Path, dry_run: bool) -> None:
    if old_name == new_name:
        raise ValueError("old and new experiment names are identical")
    old_path = saves_dir / old_name
    new_path = saves_dir / new_name
    if not old_path.exists():
        raise FileNotFoundError(f"{old_path} does not exist")
    if new_path.exists():
        raise FileExistsError(f"{new_path} already exists")
    yaml_files, ckpt_files = _collect_targets(old_path)
    if dry_run:
        _print_dry_run(old_path, new_path, yaml_files, ckpt_files)
        return
    old_path.rename(new_path)
    rel_yaml = [path.relative_to(old_path) for path in yaml_files]
    rel_ckpt = [path.relative_to(old_path) for path in ckpt_files]
    updated_yaml = 0
    updated_ckpt = 0
    for rel in rel_yaml:
        target = new_path / rel
        if target.exists() and _update_yaml(target, new_name):
            updated_yaml += 1
    for rel in rel_ckpt:
        target = new_path / rel
        if target.exists() and _update_checkpoint(target, new_name):
            updated_ckpt += 1
    print(f"Renamed {old_path} -> {new_path}")
    print(f"Updated {updated_yaml} yaml files and {updated_ckpt} checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename a saved experiment folder and its metadata")
    parser.add_argument("--old", required=True, dest="old_name", help="Existing experiment folder name")
    parser.add_argument("--new", required=True, dest="new_name", help="Desired experiment folder name")
    parser.add_argument("--saves-dir", default=None, help="Override the default values/saves directory")
    parser.add_argument("--dry", action="store_true", help="Preview the rename without applying changes")
    args = parser.parse_args()
    saves_dir = _resolve_saves_dir(args.saves_dir)
    rename_experiment(args.old_name, args.new_name, saves_dir, args.dry)


if __name__ == "__main__":
    main()
