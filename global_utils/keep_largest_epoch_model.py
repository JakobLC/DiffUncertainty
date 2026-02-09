from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch


def _resolve_saves_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "saves").resolve()


def _ensure_within(parent: Path, target: Path) -> None:
    try:
        target.relative_to(parent)
    except ValueError as exc:
        raise ValueError(f"{target} is outside of {parent}") from exc


def _is_version_dir(path: Path) -> bool:
    return path.is_dir() and (path / "checkpoints").is_dir()


def _resolve_target_path(saves_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = saves_dir / path
    resolved = path.resolve()
    _ensure_within(saves_dir, resolved)
    return resolved


def _determine_scope(target_path: Path) -> tuple[Path, List[Path]]:
    if _is_version_dir(target_path):
        return target_path.parent, [target_path]
    if not target_path.is_dir():
        raise FileNotFoundError(f"{target_path} is not a directory")
    version_dirs = sorted(child for child in target_path.iterdir() if _is_version_dir(child))
    if not version_dirs:
        raise ValueError(f"No version folders with checkpoints found under {target_path}")
    return target_path, version_dirs


def _collect_candidates(ckpt_dir: Path) -> List[Path]:
    return sorted(path for path in ckpt_dir.glob("last*.ckpt") if path.is_file())


def _read_epoch(path: Path) -> int:
    data = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("epoch", "current_epoch"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    raise ValueError(f"Could not locate an epoch value inside {path}")


def _format_rel(path: Path, exp_dir: Path) -> str:
    return str(path.relative_to(exp_dir))


def _describe_epoch(epoch: int | None) -> str:
    return str(epoch) if epoch is not None else "n/a"


def _plan_version(version_dir: Path, dry: bool) -> tuple[List[Path], Dict[Path, int], Path, Path]:
    ckpt_dir = version_dir / "checkpoints"
    candidates = _collect_candidates(ckpt_dir)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files matching last*.ckpt found in {ckpt_dir}")
    epoch_map: Dict[Path, int] = {}
    if len(candidates) > 1:
        for path in candidates:
            epoch_map[path] = _read_epoch(path)
    elif dry:
        epoch_map[candidates[0]] = _read_epoch(candidates[0])
    if len(candidates) == 1:
        best_path = candidates[0]
    else:
        best_path = min(candidates, key=lambda path: (-epoch_map[path], path.name))
    final_path = ckpt_dir / "last.ckpt"
    losers = [path for path in candidates if path != best_path]
    return losers, epoch_map, best_path, final_path


def _log_version_actions(
    version_dir: Path,
    exp_dir: Path,
    best_path: Path,
    final_path: Path,
    losers: Sequence[Path],
    epoch_map: Dict[Path, int],
    dry: bool,
) -> None:
    prefix = "[DRY] " if dry else ""
    version_rel = _format_rel(version_dir, exp_dir)
    best_rel = _format_rel(best_path, exp_dir)
    final_rel = _format_rel(final_path, exp_dir)
    best_epoch = epoch_map.get(best_path)
    action = f"rename to {final_rel}" if best_path != final_path else f"stays at {final_rel}"
    print(f"{prefix}{version_rel}: keep {best_rel} (epoch {_describe_epoch(best_epoch)}) -> {action}")
    if losers:
        for path in losers:
            epoch_text = _describe_epoch(epoch_map.get(path))
            print(f"{prefix}  delete {_format_rel(path, exp_dir)} (epoch {epoch_text})")
    else:
        print(f"{prefix}  no other checkpoints to delete")


def _apply_actions(best_path: Path, final_path: Path, losers: Iterable[Path], dry: bool) -> None:
    if dry:
        return
    if best_path != final_path:
        if final_path.exists():
            final_path.unlink()
        best_path.rename(final_path)
    for path in losers:
        if path.exists():
            path.unlink()


def keep_largest_epoch_models(target: str, saves_dir: Path, dry: bool) -> None:
    target_path = _resolve_target_path(saves_dir, target)
    exp_dir, version_dirs = _determine_scope(target_path)
    for version_dir in version_dirs:
        try:
            losers, epoch_map, best_path, final_path = _plan_version(version_dir, dry)
        except FileNotFoundError as exc:
            rel_version = _format_rel(version_dir, exp_dir)
            print(f"Skipping {rel_version}: {exc}")
            continue
        _log_version_actions(version_dir, exp_dir, best_path, final_path, losers, epoch_map, dry)
        _apply_actions(best_path, final_path, losers, dry)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep only the largest-epoch last*.ckpt within each version of an experiment"
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Experiment folder (e.g., chaksu128) or specific version (e.g., chaksu128/version_0) relative to saves",
    )
    parser.add_argument(
        "--saves-dir",
        default=None,
        help="Override the default values/saves directory",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Print planned actions without applying changes",
    )
    args = parser.parse_args()
    saves_dir = _resolve_saves_dir(args.saves_dir)
    keep_largest_epoch_models(args.folder, saves_dir, args.dry)


if __name__ == "__main__":
    main()
