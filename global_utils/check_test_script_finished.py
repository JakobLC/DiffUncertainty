from __future__ import annotations

import shlex
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure repository root (values/) is importable when running this utility directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
	sys.path.append(REPO_ROOT.as_posix())

from uncertainty_modeling.unc_mod_utils.test_utils import (
	_build_checkpoint_groups,
	_build_group_version_name,
	_ema_mode_to_flags,
	_extract_version_from_ckpt_path,
	_parse_test_splits,
)


command = """python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_diffusion_unet-s_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_diffusion_unet-s_lidc_2d_small/checkpoints/last.ckpt" --version_name swag_diag_diffusion_unet-s_lidc_2d_small --no-swag-low-rank-cov --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_ssn_unet-m_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_ssn_unet-m_lidc_2d_small/checkpoints/last.ckpt" --version_name swag_diag_ssn_unet-m_lidc_2d_small --no-swag-low-rank-cov --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/dropout_diffusion_unet-m_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_diffusion_unet-s_lidc_2d_small_aug/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_diffusion_unet-s_lidc_2d_small_aug/checkpoints/last.ckpt" --version_name swag_diag_diffusion_unet-s_lidc_2d_small_aug --no-swag-low-rank-cov --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_diffusion_unet-m_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_diffusion_unet-m_lidc_2d_small/checkpoints/last.ckpt" --version_name swag_diag_diffusion_unet-m_lidc_2d_small --no-swag-low-rank-cov --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_ssn_unet-s_lidc_2d_small_aug/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_ssn_unet-s_lidc_2d_small_aug/checkpoints/last.ckpt" --version_name swag_diag_ssn_unet-s_lidc_2d_small_aug --no-swag-low-rank-cov --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_softmax_unet-s_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_softmax_unet-s_lidc_2d_small/checkpoints/last.ckpt" --version_name swag_diag_softmax_unet-s_lidc_2d_small --no-swag-low-rank-cov --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/dropout_softmax_unet-m_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/dropout_softmax_unet-s_lidc_2d_small_aug/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_ssn_unet-s_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_ssn_unet-s_lidc_2d_small/checkpoints/last.ckpt" --version_name swag_diag_ssn_unet-s_lidc_2d_small --no-swag-low-rank-cov --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/dropout_softmax_unet-s_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/dropout_diffusion_unet-s_lidc_2d_small_aug/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/dropout_ssn_unet-s_lidc_2d_small_aug/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/dropout_ssn_unet-s_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/dropout_diffusion_unet-s_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_softmax_unet-m_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_softmax_unet-m_lidc_2d_small/checkpoints/last.ckpt" --version_name swag_diag_softmax_unet-m_lidc_2d_small --no-swag-low-rank-cov --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/dropout_ssn_unet-m_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_softmax_unet-s_lidc_2d_small_aug/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/swag_softmax_unet-s_lidc_2d_small_aug/checkpoints/last.ckpt" --version_name swag_diag_softmax_unet-s_lidc_2d_small_aug --no-swag-low-rank-cov --test_split id_test,val,ood_blur,ood_noise --ema_mode ema
# f
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/ens12*_ssn_unet-s_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema --ensemble_mode --wildcard_replace 1,2,3,4,5
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/ens12*_diffusion_unet-s_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema --ensemble_mode --wildcard_replace 1,2,3,4,5
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/ens12*_diffusion_unet-m_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema --ensemble_mode --wildcard_replace 1,2,3,4,5
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/ens12*_softmax_unet-m_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema --ensemble_mode --wildcard_replace 1,2,3,4,5
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/ens12*_softmax_unet-s_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema --ensemble_mode --wildcard_replace 1,2,3,4,5
python uncertainty_modeling/test_2D.py --checkpoint_paths "/home/jloch/Desktop/diff/luzern/values/saves/ood_aug/ens12*_ssn_unet-m_lidc_2d_small/checkpoints/last.ckpt" --test_split id_test,val,ood_blur,ood_noise --ema_mode ema --ensemble_mode --wildcard_replace 1,2,3,4,5
# f
python evaluation/eval_experiments.py
f"""
#python evaluation/eval_experiments.py experiments=[${OOD_AUG_ENSEMBLE}]

@dataclass
class ParsedCommand:
	original: str
	checkpoint_entries: List[str]
	test_split_arg: str
	wildcard_tokens: List[str] = field(default_factory=list)
	ensemble_mode: bool = False
	ema_mode: str = "normal"
	version_override: Optional[str] = None
	exp_name_override: Optional[str] = None
	save_dir_override: Optional[str] = None


@dataclass
class MissingDetail:
	version: str
	split_raw: str
	split_folder: str
	ema_label: str
	results_root: Path
	expect_ema: bool


def _parse_test_command(line: str) -> Optional[ParsedCommand]:
	stripped = line.strip()
	if not stripped or stripped.startswith("#"):
		return None
	if "test_2D.py" not in stripped:
		return None
	tokens = shlex.split(stripped)
	checkpoint_entries: List[str] = []
	test_split_arg: Optional[str] = None
	wildcard_tokens: List[str] = []
	version_override: Optional[str] = None
	exp_name_override: Optional[str] = None
	save_dir_override: Optional[str] = None
	ema_mode = "normal"
	ensemble_mode = False

	idx = 0
	while idx < len(tokens):
		token = tokens[idx]
		if token in {"python", "python3"} or "test_2D.py" in token:
			idx += 1
			continue
		if token == "--checkpoint_paths":
			idx += 1
			values: List[str] = []
			while idx < len(tokens) and not tokens[idx].startswith("--"):
				values.append(tokens[idx])
				idx += 1
			if not values:
				raise ValueError(f"--checkpoint_paths missing values in command: {line}")
			checkpoint_entries.extend(values)
			continue
		if token == "--test_split":
			if idx + 1 >= len(tokens):
				raise ValueError(f"--test_split missing value in command: {line}")
			test_split_arg = tokens[idx + 1]
			idx += 2
			continue
		if token == "--version_name":
			if idx + 1 >= len(tokens):
				raise ValueError(f"--version_name missing value in command: {line}")
			version_override = tokens[idx + 1]
			idx += 2
			continue
		if token == "--exp_name":
			if idx + 1 >= len(tokens):
				raise ValueError(f"--exp_name missing value in command: {line}")
			exp_name_override = tokens[idx + 1]
			idx += 2
			continue
		if token == "--save_dir":
			if idx + 1 >= len(tokens):
				raise ValueError(f"--save_dir missing value in command: {line}")
			save_dir_override = tokens[idx + 1]
			idx += 2
			continue
		if token == "--wildcard_replace":
			if idx + 1 >= len(tokens):
				raise ValueError(f"--wildcard_replace missing value in command: {line}")
			wildcard_tokens = [chunk.strip() for chunk in tokens[idx + 1].split(",") if chunk.strip()]
			idx += 2
			continue
		if token == "--ema_mode":
			if idx + 1 >= len(tokens):
				raise ValueError(f"--ema_mode missing value in command: {line}")
			ema_mode = tokens[idx + 1]
			idx += 2
			continue
		if token == "--ensemble_mode":
			ensemble_mode = True
			idx += 1
			continue
		# Skip flags we do not explicitly need (boolean or value-less)
		idx += 1

	if not checkpoint_entries:
		raise ValueError(f"No --checkpoint_paths provided in command: {line}")
	if test_split_arg is None:
		raise ValueError(f"--test_split is required in command: {line}")
	return ParsedCommand(
		original=stripped,
		checkpoint_entries=checkpoint_entries,
		test_split_arg=test_split_arg,
		wildcard_tokens=wildcard_tokens,
		ensemble_mode=ensemble_mode,
		ema_mode=ema_mode,
		version_override=version_override,
		exp_name_override=exp_name_override,
		save_dir_override=save_dir_override,
	)


def _infer_layout_from_checkpoint(path_str: str) -> Tuple[Path, str, str]:
	path = Path(path_str).expanduser()
	if path.is_file() and path.suffix == ".ckpt":
		parent = path.parent
		if parent.name in {"checkpoints", "scheduled_ckpts"} and parent.parent:
			exp_dir = parent.parent
		else:
			exp_dir = parent
	else:
		exp_dir = path
	exp_parent = exp_dir.parent
	if exp_parent is None or exp_parent.parent is None:
		raise ValueError(f"Cannot infer save directory layout from checkpoint path: {path_str}")
	save_root = exp_parent.parent
	exp_name = exp_parent.name
	version_from_path = exp_dir.name
	return save_root, exp_name, version_from_path


def _format_split_folder(split_name: str) -> str:
	return split_name[:-5] if split_name.endswith("_test") else split_name


def _metrics_exists(results_root: Path, split_folder: str, expect_ema: bool) -> bool:
	if not results_root.exists():
		return False
	direct = results_root / split_folder / "metrics.json"
	if direct.exists():
		return True
	try:
		children = list(results_root.iterdir())
	except FileNotFoundError:
		return False
	for child in children:
		if not child.is_dir():
			continue
		is_ema_dir = child.name.endswith("_ema")
		if expect_ema and not is_ema_dir:
			continue
		if not expect_ema and is_ema_dir:
			continue
		candidate = child / split_folder / "metrics.json"
		if candidate.exists():
			return True
	return False


def _build_rerun_command(original_line: str, missing_splits: Sequence[str]) -> str:
	tokens = shlex.split(original_line)
	new_tokens: List[str] = []
	idx = 0
	replaced = False
	while idx < len(tokens):
		token = tokens[idx]
		new_tokens.append(token)
		if token == "--test_split":
			if idx + 1 >= len(tokens):
				raise ValueError(f"Malformed --test_split in command: {original_line}")
			idx += 1  # Skip original value
			new_tokens.append(",".join(missing_splits))
			replaced = True
		idx += 1
	if not replaced:
		raise ValueError(f"No --test_split flag found in command: {original_line}")
	return " ".join(shlex.quote(token) for token in new_tokens)


def check_test_script_finished(multiline_commands: str) -> None:
	parsed_commands: List[ParsedCommand] = []
	for raw_line in multiline_commands.splitlines():
		parsed = _parse_test_command(raw_line)
		if parsed is not None:
			parsed_commands.append(parsed)

	total_expected = 0
	total_missing = 0
	rerun_entries: List[Tuple[str, Dict[str, List[MissingDetail]]]] = []
	split_orders: Dict[str, List[str]] = {}

	for cmd in parsed_commands:
		split_list = _parse_test_splits(cmd.test_split_arg)
		split_orders[cmd.original] = split_list
		split_to_missing: Dict[str, List[MissingDetail]] = {}

		checkpoint_groups = _build_checkpoint_groups(
			cmd.checkpoint_entries,
			cmd.wildcard_tokens,
			cmd.ensemble_mode,
		)
		ema_options = _ema_mode_to_flags(cmd.ema_mode)

		for group in checkpoint_groups:
			auto_version = _build_group_version_name(
				cmd.checkpoint_entries,
				cmd.wildcard_tokens,
				group,
			)
			version = cmd.version_override or auto_version or _extract_version_from_ckpt_path(group[0])
			save_root, exp_name, version_from_path = _infer_layout_from_checkpoint(group[0])
			if cmd.exp_name_override is not None:
				exp_name = cmd.exp_name_override
			if cmd.save_dir_override is not None:
				save_root = Path(cmd.save_dir_override).expanduser()
			if version is None:
				version = version_from_path
			results_root = Path(save_root) / exp_name / "test_results" / version

			for split in split_list:
				split_folder = _format_split_folder(split)
				for ema_label, use_ema in ema_options:
					total_expected += 1
					success = _metrics_exists(results_root, split_folder, use_ema)
					if not success:
						total_missing += 1
						detail = MissingDetail(
							version=version,
							split_raw=split,
							split_folder=split_folder,
							ema_label=ema_label,
							results_root=results_root,
							expect_ema=use_ema,
						)
						split_to_missing.setdefault(split, []).append(detail)

		if split_to_missing:
			rerun_entries.append((cmd.original, split_to_missing))

	print(f"Missing metrics: {total_missing}/{total_expected}")
	if not total_missing:
		return
	print("Commands to rerun (trimmed to missing splits):")
	for original_cmd, missing_info in rerun_entries:
		split_order = split_orders.get(original_cmd, [])
		ordered_missing = [split for split in split_order if split in missing_info]
		rerun_line = _build_rerun_command(original_cmd, ordered_missing)
		print(rerun_line)
		for split in ordered_missing:
			for detail in missing_info[split]:
				ema_tag = "ema" if detail.expect_ema else "normal"
				hint = f"{detail.results_root}/<e{'*_ema' if detail.expect_ema else '*'}>/{detail.split_folder}/metrics.json"
				print(
					f"  - version={detail.version} split={split} ema={ema_tag} expected at {hint}"
				)


def _load_commands_from_args(args: argparse.Namespace) -> str:
	if args.commands_text:
		return args.commands_text
	if args.commands_file:
		file_path = Path(args.commands_file).expanduser()
		return file_path.read_text()
	return command


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Check for missing metrics.json outputs for test_2D.py command batches."
	)
	parser.add_argument(
		"-t",
		"--commands-text",
		type=str,
		default=None,
		help="Override the command batch inline instead of using the default list.",
	)
	parser.add_argument(
		"-f",
		"--commands-file",
		type=str,
		default=None,
		help="Path to a text file containing the commands to inspect.",
	)
	args = parser.parse_args()
	commands_to_check = _load_commands_from_args(args)
	check_test_script_finished(commands_to_check)


if __name__ == "__main__":
	main()