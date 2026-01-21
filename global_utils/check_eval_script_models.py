from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

# Ensure repository root is on sys.path for shared utilities
REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
  sys.path.append(REPO_ROOT.as_posix())


SAMPLE_YAML = """
OOD_AUG:
  iter_params:
    shift: [ "ood_aug" ]
    pred_model: [ "diffusion", "softmax", "ssn" ]
    seed: [ "120" ]
    network: [ "unet-s"]
    eu: [ "dropout", "swag", "swag_diag" ]
    cfg: [ "standard" ]
    data: [ "lidc_2d_small" ]
    aug: [ "", "_aug" ]
  naming_scheme_pred_model: "ood_aug"
  naming_scheme_version: "{eu}_{pred_model}_{network}_lidc_2d_small{aug}/e320_ema"
  aggregations: [ "patch_level", "threshold" ]
  fold: 0
  rank: 5
  image_ending: ".png"
  unc_ending: ".tif"
  n_reference_segs: 4
  only_pu: false
  evaluate_training_data: ${evaluate_training_data}
"""


@dataclass
class EvalSpec:
  name: str
  iter_params: Dict[str, Sequence[str]]
  naming_scheme_pred_model: str
  naming_scheme_version: str


def _parse_eval_specs(yaml_text: str) -> List[EvalSpec]:
  data = yaml.safe_load(yaml_text)
  specs: List[EvalSpec] = []
  if not isinstance(data, dict):
    raise ValueError("Top-level YAML must be a mapping")
  for name, cfg in data.items():
    if not isinstance(cfg, dict):
      raise ValueError(f"Evaluation entry '{name}' must be a mapping")
    iter_params = cfg.get("iter_params")
    if not iter_params:
      raise ValueError(f"Eval entry '{name}' lacks iter_params")
    converted: Dict[str, Sequence[str]] = {}
    for key, value in iter_params.items():
      if isinstance(value, str):
        converted[key] = [value]
      elif isinstance(value, (list, tuple)):
        converted[key] = list(value)
      else:
        raise TypeError(f"iter_params[{key}] must be list or str")
    naming_scheme_pred_model = cfg.get("naming_scheme_pred_model")
    naming_scheme_version = cfg.get("naming_scheme_version")
    if naming_scheme_pred_model is None or naming_scheme_version is None:
      raise ValueError(
        f"Eval entry '{name}' must define naming_scheme_pred_model and naming_scheme_version"
      )
    specs.append(
      EvalSpec(
        name=name,
        iter_params=converted,
        naming_scheme_pred_model=str(naming_scheme_pred_model),
        naming_scheme_version=str(naming_scheme_version),
      )
    )
  return specs


def _expand_iter_params(iter_params: Dict[str, Sequence[str]]) -> Iterable[Dict[str, str]]:
  keys = sorted(iter_params.keys())
  values_product = product(*(iter_params[key] for key in keys))
  for combo in values_product:
    yield {key: value for key, value in zip(keys, combo)}


def _format_pred_model_dir(base_path: Path, schema: str, combo: Dict[str, str]) -> Path:
  formatted = schema.format(**combo)
  return base_path / formatted


def _format_version_dir(pred_model_dir: Path, schema: str, combo: Dict[str, str]) -> Path:
  formatted = schema.format(**combo)
  return pred_model_dir / "test_results" / formatted


def _validate_paths(base_path: Path, spec: EvalSpec) -> Tuple[int, List[str]]:
  missing: List[str] = []
  total_checked = 0
  for combo in _expand_iter_params(spec.iter_params):
    total_checked += 1
    pred_model_dir = _format_pred_model_dir(base_path, spec.naming_scheme_pred_model, combo)
    version_dir = _format_version_dir(pred_model_dir, spec.naming_scheme_version, combo)
    if not pred_model_dir.exists():
      missing.append(f"{pred_model_dir} (missing model directory)")
    if not version_dir.exists():
      missing.append(f"{version_dir} (missing test_results entry)")
  return total_checked, missing


def check_eval_script_models(
  yaml_text: str,
  base_path: Path,
) -> None:
  specs = _parse_eval_specs(yaml_text)
  total_models = 0
  total_missing = 0
  report: Dict[str, List[str]] = {}

  for spec in specs:
    checked, missing_entries = _validate_paths(base_path, spec)
    total_models += checked
    if missing_entries:
      report[spec.name] = missing_entries
      total_missing += len(missing_entries)

  print(f"Evaluated {total_models} parameter combinations across {len(specs)} specs.")
  if not report:
    print("All expected model and test_results directories are present.")
    return
  print(f"Missing entries: {total_missing}")
  for name, entries in report.items():
    print(f"[{name}] Missing:")
    for entry in entries:
      print(f"  - {entry}")


def _build_cli_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Validate that eval config iter_params correspond to existing model/test_results folders."
  )
  parser.add_argument(
    "-y",
    "--yaml-text",
    type=str,
    default=None,
    help="Inline YAML snippet describing evaluation specs. Defaults to SAMPLE_YAML.",
  )
  parser.add_argument(
    "-f",
    "--yaml-file",
    type=str,
    default=None,
    help="Path to a YAML file defining the evaluation specs.",
  )
  parser.add_argument(
    "-b",
    "--base-path",
    type=str,
    default="/home/jloch/Desktop/diff/luzern/values/saves",
    help="Base path containing saved models (e.g., values/saves).",
  )
  return parser


def _load_yaml_from_args(args: argparse.Namespace) -> str:
  if args.yaml_text:
    return args.yaml_text
  if args.yaml_file:
    return Path(args.yaml_file).expanduser().read_text()
  return SAMPLE_YAML


def main() -> None:
  parser = _build_cli_parser()
  args = parser.parse_args()
  yaml_text = _load_yaml_from_args(args)
  base_path = Path(args.base_path).expanduser()
  check_eval_script_models(yaml_text, base_path)


if __name__ == "__main__":
  main()