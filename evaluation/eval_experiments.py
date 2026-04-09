from itertools import product
from pathlib import Path
import shutil
from collections import Counter

import hydra
from omegaconf import ListConfig
import sys

sys.path.append(Path(__file__).parent.parent.as_posix())

from experiment_version import ExperimentVersion
from experiment_dataloader import ExperimentDataloader
from pydantic.utils import deep_update


class EvalExperiments:
    LIDC_SPLITS = ["id", "val", "ood_noise", "ood_blur", "ood_jpeg", "ood_contrast"]
    CHAKSU_SPLITS = ["id", "val", "ood"]

    def __init__(self, config):
        # base path is the path to the first experiment cycle
        self.base_path = Path(config.base_path)
        self.second_cycle_path = (
            config.second_cycle_path if "second_cycle_path" in config.keys() else None
        )
        self.versions = self._init_versions(config)
        self.tasks = config.tasks
        self.config = config
        self._version_status = {}
        return

    def _init_versions(self, config):
        versions = []
        for experiment in config.experiments:
            filtered_config = [
                [(key, v) for v in values]
                for key, values in experiment.iter_params.items()
            ]
            for params in product(*filtered_config):
                version_params = {i[0]: i[1] for i in params}
                exp_config = dict(experiment)
                exp_config.pop("iter_params")
                version_params.update(exp_config)
                # Defaults for optional execution controls.
                version_params.setdefault("skip_missing", False)
                version_params.setdefault("skip_finished", False)
                version_params["base_path"] = self.base_path
                version_params["second_cycle_path"] = self.second_cycle_path
                # If the experiment did not include a datamodule_config, allow a top-level
                # `datamodule_config` to be used for all experiments. This lets users
                # define the datamodule once at the root rather than repeating the
                # import for every experiment mapping.
                if "datamodule_config" not in version_params and "datamodule_config" in config:
                    version_params["datamodule_config"] = config.datamodule_config
                # Merge model-dependent settings from config if present (naming schemes, aggregations)
                # but DO NOT trust unc_types from config; we derive it from only_pu below.
                if "prediction_models" in experiment and version_params.get("pred_model") in experiment.prediction_models:
                    model_cfg = dict(experiment.prediction_models[version_params["pred_model"]])
                    # Drop any legacy unc_types from configs to avoid double-definition
                    if "unc_types" in model_cfg:
                        model_cfg.pop("unc_types")
                    version_params.update(model_cfg)

                # Derive uncertainty types from per-experiment only_pu if present,
                # otherwise fall back to top-level config.only_pu for backward compatibility
                only_pu = bool(version_params.get("only_pu", getattr(config, "only_pu", False)))
                pred_model_name = str(version_params.get("pred_model"))
                # Consistency checks as requested
                if pred_model_name == "Softmax" and not only_pu:
                    raise ValueError("only_pu must be True when pred_model is 'Softmax'.")
                if pred_model_name != "Softmax" and only_pu:
                    raise ValueError("only_pu must be False when pred_model is not 'Softmax'.")
                # Set unc_types based on only_pu
                version_params["unc_types"] = (
                    ["TU"]
                    if only_pu
                    else [
                        "TU",
                        "AU",
                        "EU",
                    ]
                )
                exp_version = ExperimentVersion(**version_params)
                versions.append(exp_version)
        return versions

    @staticmethod
    def _required_unc_folders(version):
        required_unc = set()
        for unc_type in version.unc_types:
            if unc_type == "predictive_uncertainty":
                required_unc.add("TU")
            else:
                required_unc.add(str(unc_type))
        return sorted(required_unc)

    @staticmethod
    def _candidate_dataset_dirs(exp_path):
        children = [p for p in sorted(exp_path.iterdir()) if p.is_dir()]
        metric_children = [p for p in children if (p / "metrics.json").is_file()]
        if metric_children:
            return metric_children
        if (exp_path / "metrics.json").is_file():
            return [exp_path]
        return children if children else [exp_path]

    def _expected_dataset_splits(self, version):
        data_key = str(version.version_params.get("data", "")).lower()
        exp_name_key = str(getattr(version, "exp_name", "")).lower()
        combined_key = f"{data_key} {exp_name_key}"
        if "lidc" in combined_key:
            return list(self.LIDC_SPLITS)
        if "chaksu" in combined_key:
            return list(self.CHAKSU_SPLITS)
        return None

    def _all_expected_dataset_dirs(self, version):
        exp_path = Path(version.exp_path)
        expected_splits = self._expected_dataset_splits(version)
        if expected_splits is not None:
            return [exp_path / split for split in expected_splits]
        return self._candidate_dataset_dirs(exp_path)

    def _get_task_params(self, task_name):
        if "task_params" not in self.config:
            return None
        if task_name not in self.config.task_params:
            return None
        return self.config.task_params[task_name]

    def _resolve_task_dataset_dirs(self, version, task_name):
        task_params = self._get_task_params(task_name)
        if task_params is None:
            return []
        dataset_spec = task_params["datasets"] if "datasets" in task_params.keys() else None
        dataset_splits = self._normalize_dataset_spec(dataset_spec)
        exp_path = Path(version.exp_path)

        if len(dataset_splits) == 1 and dataset_splits[0] == "all":
            return self._all_expected_dataset_dirs(version)

        resolved_dirs = []
        for split in dataset_splits:
            if split is None:
                resolved_dirs.append(exp_path)
            elif isinstance(split, str) and "&" in split:
                # Paired splits are not used by the completion files checked here.
                continue
            else:
                resolved_dirs.append(exp_path / str(split))
        return resolved_dirs

    def _is_missing_version(self, version):
        exp_path = Path(version.exp_path)
        if not exp_path.exists():
            return True

        required_folders = ["pred_seg"] + self._required_unc_folders(version)
        for dataset_dir in self._all_expected_dataset_dirs(version):
            for folder_name in required_folders:
                if not (dataset_dir / folder_name).is_dir():
                    return True
        return False

    def _is_finished_version(self, version):
        exp_path = Path(version.exp_path)
        if not exp_path.exists():
            return False

        # "Finished" means outputs for the full evaluation task-set are present.
        # The expected split coverage is task-dependent (e.g. calibration excludes val).
        if self._get_task_params("threshold") is not None:
            for file_name in ["quantile_analysis.json", "threshold_analysis.json"]:
                if not (exp_path / file_name).is_file():
                    return False

        if self._get_task_params("ood_detection") is not None:
            if not (exp_path / "ood_detection.json").is_file():
                return False

        if self._get_task_params("area") is not None:
            for dataset_dir in self._resolve_task_dataset_dirs(version, "area"):
                if not (dataset_dir / "area.json").is_file():
                    return False

        if self._get_task_params("aggregation") is not None:
            required_unc = self._required_unc_folders(version)
            for dataset_dir in self._resolve_task_dataset_dirs(version, "aggregation"):
                for unc_name in required_unc:
                    if not (dataset_dir / f"aggregated_{unc_name}.json").is_file():
                        return False

        if self._get_task_params("calibration") is not None:
            for dataset_dir in self._resolve_task_dataset_dirs(version, "calibration"):
                if not (dataset_dir / "calibration.json").is_file():
                    return False

        if self._get_task_params("ambiguity_modeling") is not None:
            for dataset_dir in self._resolve_task_dataset_dirs(version, "ambiguity_modeling"):
                if not (dataset_dir / "ambiguity_modeling.json").is_file():
                    return False

        return True

    def _classify_versions(self):
        statuses = {}
        for version in self.versions:
            missing = self._is_missing_version(version)
            finished = self._is_finished_version(version)
            statuses[version.exp_path.as_posix()] = {
                "missing": missing,
                "finished": finished,
                "skip_missing": bool(version.version_params.get("skip_missing", False)),
                "skip_finished": bool(version.version_params.get("skip_finished", False)),
            }
        self._version_status = statuses
        return statuses

    def _print_status_summary(self):
        if not self._version_status:
            return
        matrix_counter = Counter(
            (status["missing"], status["finished"])
            for status in self._version_status.values()
        )
        total = len(self._version_status)
        missing_count = sum(1 for status in self._version_status.values() if status["missing"])
        finished_count = sum(
            1 for status in self._version_status.values() if status["finished"]
        )

        print("Preflight version status summary")
        print(
            f"- Missing: {missing_count} | Not missing: {total - missing_count} | Total: {total}"
        )
        print(
            f"- Finished: {finished_count} | Unfinished: {total - finished_count} | Total: {total}"
        )
        print("- Missing x Finished matrix (rows=missing, cols=finished)")
        print("                 finished=False  finished=True")
        print(
            "missing=False"
            f"      {matrix_counter[(False, False)]:>6}"
            f"         {matrix_counter[(False, True)]:>6}"
        )
        print(
            "missing=True "
            f"      {matrix_counter[(True, False)]:>6}"
            f"         {matrix_counter[(True, True)]:>6}"
        )

    def _should_skip_version(self, version):
        status = self._version_status.get(version.exp_path.as_posix(), None)
        if status is None:
            return False
        if status["skip_missing"] and status["missing"]:
            return True
        if status["skip_finished"] and status["finished"]:
            return True
        return False

    def _resolve_dataset_splits(self, task_params, version):
        dataset_spec = task_params["datasets"] if "datasets" in task_params.keys() else None
        dataset_splits = self._normalize_dataset_spec(dataset_spec)
        if len(dataset_splits) == 1 and dataset_splits[0] == "all":
            dataset_splits = self._discover_available_dataset_splits(version)
        return dataset_splits

    @staticmethod
    def _normalize_dataset_spec(dataset_spec):
        if dataset_spec is None:
            return [None]
        if isinstance(dataset_spec, str):
            return [dataset_spec]
        return list(dataset_spec)

    def _discover_available_dataset_splits(self, version):
        exp_path = Path(version.exp_path)
        if not exp_path.exists():
            raise FileNotFoundError(
                f"Experiment path {exp_path} does not exist; cannot discover dataset splits."
            )
        dataset_dirs = []
        missing_metrics = []
        for child in sorted(exp_path.iterdir()):
            if not child.is_dir():
                continue
            metrics_file = child / "metrics.json"
            if metrics_file.is_file():
                dataset_dirs.append(child.name)
            else:
                missing_metrics.append(child.name)
        if dataset_dirs:
            if missing_metrics:
                print(
                    f"Skipping dataset splits without metrics.json under {exp_path}: {missing_metrics}"
                )
            print(f"Auto-detected dataset splits for {exp_path}: {dataset_dirs}")
            return dataset_dirs
        if (exp_path / "metrics.json").is_file():
            print(
                f"Auto-detected single dataset (no split subfolders) for {exp_path}."
            )
            return [None]
        raise FileNotFoundError(
            f"No dataset splits with metrics.json found under {exp_path}; specify datasets explicitly or run evaluations first."
        )

    def analyse_accumulated(self, task_params):
        # This is only used if the results are accumulated across multiple versions
        results_dict_task = {}
        for version in self.versions:
            if self._should_skip_version(version):
                continue
            dataset_splits = self._resolve_dataset_splits(task_params, version)
            for dataset_split in dataset_splits:
                exp_dataloader = ExperimentDataloader(version, dataset_split)
                results_dict = hydra.utils.instantiate(
                    task_params.function,
                    exp_dataloader=exp_dataloader,
                    _recursive_=False,
                )
                results_dict_task = deep_update(results_dict_task, results_dict)
        hydra.utils.instantiate(
            task_params.postprocess_function,
            results_dict=results_dict_task,
            _recursive_=False,
        )

    def analyse_single_version(self, task_params):
        for version in self.versions:
            if self._should_skip_version(version):
                continue
            dataset_splits = self._resolve_dataset_splits(task_params, version)
            for dataset_split in dataset_splits:
                exp_dataloader = ExperimentDataloader(version, dataset_split)
                hydra.utils.instantiate(
                    task_params.function,
                    exp_dataloader=exp_dataloader,
                    _recursive_=False,
                )

    def analyse_subtasks(self, tasks):
        for subtask_params in tasks:
            accumulated = (
                subtask_params.accumulated
                if "accumulated" in subtask_params.keys()
                else False
            )
            if accumulated:
                self.analyse_accumulated(task_params=subtask_params)
            else:
                self.analyse_single_version(task_params=subtask_params)

    def analyse(self):
        self._classify_versions()
        self._print_status_summary()
        for task in self.tasks:
            print(f"ANALYSING TASK: {task}")
            # Special-case cleanup task which does not require an entry in task_params
            if str(task) == "cleanup":
                self.cleanup()
                print(task)
                continue
            task_params = self.config.task_params[task]
            if type(self.config.task_params[task]) == ListConfig:
                self.analyse_subtasks(task_params)
            else:
                accumulated = (
                    task_params.accumulated
                    if "accumulated" in task_params.keys()
                    else False
                )
                if accumulated:
                    self.analyse_accumulated(task_params=task_params)
                else:
                    self.analyse_single_version(task_params=task_params)
                print(task)
        return

    def cleanup(self):
        """Remove large image folders under each version/dataset test folder.

        Deletes the following subdirectories if present under each dataset folder:
        - `AU`
        - `EU`
        - `TU`
        - `pred_seg`

        JSON files and other files are left untouched.
        """
        folders_to_remove = [
            "AU",
            "EU",
            "TU",
            "pred_seg",
        ]
        for version in self.versions:
            exp_path = Path(version.exp_path)
            if not exp_path.exists():
                print(f"Skipping missing version path: {exp_path}")
                continue
            # If there are dataset split subdirectories, iterate them; otherwise treat exp_path as single dataset
            children = [p for p in exp_path.iterdir() if p.is_dir()]
            if children:
                dataset_dirs = children
            else:
                dataset_dirs = [exp_path]

            for dataset_dir in dataset_dirs:
                for sub in folders_to_remove:
                    target = dataset_dir / sub
                    if target.exists():
                        try:
                            if target.is_dir():
                                shutil.rmtree(target)
                                print(f"Removed {target}")
                            else:
                                # If it's a file, remove file
                                target.unlink()
                                print(f"Removed file {target}")
                        except Exception as e:
                            print(f"Failed removing {target}: {e}")


@hydra.main(config_path="configs", config_name="eval_config_lidc", version_base=None)
def main(eval_config):
    evaluator = EvalExperiments(eval_config)
    evaluator.analyse()


if __name__ == "__main__":
    main()
