from itertools import product
from pathlib import Path
import shutil

import hydra
from omegaconf import ListConfig
import sys

from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

from experiment_version import ExperimentVersion
from experiment_dataloader import ExperimentDataloader
from pydantic.utils import deep_update


class EvalExperiments:
    def __init__(self, config):
        # base path is the path to the first experiment cycle
        self.base_path = Path(config.base_path)
        self.second_cycle_path = (
            config.second_cycle_path if "second_cycle_path" in config.keys() else None
        )
        self.versions = self._init_versions(config)
        self.tasks = config.tasks
        self.config = config
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
                    ["predictive_uncertainty"]
                    if only_pu
                    else [
                        "predictive_uncertainty",
                        "aleatoric_uncertainty",
                        "epistemic_uncertainty",
                    ]
                )
                exp_version = ExperimentVersion(**version_params)
                versions.append(exp_version)
        return versions

    def analyse_accumulated(self, task_params):
        # This is only used if the results are accumulated across multiple versions
        results_dict_task = {}
        for version in self.versions:
            if "datasets" in task_params.keys():
                dataset_splits = task_params["datasets"]
            else:
                dataset_splits = [None]
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
            if "datasets" in task_params.keys():
                dataset_splits = task_params["datasets"]
            else:
                dataset_splits = [None]
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
        - `aleatoric_uncertainty`
        - `epistemic_uncertainty`
        - `pred_entropy`
        - `pred_seg`

        JSON files and other files are left untouched.
        """
        folders_to_remove = [
            "aleatoric_uncertainty",
            "epistemic_uncertainty",
            "pred_entropy",
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


@hydra.main(config_path="configs", config_name="eval_config_lidc_small", version_base=None)
def main(eval_config):
    evaluator = EvalExperiments(eval_config)
    evaluator.analyse()


if __name__ == "__main__":
    main()
