"""
score.py

Aggregation & "official scoring" for all VLM-Bench evaluations; to be run after dumping generations via `evaluate.py`.

Where possible, uses the *official evaluation script* to evaluate the performance of a model on a validation/testdev
split --> as an example, using the official GQA `eval.py` or VQAv2 leaderboard script for evaluating VQA performance.

Run from the repository root:
    => python scripts/score.py < args >
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import draccus
import yaml
import time

from vlm_eval.conf import DatasetConfig, DatasetRegistry
from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks import get_scorer

# === By default - official scoring scripts only support the `-full` dataset variants; add overrides below ===
VALID_DATASET_ID_OVERRIDES = {}


# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)

@dataclass
class ScoreConfig:
    # fmt: off

    # DatasetConfig from `vlm_eval/conf/datasets.py`; override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.AI2D_FULL.dataset_id)
    )

    # === Model Parameters =>> Prismatic ===
    model_id: str = "prism-clip+7b"                 # Model ID to load and run (instance of `model_family`)

    # === Model Parameters =>> Official LLaVa ===
    # model_id: str = "llava-v1.5-7b"

    # === Model Parameters =>> Official InstructBLIP ===
    # model_id: str = "instructblip-vicuna-7b"

    config_yaml: Optional[Path] = None

    # Artifact Parameters
    results_dir: Path = Path(                       # Path to results directory (writing predicted output, metrics)
        "results"
    )

    # fmt: on


def score_after_parse(cfg):
    overwatch.info(f"Starting Official Scoring for Dataset `{cfg.dataset.dataset_id}` => Model `{cfg.model_id}`")

    # Short-Circuit (if results/metrics already exist)
    dataset_family, dataset_id = cfg.dataset.dataset_family, cfg.dataset.dataset_id
    task_results_dir = cfg.results_dir / cfg.dataset.dataset_family / cfg.dataset.dataset_id / cfg.model_id
    if (metrics_json := task_results_dir / "metrics.json").exists():
        overwatch.info(f"Metrics JSON already exists at `{metrics_json}` =>> Exiting!")
        with open(metrics_json, "r") as f:
            metrics = json.load(f)
            model, dataset, split, summary, experiment_tags = (
                metrics["model"],
                metrics["dataset"],
                cfg.dataset.split,
                metrics["summary"],
                metrics["experiment_tags"],
            )
            accuracy_keys = [k for k in metrics["summary"].keys() if (k.startswith("accuracy__") or k == "accuracy")]
            if len(accuracy_keys) == 1:
                result_string = (
                    f"Results for Model `{model}` on {dataset} (Split = {split})\n"
                    f"          => Accuracy (Official): {summary['accuracy']:.3f}"
                )
            else:
                dataset_names = [k.split("__")[1] for k in accuracy_keys]
                result_string = (
                    f"Results for Model `{model}` on {dataset} ({'/'.join(dataset_names)}) (Split = {split})\n"
                )
                for d in dataset_names:
                    result_string += f"          => {d}  Accuracy (Official): {summary[f'accuracy__{d}']:.3f}\n"

            # Log to Console
            overwatch.info(result_string.strip())

        return

    # Merge per-Rank Results & Assert on Expected Length
    full_results = {}
    for rank_json in task_results_dir.glob("results+rank*.json"):
        with open(rank_json, "r") as f:
            full_results.update(json.load(f))

    # Validate on Expected # of Examples
    assert (
        len(full_results) == cfg.dataset.expected_examples
    ), f"Expected {cfg.dataset.expected_examples} model outputs, only found {len(full_results)}!"

    # Per-Family Dataset Handling
    root_dir = cfg.dataset.root_dir
    scorer = get_scorer(
        dataset_family,
        dataset_id,
        task_results_dir,
        full_results,
        annotations_file=root_dir / cfg.dataset.annotations_file,
        questions_file=root_dir / cfg.dataset.questions_file if cfg.dataset.questions_file is not None else None,
        split=cfg.dataset.split,
    )
    num_retry = 3
    for i in range(num_retry):
        try:
            summary_scores = scorer.score(cfg.model_id)
            break
        except Exception as e:
            if i < num_retry - 1:
                overwatch.warning(f"Error in scoring: {e}; retrying...")
                time.sleep(5)
            else:
                overwatch.warning(f"#!# Error in scoring: {e}; skiping... #!#")

    # Open Model Config =>> `config.yaml`
    if cfg.config_yaml is not None:
        with open(cfg.config_yaml, "r") as f:
            full_cfg = yaml.safe_load(f)

        # Extract Experiment "Tag" Parameters =>> for Leaderboard Display
        experiment_tags = {k: full_cfg["model"][k] for k in ["experiment_tag", "config_line_no", "model_split"]}
    else:
        # Experiment Tags for "Official Models" don't make sense; set to empty
        experiment_tags = {}

    # Finalize Metrics & Write to Disk
    metrics = {
        "dataset": cfg.dataset.dataset_id,
        "n_examples": cfg.dataset.expected_examples,
        "model": cfg.model_id,
        "experiment_tags": experiment_tags,
        "summary": summary_scores,
        "examples": full_results,
    }
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)


@draccus.wrap()
def score(cfg: ScoreConfig) -> None:
    score_after_parse(cfg)

if __name__ == "__main__":
    score()
