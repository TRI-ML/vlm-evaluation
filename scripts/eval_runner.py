# export model_dir=$model_id+stage-finetune+x7

# # List of tasks
# tasks=("vqa-v2-full" "gqa-full" "vizwiz-full" "text-vqa-full" "refcoco-full" "ocid-ref-full" "pope-full")

# # Loop through each task and run the evaluation and scoring scripts
# for task in "${tasks[@]}"; do
#     echo "Running task: $task"
#     accelerate launch --num_processes=8 ../vlm-evaluation/scripts/evaluate.py --model_id $model_id --model_dir runs/$model_dir/ --dataset.type $task --dataset.root_dir /datasets/prismatic-vlms/vlm-evaluation/
# done


# for task in "${tasks[@]}"; do
#         echo "Evaluating task: $task"
#     python ../vlm-evaluation/scripts/score.py --model_id $model_id --dataset.type $task --dataset.root_dir /datasets/prismatic-vlms/vlm-evaluation/ --results_dir results
# done
import os
import argparse
import draccus
from dataclasses import dataclass, field
from accelerate.utils import set_seed
from pathlib import Path
from typing import Union, Optional

from prismatic.util.distributed_utils import world_info_from_env
from prismatic.util.file_utils import remote_sync_with_expon_backoff
from vlm_eval.models import load_vlm
from vlm_eval.conf.datasets import DatasetConfig, DatasetRegistry
from scripts.evaluate import EvaluationConfig, evaluate_after_parse
from scripts.score import ScoreConfig, score_after_parse

TASK_LIST=["vqa-v2-full", "vqa-v2-slim", "gqa-full", "vizwiz-full", "text-vqa-full", "refcoco-full", "ocid-ref-full", "pope-full"]

@dataclass
class EvalRunnerConfig:
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.AI2D_FULL.dataset_id)
    )

    # Eval Configs
    tasks: str = "vqa-v2-full"                      # comma-separated (or use 'all')
     
    # === Model Parameters =>> Prismatic ===
    model_family: str = "prismatic"                 # Model family to load from in < `prismatic` | `llava-v15` | ... >
    model_id: Optional[str] = None
    model_dir: Optional[str] = None                 # Can be local or S3

    # Inference Parameters
    device_batch_size: int = 1                      # Device Batch Size set to 1 until LLaVa/HF LLaMa fixes bugs!
    num_workers: int = 2                            # Number of Dataloader Workers (on each process)

    # Artifact Parameters
    dataset_dir: Path = Path(                       # Path to dataset directory
        "/datasets/prismatic-vlms/vlm-evaluation/"
    )
    results_dir: Path = Path(                       # Path to results directory (writing predicted output, metrics)
        "results"
    )
    remote_sync: str = None
    remote_sync_frequency: int = 300

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")  # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                  # Random Seed (for reproducibility)

    def __post_init__(self) -> None:
        self.run_dir = self.model_dir


@draccus.wrap()
def main(cfg: EvalRunnerConfig):
    assert cfg.model_id is not None and cfg.model_dir is not None

    # Parse tasks
    datasets = []
    if cfg.tasks == "all":
        tasks = TASK_LIST
    else:
        tasks = cfg.tasks.split(',')
    for t in tasks:
        assert t in TASK_LIST, f"Task {t} not found in TASK_LIST {TASK_LIST}"
        datasets.append(DatasetConfig.get_choice_class(t))

    # Load VLM
    cfg.run_dir = cfg.model_dir
    set_seed(cfg.seed)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]

    vlm = load_vlm(cfg.model_family, cfg.model_id, cfg.run_dir, hf_token=hf_token, ocr=cfg.dataset.ocr)

    for dataset in datasets:
        cfg.dataset = dataset
        cfg.dataset.root_dir = Path(cfg.dataset_dir)
        print(f"Now evaluating: {dataset.dataset_id}")
        evaluate_after_parse(cfg=cfg, vlm=vlm)

        sc = ScoreConfig()
        sc.dataset = dataset
        sc.dataset.root_dir = Path(cfg.dataset_dir)
        sc.model_id = cfg.model_id
        print(f"Now scoring: {dataset.dataset_id}")
        score_after_parse(cfg=sc)

        _, global_rank, _ = world_info_from_env()
        if global_rank == 0 and cfg.remote_sync is not None:
            print(f"Syncing results to {os.path.join(cfg.remote_sync, cfg.results_dir)}:")
            task_name_full = dataset.dataset_id
            task_name_short = task_name_full[:-5]
            result = remote_sync_with_expon_backoff(
                cfg.remote_sync_frequency,
                os.path.join(cfg.results_dir, task_name_short, task_name_full, cfg.model_id),
                os.path.join(cfg.remote_sync, cfg.results_dir, task_name_short, task_name_full, cfg.model_id),
            )
            if result:
                print("Final remote sync successful.")
            else:
                print("Final remote sync failed.")


if __name__=="__main__":
    main()