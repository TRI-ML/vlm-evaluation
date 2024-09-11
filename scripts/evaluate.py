"""
evaluate.py

Entry point for all VLM-Evaluation evaluations; specify model and dataset, get results.

Run with `accelerate` from repository root (for naive parallelization):
    =>> [Single-GPU] CUDA_VISIBLE_DEVICES={0-7} accelerate launch --num_processes=1 scripts/evaluate.py < args >
    =>> [Multi-GPU]  accelerate launch --num_processes={>1} scripts/evaluate.py < args >
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional

import draccus
from accelerate.utils import set_seed

from vlm_eval.conf import DatasetConfig, DatasetRegistry
from vlm_eval.models import load_vlm
from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks import get_task_runner

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)


@dataclass
class EvaluationConfig:
    # fmt: off

    # DatasetConfig from `vlm_eval/conf/datasets.py`; override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.AI2D_FULL.dataset_id)
    )

    # === Model Parameters =>> Prismatic ===
    model_family: str = "prismatic"                 # Model family to load from in < `prismatic` | `llava-v15` | ... >
    model_id: Optional[str] = (                     # Model ID to load and run (instance of `model_family`)
        "prism-clip+7b"
    )
    model_dir: Optional[Path] = None                # Path to model checkpoint to load --> should be self-contained

    # === Model Parameters =>> Official LLaVa ===
    # model_family: str = "llava-v15"
    # model_id: str = "llava-v1.5-7b"
    # model_dir: Path = "liuhaotian/llava-v1.5-7b"

    # === Model Parameters =>> Official InstructBLIP ===
    # model_family: str = "instruct-blip"
    # model_id: str = "instructblip-vicuna-7b"
    # model_dir: Path = "Salesforce/instructblip-vicuna-7b"

    # Inference Parameters
    device_batch_size: int = 1                      # Device Batch Size set to 1 until LLaVa/HF LLaMa fixes bugs!
    num_workers: int = 2                            # Number of Dataloader Workers (on each process)

    # Artifact Parameters
    results_dir: Path = Path(                       # Path to results directory (writing predicted output, metrics)
        "results"
    )

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")  # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                  # Random Seed (for reproducibility)

    def __post_init__(self) -> None:
        self.run_dir = self.model_dir

    # fmt: on

def evaluate_after_parse(cfg, vlm=None):
    overwatch.info(f"Starting Evaluation for Dataset `{cfg.dataset.dataset_id}` w/ Model `{cfg.model_id}`")
    set_seed(cfg.seed)

    # Short-Circuit (if results/metrics already exist)
    task_results_dir = cfg.results_dir / cfg.dataset.dataset_family / cfg.dataset.dataset_id / cfg.model_id
    if (task_results_dir / "metrics.json").exists():
        overwatch.info(f"Metrics for `{cfg.dataset.dataset_id}` w/ `{cfg.model_id}` exist =>> exiting!")
        return

    # Build the VLM --> Download/Load Pretrained Model from Checkpoint
    overwatch.info("Initializing VLM =>> Bundling Models, Image Processors, and Tokenizer")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    if vlm is None:
        vlm = load_vlm(cfg.model_family, cfg.model_id, cfg.run_dir, hf_token=hf_token, ocr=cfg.dataset.ocr)

    # Create Task Runner
    overwatch.info(f"Building Evaluation Runner for Dataset `{cfg.dataset.dataset_id}`")
    task_runner = get_task_runner(
        cfg.dataset.dataset_family,
        cfg.dataset.root_dir,
        cfg.dataset.index_file,
        task_results_dir,
        cfg.model_id,
        prompt_fn=vlm.get_prompt_fn(cfg.dataset.dataset_family),
        image_processor=vlm.image_processor,
    )

    # Run Evaluation
    overwatch.info("Starting (Distributed) Evaluation Loop")
    task_runner.evaluate(vlm, cfg.device_batch_size, cfg.num_workers)


@draccus.wrap()
def evaluate(cfg: EvaluationConfig) -> None:
    evaluate_after_parse(cfg)


if __name__ == "__main__":
    evaluate()
