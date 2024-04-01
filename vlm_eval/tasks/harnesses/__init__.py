from pathlib import Path
from typing import Callable, Dict, Optional, Protocol

from vlm_eval.util.interfaces import VLM, ImageProcessor

from .gqa import GQAScorer, GQATaskRunner
from .ocidref import OCIDRefScorer, OCIDRefTaskRunner
from .pope import PopeScorer, PopeTaskRunner
from .refcoco import RefCOCOScorer, RefCOCOTaskRunner
from .tallyqa import TallyQAScorer, TallyQATaskRunner
from .textvqa import TextVQAScorer, TextVQATaskRunner
from .vizwiz import VizWizScorer, VizWizTaskRunner
from .vqav2 import VQAv2Scorer, VQAv2TaskRunner
from .vsr import VSRScorer, VSRTaskRunner
from .ai2d import AI2DScorer, AI2DTaskRunner


# === Protocol Definitions ===
class TaskRunner(Protocol):
    def evaluate(self, vlm: VLM, device_batch_size: int, num_workers: int) -> None:
        ...


class Scorer(Protocol):
    def score(self, model_id: str) -> Dict[str, float]:
        ...


# === Task Runner Dispatch by Dataset Family ===
DATASET2RUNNER = {
    "vqa-v2": VQAv2TaskRunner,
    "gqa": GQATaskRunner,
    "vizwiz": VizWizTaskRunner,
    "pope": PopeTaskRunner,
    "text-vqa": TextVQATaskRunner,
    "vsr": VSRTaskRunner,
    "tally-qa": TallyQATaskRunner,
    "refcoco": RefCOCOTaskRunner,
    "ocid-ref": OCIDRefTaskRunner,
    "ai2d": AI2DTaskRunner,
}

# === Score Function Dispatch by Dataset Family ===
DATASET2SCORER = {
    "vqa-v2": VQAv2Scorer,
    "gqa": GQAScorer,
    "vizwiz": VizWizScorer,
    "pope": PopeScorer,
    "text-vqa": TextVQAScorer,
    "vsr": VSRScorer,
    "tally-qa": TallyQAScorer,
    "refcoco": RefCOCOScorer,
    "ocid-ref": OCIDRefScorer,
    "ai2d": AI2DScorer,
}


def get_task_runner(
    dataset_family: str,
    root_dir: Path,
    index_file: Path,
    task_results_dir: Path,
    model_id: str,
    prompt_fn: Callable[[str], str],
    image_processor: ImageProcessor,
) -> TaskRunner:
    assert dataset_family in DATASET2RUNNER, f"Dataset Family `{dataset_family}` not supported!"
    return DATASET2RUNNER[dataset_family](root_dir, index_file, task_results_dir, model_id, prompt_fn, image_processor)


def get_scorer(
    dataset_family: str,
    dataset_id: str,
    task_results_dir: Path,
    full_results: Dict[str, Dict],
    annotations_file: Path,
    questions_file: Optional[Path] = None,
    split: str = "val",
) -> Scorer:
    assert dataset_family in DATASET2SCORER, f"Dataset Family `{dataset_family}` not supported!"
    return DATASET2SCORER[dataset_family](
        dataset_id, task_results_dir, full_results, annotations_file, questions_file=questions_file, split=split
    )
