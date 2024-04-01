"""
builders.py

Utility functions for writing Map and Iterable (WebDataset, Mosaic Streaming) variants of various evaluation datasets.
"""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks.harnesses.gqa import GQAIndexDataset, build_gqa_indices
from vlm_eval.tasks.harnesses.ocidref import OCIDRefIndexDataset, build_ocidref_indices
from vlm_eval.tasks.harnesses.pope import PopeIndexDataset, build_pope_indices
from vlm_eval.tasks.harnesses.refcoco import RefCOCOIndexDataset, build_refcoco_indices
from vlm_eval.tasks.harnesses.tallyqa import TallyQAIndexDataset, build_tallyqa_indices
from vlm_eval.tasks.harnesses.textvqa import TextVQAIndexDataset, build_textvqa_indices
from vlm_eval.tasks.harnesses.vizwiz import VizWizIndexDataset, build_vizwiz_indices
from vlm_eval.tasks.harnesses.vqav2 import VQAv2IndexDataset, build_vqav2_indices
from vlm_eval.tasks.harnesses.vsr import VSRIndexDataset, build_vsr_indices
from vlm_eval.tasks.harnesses.ai2d import AI2DIndexDataset, build_ai2d_indices

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Define Dispatch Registry for each Task (Dataset Family) ===
BUILDER_DISPATCH: Dict[str, Dict[str, Callable]] = {
    # fmt: off

    # "Standard" Datasets (from Literature)
    "vqa-v2": {"build_indices": build_vqav2_indices, "get_index_datasets": VQAv2IndexDataset},
    "gqa": {"build_indices": build_gqa_indices, "get_index_datasets": GQAIndexDataset},
    "vizwiz": {"build_indices": build_vizwiz_indices, "get_index_datasets": VizWizIndexDataset},
    "pope": {"build_indices": build_pope_indices, "get_index_datasets": PopeIndexDataset},
    "text-vqa": {"build_indices": build_textvqa_indices, "get_index_datasets": TextVQAIndexDataset},
    "vsr": {"build_indices": build_vsr_indices, "get_index_datasets": VSRIndexDataset},
    "refcoco": {"build_indices": build_refcoco_indices, "get_index_datasets": RefCOCOIndexDataset},
    "ocid-ref": {"build_indices": build_ocidref_indices, "get_index_datasets": OCIDRefIndexDataset},
    "tally-qa": {"build_indices": build_tallyqa_indices, "get_index_datasets": TallyQAIndexDataset},
    "ai2d": {"build_indices": build_ai2d_indices, "get_index_datasets": AI2DIndexDataset},

    # fmt: on
}


def build_index_datasets(
    dataset_family: str, root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]] = None, seed: int = 21
) -> List[Dataset]:
    """
    Given a dataset identifier and optional list of dataset sizes, return a set of PyTorch Map-style Datasets
    (building metadata/index files if necessary) that wrap the metadata fields of the given dataset (e.g., returning
    image paths and strings instead of processed image tensors or tokens).

    These "index" datasets are to be used for local debugging, and more importantly for synthesizing Iterable,
    compressed datasets (WebDataset, Mosaic Streaming).

    To enable this, the underlying assumptions we make for each "index" dataset are as follows:
        1) Entire dataset metadata fits into RAM (to enable quick indexing)
        2) Individual media files (images) exist on local disk, allowing for random access given an image path.

    We define the properties/attributes of each dataset type below (... denotes dataset-specific metadata):
        + `dataset_type = vqa`:
            -> metadata{-`n_slim in slim_datasets_sizes`}.json
                {"question_id" -> {"question_id": str, "question": str, "img_path": Path, "answer": str, ...}}

    :param dataset_family: Dataset family (e.g., "vqa-v2" | "nocaps" | ...) to load from `DATASET_REGISTRY`
    :param root_dir: Absolute path to the project's default root directory with task/downloaded data
    :param slim_dataset_sizes: List of "slim" dataset sizes to build (each "slim" dataset is a subset of the larger)
    :param seed: Random seed for setting initial order of examples in each dataset (some datasets sort questions)

    :return: List of "index" datasets (Pytorch Dataset) of length (1 + len(slim_dataset_sizes))
    """
    overwatch.info(f"Building Index Files for Dataset Family `{dataset_family}`", ctx_level=1)
    assert dataset_family in BUILDER_DISPATCH, f"Dataset Family `{dataset_family}` does not have a valid IndexDataset!"
    index_files = BUILDER_DISPATCH[dataset_family]["build_indices"](root_dir, slim_dataset_sizes, seed=seed)

    overwatch.info("Assembling Map-Style Datasets from Index Files", ctx_level=1)
    index_datasets = [BUILDER_DISPATCH[dataset_family]["get_index_datasets"](root_dir, f) for f in index_files]

    return index_datasets
