"""
vsr.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the Visual Spatial Reasoning (VSR)
dataset. Only loads and processes the VSR Zero-Shot Test Split (`test.jsonl`) -- the default evaluation split used
by InstructBLIP and follow-up work.
"""
import json
import os
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional, Tuple

import jsonlines
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import Compose
from tqdm import tqdm

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks.registry import DATASET_REGISTRY
from vlm_eval.util.interfaces import VLM, ImageProcessor

# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)


# === Dataset Indexing / Building Utilities ===
def build_vsr_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21) -> List[Path]:
    """Parse VSR --> build & write index files w/ necessary VSR keys + additional dataset-specific data."""
    paths = DATASET_REGISTRY["vsr"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    # Short-Circuit (if index files have already been built)
    assert slim_dataset_sizes is None, "VSR is Tiny -- no slim dataset!"
    index_files = [dataset_dir / "metadata-full.json"]
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    # Otherwise, load the raw annotations (questions & true/false answers) from the downloaded VSR raw data
    with jsonlines.open(root_dir / paths["questions_answers"], "r") as reader:
        examples = [obj for obj in reader]

    # Build Full Metadata Structure
    index = {}
    for idx, example in tqdm(
        enumerate(examples), total=len(examples), desc="=> Processing VSR Raw Dataset:", leave=False
    ):
        example_id: int = idx

        # Get Image Path
        img_path = paths["images"] / example["image"]
        assert (root_dir / img_path).exists(), f"Image `{img_path}` for Example ID `{example_id}` does not exist!"

        # Build Metadata Entry
        # fmt: off
        index[example_id] = {
            # [Required] VSR True/False Task Keys
            "example_id": example_id,
            "caption": example["caption"],
            "img_path": str(img_path),
            "true_false": True if example["label"] == 1 else False,

            # [Dataset-Specific] Additional Keys
            "subj": example["subj"],
            "obj": example["obj"],
            "relation": example["relation"],
        }
        # fmt: on

    # Assertions on known quantities from VSR Zero-Shot Test Set
    assert len(index) == 1222, "Expected 1,222 unique captions/answers for VSR Zero-Shot Test Set!"
    assert len({v["img_path"] for v in index.values()}) == 715, "Expected 715 unique images for VSR Zero-Shot Test!"

    # IMPORTANT =>> Shuffle Question ID order *once* then slice into when building slim datasets
    #               This allows us to 1) have balanced images / shards for the full-scale validation dataset and
    #                                 2) have slim datasets that build off each other (enables caching / testing)
    all_qids = list(index.keys())
    Random(seed).shuffle(all_qids)  # Python `random.shuffle` is an in-place operation for... reasons...

    # Write `metadata.json` (for the complete evaluation set)
    for index_file in index_files:
        if index_file.name == "metadata-full.json":
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in all_qids}, f)

        else:
            raise ValueError(f"Received unexpected index file `{index_file}`")

    return index_files


# === Index (Metadata-Only) Dataset Declarations ===
class VSRIndexDataset(Dataset):
    def __init__(self, root_dir: Path, index_file: Path) -> None:
        """Constructs a lightweight PyTorch Dataset that loads from an index file and just returns metadata."""
        self.root_dir, self.index_file = root_dir, index_file

        # Load from `index_file` --> Dict :: example_id -> { caption / true-false / image data} --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, Path, bool]:
        """Return (example_id: int, caption: str, img_path: Path, true_false: bool) for an example."""
        ex = self.examples[idx]
        return ex["example_id"], ex["caption"], Path(self.root_dir / ex["img_path"]), ex["true_false"]

    def __len__(self) -> int:
        return len(self.examples)


# === Map/Iterable Dataset Declarations ===
class VSRMapDataset(Dataset):
    def __init__(
        self, root_dir: Path, index_file: Path, prompt_fn: Callable[[str], str], image_processor: ImageProcessor
    ) -> None:
        """
        Constructs a fully-fledged PyTorch Map-Style Dataset for evaluating on splits of the VSR Zero-Shot Test Set. In
        addition to the path to the dataset `index_file` to load from, requires a `prompt_fn` for formatting individual
        questions (model-specific), and an `image_processor` for applying any required image transforms.

        :param root_dir: Absolute path to the project's default root directory with downloads/task data
        :param prompt_fn: Callable that maps a question with the expected prompt template (model-specific)
        :param image_processor: Callable that applies the expected image transforms before yielding (model-specific)
        """
        self.prompt_fn, self.image_processor = prompt_fn, image_processor
        self.root_dir, self.index_file = root_dir, index_file

        # Load from `index_file` --> Dict :: example_id -> { caption / true-false / image data} --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, torch.Tensor, str, bool]:
        """Return (example_id: int, caption_prompt: str, pixel_values: torch.Tensor, caption: str, true_false: bool)."""
        ex = self.examples[idx]
        caption_prompt = self.prompt_fn(ex["caption"])

        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
            pixel_values = self.image_processor(Image.open(self.root_dir / ex["img_path"]).convert("RGB"))
        else:
            # Assume `image_transform` is an HF ImageProcessor...
            pixel_values = self.image_processor(
                Image.open(self.root_dir / ex["img_path"]).convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]

        return ex["example_id"], caption_prompt, pixel_values, ex["caption"], ex["true_false"]

    def __len__(self) -> int:
        return len(self.examples)


# === VSR Task Runner ===
class VSRTaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
    ) -> None:
        """Task Runner for the VSR Dataset; loads data, then runs (distributed) VLM evaluation and writes results."""
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor

        # === Unfortunate Pattern =>> Accelerate injects a lot of additional stuff into env; minimize collateral ===
        from accelerate import PartialState

        self.distributed_state = PartialState()

        # Short-Circuit (if results/metrics already exist)
        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"VSR Metrics for Model `{self.model_id}` already exist =>> Exiting!", ctx_level=1)
            return

        # Build (Map/Iterable) Dataset, using Model-Specific Prompt & Image Processor
        overwatch.info(f"Assembling VSR Map-Style Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = VSRMapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor)

    def evaluate(self, vlm: VLM, device_batch_size: int, num_workers: int) -> None:
        """Initialize Dataloader & partition data across ranks, writing metrics to disk on termination."""
        sampler = DistributedSampler(
            self.dataset,
            num_replicas=self.distributed_state.num_processes,
            rank=self.distributed_state.process_index,
            shuffle=False,
            drop_last=False,
        )
        dataloader = DataLoader(self.dataset, batch_size=device_batch_size, sampler=sampler, num_workers=num_workers)

        # Start Evaluation
        result_qa_pairs = {}
        try:
            overwatch.info(f"Distributing Evaluation across {self.distributed_state.num_processes} GPUs", ctx_level=1)
            for example_ids, caption_prompts, pixel_values, captions, true_false_answers in tqdm(
                dataloader,
                desc="=>> Evaluating",
                disable=not self.distributed_state.is_main_process,
            ):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.to(self.distributed_state.device)
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: v.to(self.distributed_state.device) for k, v in pixel_values.items()}
                else:
                    raise ValueError(f"Unexpected `pixel_values` type = {type(pixel_values)}")

                # Note =>> `gen_probabilities` returns the normalized probabilities of ["True", "False"] from LM
                gen_probabilities = vlm.generate_answer(
                    pixel_values, caption_prompts, return_string_probabilities=["True", "False"]
                )

                for example_id, gen_probs, caption, true_false_answer in zip(
                    example_ids, gen_probabilities, captions, true_false_answers, strict=True
                ):
                    ex_id = int(example_id.item())
                    result_qa_pairs[ex_id] = {
                        "example_id": ex_id,
                        "caption": caption,
                        "true_false_probabilities": gen_probs,
                        "ground_truth_answer": true_false_answer.item(),
                    }

        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_qa_pairs, f, indent=2)

        # Block on all processes before returning!
        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


# === Official Score Function =>> Just computes exact-match accuracy ===
class VSRScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_caption_tf_pairs: Dict[str, Dict],
        annotations_file: Path,
        split: str = "zeroshot-test",
        **_: str,
    ) -> None:
        """Computes exact-match True/False accuracy."""
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.split = annotations_file, split
        self.full_result_caption_tf_pairs = full_result_caption_tf_pairs

    def score(self, model_id: str) -> Dict[str, float]:
        """Run exact-match based scoring on True/False outputs, as well as ROC-AUC and PR-AUC computation."""
        n_correct, n_true, model_true_probabilities, gt_true_labels = 0, 0, [], []
        for example in self.full_result_caption_tf_pairs.values():
            tf_probabilities = example["true_false_probabilities"]

            # Compute Exact Match
            predicted_tf = [True, False][np.argmax(tf_probabilities)]
            n_correct += 1 if (predicted_tf == example["ground_truth_answer"]) else 0
            n_true += 1 if example["ground_truth_answer"] else 0

            # Extract Probabilities & GT Binary Labels for "True" (Positive Class)
            model_true_probabilities.append(tf_probabilities[0])
            gt_true_labels.append(1 if example["ground_truth_answer"] is True else 0)

        # Compute Exact Match Accuracy
        accuracy = float(n_correct / len(self.full_result_caption_tf_pairs))
        always_true_accuracy = float(n_true / len(self.full_result_caption_tf_pairs))

        # Compute AUC-ROC and AUC-PR Scores
        auc_roc = roc_auc_score(gt_true_labels, model_true_probabilities)
        auc_pr = average_precision_score(gt_true_labels, model_true_probabilities)

        # Create Metrics Dictionary & Log
        metrics = {
            "accuracy__VSR-ExactMatch": accuracy,
            "accuracy__VSR-AUCROC": auc_roc,
            "accuracy__VSR-AUCPR": auc_pr,
            "auc-roc": auc_roc,
            "auc-pr": auc_pr,
        }

        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (Split = {self.split})\n"
            f"          => Random (Always True)  : {always_true_accuracy:.3f}\n"
            f"          => Accuracy  (Official)  : {accuracy:.3f}\n"
            f"          => ROC AUC   (Official)  : {auc_roc:.3f}\n"
            f"          => PR  AUC   (Official)  : {auc_pr:.3f}"
        )

        return metrics
