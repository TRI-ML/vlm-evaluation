"""
pope.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the Pope visual question answering
dataset. Loads and processes all three of the POPE COCO evaluation splits -- `adversarial` | `popular` | `random` --
the default POPE evaluation.
"""
import json
import os
import re
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional, Tuple

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


# ruff: noqa: C901
def build_pope_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21) -> List[Path]:
    """Parse Pope --> build & write index files w/ necessary VQA keys + additional dataset-specific data."""
    paths = DATASET_REGISTRY["pope"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    # Short-Circuit (if index files have already been built)
    index_files = [dataset_dir / "metadata-full.json"] + (
        []
        if slim_dataset_sizes is None
        else [dataset_dir / f"metadata-slim-{n_slim}.json" for n_slim in slim_dataset_sizes]
    )
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    # Otherwise, load the raw annotations for each of the POPE splits
    split_qid2question = {"adversarial": [], "popular": [], "random": []}
    for k in split_qid2question:
        with open(root_dir / paths[f"qa_{k}"], "r") as f:
            split_qid2question[k] = [json.loads(line) for line in f]

    # Build Full Metadata Structure
    index = {}
    percent_lost_acceptable = 0.05
    for split, qid2question in split_qid2question.items():
        assert len(qid2question) > (1 - percent_lost_acceptable) * (
            count := {"adversarial": 3000, "popular": 3000, "random": 2910}[split]
        ), f"Expected {count} examples in POPE `{split}` Split!"

        for _question_id, example in tqdm(
            enumerate(qid2question), desc=f"=> Processing POPE {split} Split:", total=len(qid2question)
        ):
            # Question IDs overlap across splits, so we're going to hash!
            qid: int = hash(f"{split}-{example['question_id']}")
            assert qid not in index, "Hash collision -- do something smarter!"

            # Get Image Path
            img_path = paths["images"] / f"{example['image']}"
            assert (root_dir / img_path).exists(), f"Image `{img_path}` for Split `{split}` does not exist!"

            # Build Metadata Entry
            # fmt: off
            index[qid] = {
                # [Required] POPE VQA (Yes/No) Task Keys
                "question_id": qid,
                "question": example["text"],
                "img_path": str(img_path),
                "answer": example["label"],

                # Additional Metadata
                "split": split,
                "split_question_id": example["question_id"],
            }
            # fmt: on

    # IMPORTANT =>> Shuffle Question ID order *once* then slice into when building slim datasets
    #               This allows us to 1) have balanced images / shards for the full-scale validation dataset and
    #                                 2) have slim datasets that build off each other (enables caching / testing)
    all_qids = list(index.keys())
    Random(seed).shuffle(all_qids)  # Python `random.shuffle` is an in-place operation for... reasons...

    # Write `metadata.json` (for the complete validation set)
    for index_file in index_files:
        if index_file.name == "metadata-full.json":
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in all_qids}, f)

        elif index_file.name.startswith("metadata-slim-"):
            n_slim = int(re.search("-slim-(.+?).json", index_file.name).group(1))

            # Take the first `n_slim` examples per `split` in all_qids
            slim_qids, counts = [], {split: 0 for split in split_qid2question}
            for qid in all_qids:
                split = index[qid]["split"]
                if counts[split] < n_slim:
                    slim_qids.append(qid)
                    counts[split] += 1

                # Termination Condition
                if all([c == n_slim for c in counts.values()]):
                    break

            # Dump Sampled Examples
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in slim_qids}, f)

        else:
            raise ValueError(f"Received unexpected index file `{index_file}`")

    return index_files


# === Index (Metadata-Only) Dataset Declarations ==
class PopeIndexDataset(Dataset):
    def __init__(self, root_dir: Path, index_file: Path) -> None:
        """Constructs a lightweight PyTorch Dataset that loads from an index file and just returns metadata."""
        self.root_dir, self.index_file = root_dir, index_file

        # Load from `index_file` --> Dict :: qid -> { question / answer / image data } --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, Path, str]:
        """Return (question_id: int, question: int, img_path: Path, answer: str) for an example."""
        ex = self.examples[idx]
        return ex["question_id"], ex["question"], Path(self.root_dir / ex["img_path"]), ex["answer"]

    def __len__(self) -> int:
        return len(self.examples)


# === Map/Iterable Dataset Declarations ===
class PopeMapDataset(Dataset):
    def __init__(
        self, root_dir: Path, index_file: Path, prompt_fn: Callable[[str], str], image_processor: ImageProcessor = None
    ) -> None:
        """
        Constructs a fully-fledged PyTorch Map-Style Dataset for evaluating on splits of the Pope Val Set. In
        addition to the path to the dataset `index_file` to load from, requires a `prompt_fn` for formatting individual
        questions (model-specific), and an `image_processor` for applying any required image transforms.

        :param root_dir: Absolute path to the project's default root directory with downloads/task data
        :param prompt_fn: Callable that maps a question with the expected prompt template (model-specific)
        :param image_processor: Callable that applies the expected image transforms before yielding (model-specific)
        """
        self.prompt_fn, self.image_processor = prompt_fn, image_processor
        self.root_dir, self.index_file = root_dir, index_file

        # Load from `index_file` --> Dict :: qid -> { question / answer / image data } --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, torch.Tensor, str, str, int]:
        """Return (question_id: int, question_prompt: str, pixel_values: torch.Tensor, question: str, answer: str)."""
        ex = self.examples[idx]
        question_prompt = self.prompt_fn(ex["question"])

        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
            pixel_values = self.image_processor(Image.open(self.root_dir / ex["img_path"]).convert("RGB"))
        else:
            # Assume `image_transform` is a HF ImageProcessor...
            pixel_values = self.image_processor(
                Image.open(self.root_dir / ex["img_path"]).convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]

        return (
            ex["question_id"],
            question_prompt,
            pixel_values,
            ex["question"],
            ex["answer"],
        )

    def __len__(self) -> int:
        return len(self.examples)


# === Pope Task Runner ===
class PopeTaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
    ) -> None:
        """Task Runner for the Pope Dataset; loads data, then runs (distributed) VLM evaluation and writes results."""
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor

        # === Unfortunate Pattern =>> Accelerate injects a lot of additional stuff into env; minimize collateral ===
        from accelerate import PartialState

        self.distributed_state = PartialState()

        # Short-Circuit (if results/metrics already exist)
        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"Pope Metrics for Model `{self.model_id}` already exist =>> Exiting!", ctx_level=1)
            return

        # Build (Map/Iterable) Dataset, using Model-Specific Prompt & Image Processor
        overwatch.info(f"Assembling Pope Map-Style Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = PopeMapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor)

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
            for question_ids, question_prompts, pixel_values, questions, answers in tqdm(
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

                # Note =>> `gen_probabilities` returns the normalized probabilities of ["Yes", "No"] from LM
                gen_probabilities = vlm.generate_answer(
                    pixel_values, question_prompts, return_string_probabilities=["Yes", "No"]
                )

                for question_id, gen_probs, question, answer in zip(
                    question_ids, gen_probabilities, questions, answers, strict=True
                ):
                    qid = int(question_id.item())
                    result_qa_pairs[qid] = {
                        "question_id": qid,
                        "question": question,
                        "yes_no_probabilities": gen_probs,
                        "ground_truth_answer": answer,
                    }

        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_qa_pairs, f, indent=2)

        # Block on all processes before returning!
        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


# === Official Score Function (Calls the lightly modified Pope evaluation script in `util/evaluation/pope` ===
class PopeScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_qa_pairs: Dict[str, Dict],
        annotations_file: Path,
        split: str = "eval",
        **_: str,
    ) -> None:
        """Wrapper around the official Pope evaluation script; handles converting results to/from Pope format."""
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.split = annotations_file, split
        self.full_result_qa_pairs = full_result_qa_pairs

        # Load Annotations File to Get Split Information
        with open(self.annotations_file, "r") as f:
            self.annotations = json.load(f)

        # Factor Results by Split
        self.split2probs, self.split2labels = self.convert_results()

    def convert_results(self) -> Tuple[List[Dict[str, List[float]]], List[str]]:
        split2probs = {"adversarial": [], "popular": [], "random": []}
        split2labels = {"adversarial": [], "popular": [], "random": []}
        for qid, example in self.full_result_qa_pairs.items():
            split = self.annotations[qid]["split"]
            split2probs[split].append(example["yes_no_probabilities"])
            split2labels[split].append(example["ground_truth_answer"])

        return split2probs, split2labels

    def score(self, model_id: str) -> Dict[str, float]:
        """Evaluate Yes/No VQA Results for each POPE split, returning aggregate/split accuracies."""
        metrics, split_accuracies, all_model_yes_probabilities, all_gt_yes_labels = {}, [], [], []
        for split in self.split2probs:
            n_correct, model_yes_probabilities, gt_yes_labels = 0, [], []
            for idx in range(len(self.split2probs[split])):
                yes_no_probabilities, gt_label = self.split2probs[split][idx], self.split2labels[split][idx]
                assert gt_label in {"yes", "no"}, "Malformed label!"

                # Compute Exact Match
                predicted_yes_no = ["yes", "no"][np.argmax(yes_no_probabilities)]
                n_correct += 1 if (predicted_yes_no == gt_label) else 0

                # Extract Probabilities & GT Binary Labels for "Yes" (Positive Class)
                model_yes_probabilities.append(yes_no_probabilities[0])
                gt_yes_labels.append(1 if (gt_label == "yes") else 0)

            # Compute Split Metrics
            metrics[f"POPE-{split}-Accuracy"] = float(n_correct / len(self.split2probs[split]))
            metrics[f"POPE-{split}-AUCROC"] = roc_auc_score(gt_yes_labels, model_yes_probabilities)
            metrics[f"POPE-{split}-AUCPR"] = average_precision_score(gt_yes_labels, model_yes_probabilities)

            # Append to Global Trackers
            split_accuracies.append(float(n_correct / len(self.split2probs[split])))
            all_model_yes_probabilities.extend(model_yes_probabilities)
            all_gt_yes_labels.extend(gt_yes_labels)

        # Compute Global Metrics
        metrics["POPE-final-Accuracy"] = np.mean(split_accuracies)
        metrics["POPE-final-AUCROC"] = roc_auc_score(all_gt_yes_labels, all_model_yes_probabilities)
        metrics["POPE-final-AUCPR"] = average_precision_score(all_gt_yes_labels, all_model_yes_probabilities)

        # Create Metrics Dictionary & Log
        metrics.update({f"accuracy__{k}": v for k, v in metrics.items()})

        # Create Metrics Dictionary & Log
        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (adversarial/popular/random) (Split = Eval)\n"
            f"          => POPE-adversarial  Accuracy (Official): {metrics['POPE-adversarial-Accuracy']:.3f}\n"
            f"          => POPE-adversarial  ROC AUC  (Official): {metrics['POPE-adversarial-AUCROC']:.3f}\n"
            f"          => POPE-adversarial  PR  AUC  (Official): {metrics['POPE-adversarial-AUCPR']:.3f}\n\n"
            f"          => POPE-popular      Accuracy (Official): {metrics['POPE-popular-Accuracy']:.3f}\n"
            f"          => POPE-popular      ROC AUC  (Official): {metrics['POPE-popular-AUCROC']:.3f}\n"
            f"          => POPE-popular      PR  AUC  (Official): {metrics['POPE-popular-AUCPR']:.3f}\n\n"
            f"          => POPE-random       Accuracy (Official): {metrics['POPE-random-Accuracy']:.3f}\n"
            f"          => POPE-random       ROC AUC  (Official): {metrics['POPE-random-AUCROC']:.3f}\n"
            f"          => POPE-random       PR  AUC  (Official): {metrics['POPE-random-AUCPR']:.3f}\n\n"
            f"          => POPE-final        Accuracy (Official): {metrics['POPE-final-Accuracy']:.3f}\n"
            f"          => POPE-final        ROC AUC  (Official): {metrics['POPE-final-AUCROC']:.3f}\n"
            f"          => POPE-final        PR  AUC  (Official): {metrics['POPE-final-AUCPR']:.3f}"
        )

        return metrics
