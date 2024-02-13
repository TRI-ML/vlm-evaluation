"""
tallyqa.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the TallyQA visual question answering
(counting-focused) dataset. Only loads and process the "Simple" and "Complex" test splits.

Note =>> In order to get calibrated probabilities, we convert the 16 possible outputs (numbers 0 - 15) to a multiple
choice question-answering format with 16 choices (A - P).
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
def build_tallyqa_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21) -> List[Path]:
    """Parse TallyQA `test.json` --> build & write index files w/ VQA keys + additional dataset-specific data."""
    paths = DATASET_REGISTRY["tally-qa"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    # Short-Circuit (if index files have already been built)
    index_files = [dataset_dir / "metadata-full.json"] + (
        []
        if slim_dataset_sizes is None
        else [dataset_dir / f"metadata-slim-{n_slim}.json" for n_slim in slim_dataset_sizes]
    )
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    # Otherwise, load the raw questions/annotations
    with open(root_dir / paths["questions_answers"], "r") as f:
        examples = json.load(f)

    # Build Full Metadat Structure
    index = {}
    for example in tqdm(examples, desc="=> Processing TallyQA Raw Dataset:", leave=False):
        qid = int(example["question_id"])
        split = "simple" if example["issimple"] else "complex"

        # Get Image Path
        img_path = paths["images"] / example["image"]
        assert (root_dir / img_path).exists(), f"Image `{img_path}` for Question ID `{qid}` does not exist!"

        # Build Metadata Entry
        # fmt: off
        index[qid] = {
            # [Required] Tally QA Task Keys
            "question_id": qid,
            "question": example["question"],
            "img_path": str(img_path),
            "answer": int(example["answer"]),

            # Additional Metadata
            "split": split,
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
            slim_qids, counts = [], {split: 0 for split in ["simple", "complex"]}
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


# === Index (Metadata-Only) Dataset Declarations ===
class TallyQAIndexDataset(Dataset):
    def __init__(self, root_dir: Path, index_file: Path) -> None:
        """Constructs a lightweight Pytorch Dataset that loads from an index file and just returns metadata."""
        self.root_dir, self.index_file = root_dir, index_file

        # Load from `index_file` --> Dict :: qid -> { question / answer / image data } --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, Path, int]:
        """Return (question_id: int, question: str, img_path: Path, answer: int) for an example."""
        ex = self.examples[idx]
        return ex["question_id"], ex["question"], Path(self.root_dir / ex["img_path"]), ex["answer"]

    def __len__(self) -> int:
        return len(self.examples)


# === Map/Iterable Dataset Declarations ===
class TallyQAMapDataset(Dataset):
    def __init__(
        self, root_dir: Path, index_file: Path, prompt_fn: Callable[[str], str], image_processor: ImageProcessor
    ) -> None:
        """
        Constructs a fully-fledged PyTorch Map-Style Dataset for evaluating on splits of the TallyQA Test Set.
        In addition to the path to the dataset `index_file` to load from, requires a `prompt_fn` for formatting
        individual questions (model-specific), and an `image_processor` for applying any required image transforms.

        :param root_dir: Absolute path to the project's default root directory with downloads/task data
        :param prompt_fn: Callable that maps a question with the expected prompt template (model-specific)
        :param image_processor: Callable that applies the expected image transforms before yielding (model-specific)
        """
        self.prompt_fn, self.image_processor = prompt_fn, image_processor
        self.root_dir, self.index_file = root_dir, index_file

        # Load from `index_file` --> Dict :: qid -> { question / answer / image data } --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, torch.Tensor, str, int]:
        """Return (qid: int, qprompt: str, pixel_values: torch.Tensor, question: str, answer: int)."""
        ex = self.examples[idx]
        question_prompt = self.prompt_fn(ex["question"])

        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
            pixel_values = self.image_processor(Image.open(self.root_dir / ex["img_path"]).convert("RGB"))
        else:
            # Assume `image_transform` is an HF ImageProcessor...
            pixel_values = self.image_processor(
                Image.open(self.root_dir / ex["img_path"]).convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]

        return ex["question_id"], question_prompt, pixel_values, ex["question"], ex["answer"]

    def __len__(self) -> int:
        return len(self.examples)


# === TallyQA Task Runner ===
class TallyQATaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
    ) -> None:
        """Task Runner for the TallyQA Dataset; loads data then runs (distributed) VLM evaluation and writes results."""
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor

        # === Unfortunate Pattern =>> Accelerate injects a lot of additional stuff into env; minimize collateral ===
        from accelerate import PartialState

        self.distributed_state = PartialState()

        # Short-Circuit (if results/metrics already exist)
        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"TallyQA Metrics for Model `{self.model_id}` already exist =>> Existing!")
            return

        # Build (Map/Iterable) Dataset, using Model-Specific Prompt & Image Processor
        overwatch.info(f"Assembling TallyQA Map-Style Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = TallyQAMapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor)

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

        # We treat TallyQA as a Multiple Choice Task to extract probabilities =>> set "return_string_probabilities"
        return_string_probabilities = [chr(ord("A") + idx) for idx in range(16)]

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

                # Note =>> `gen_probabilities` returns the normalized probabilities of ["A", "B", ... "P"] from LM
                gen_probabilities = vlm.generate_answer(
                    pixel_values, question_prompts, return_string_probabilities=return_string_probabilities
                )

                for qid, gen_probs, question, answer in zip(
                    question_ids, gen_probabilities, questions, answers, strict=True
                ):
                    qid = int(qid.item())
                    result_qa_pairs[qid] = {
                        "question_id": qid,
                        "question": question,
                        "mc_probabilities": gen_probs,
                        "ground_truth_answer": answer.item(),
                    }

        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_qa_pairs, f, indent=2)

        # Block on all processes before returning!
        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


# === Official Score Function =>> Computes Exact-Match Accuracy ===
class TallyQAScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_qa_pairs: Dict[str, Dict],
        annotations_file: Path,
        split: str = "test",
        **_: str,
    ) -> None:
        """Computes exact-match multiple choice accuracy."""
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.split = annotations_file, split
        self.full_result_qa_pairs = full_result_qa_pairs

        with open(self.annotations_file, "r") as f:
            self.annotations = json.load(f)

    def score(self, model_id: str) -> Dict[str, float]:
        """Run exact-match based scoring on multiple choice outputs, as well as ROC-AUC and PR-AUC computation."""
        n_correct, n_total = {"simple": 0, "complex": 0}, {"simple": 0, "complex": 0}
        model_probabilities, gt_labels = {"simple": [], "complex": []}, {"simple": [], "complex": []}
        for example in self.full_result_qa_pairs.values():
            mc_probabilities = example["mc_probabilities"]
            split = self.annotations[str(example["question_id"])]["split"]

            # Compute Exact Match
            predicted_num = np.argmax(mc_probabilities)
            n_correct[split] += 1 if (predicted_num == example["ground_truth_answer"]) else 0
            n_total[split] += 1

            # Add Probabilities and Labels to Tracker
            model_probabilities[split].append(mc_probabilities)
            gt_labels[split].append(example["ground_truth_answer"])

        # Compute Exact Match Accuracies
        accuracies = {split: n_correct[split] / n_total[split] for split in n_correct}
        accuracies["final"] = sum(n_correct.values()) / sum(n_total.values())

        # Compute AUC-ROC (one-vs-one) and AUC-PR Scores
        auc_roc = {
            split: roc_auc_score(gt_labels[split], model_probabilities[split], multi_class="ovo") for split in gt_labels
        }
        auc_roc["final"] = roc_auc_score(
            sum(gt_labels.values(), []), sum(model_probabilities.values(), []), multi_class="ovo"
        )

        auc_pr = {split: average_precision_score(gt_labels[split], model_probabilities[split]) for split in gt_labels}
        auc_pr["final"] = average_precision_score(sum(gt_labels.values(), []), sum(model_probabilities.values(), []))

        # Create Metrics Dictionary & Log
        metrics = {
            "accuracy__TallyQA-simple-Accuracy": accuracies["simple"],
            "accuracy__TallyQA-simple-AUCROC": auc_roc["simple"],
            "accuracy__TallyQA-simple-AUCPR": auc_pr["simple"],
            "accuracy__TallyQA-complex-Accuracy": accuracies["complex"],
            "accuracy__TallyQA-complex-AUCROC": auc_roc["complex"],
            "accuracy__TallyQA-complex-AUCPR": auc_pr["complex"],
            "accuracy__TallyQA-final-Accuracy": accuracies["final"],
            "accuracy__TallyQA-final-AUCROC": auc_roc["final"],
            "accuracy__TallyQA-final-AUCPR": auc_pr["final"],
        }

        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (simple/complex/final) (Split = Test)\n"
            f"          => TallyQA-simple   Accuracy (Official): {metrics['accuracy__TallyQA-simple-Accuracy']:.3f}\n"
            f"          => TallyQA-simple   ROC AUC  (Official): {metrics['accuracy__TallyQA-simple-AUCROC']:.3f}\n"
            f"          => TallyQA-simple   PR  AUC  (Official): {metrics['accuracy__TallyQA-simple-AUCPR']:.3f}\n\n"
            f"          => TallyQA-complex  Accuracy (Official): {metrics['accuracy__TallyQA-complex-Accuracy']:.3f}\n"
            f"          => TallyQA-complex  ROC AUC  (Official): {metrics['accuracy__TallyQA-complex-AUCROC']:.3f}\n"
            f"          => TallyQA-complex  PR  AUC  (Official): {metrics['accuracy__TallyQA-complex-AUCPR']:.3f}\n\n"
            f"          => TallyQA-final    Accuracy (Official): {metrics['accuracy__TallyQA-final-Accuracy']:.3f}\n"
            f"          => TallyQA-final    ROC AUC  (Official): {metrics['accuracy__TallyQA-final-AUCROC']:.3f}\n"
            f"          => TallyQA-final    PR  AUC  (Official): {metrics['accuracy__TallyQA-final-AUCPR']:.3f}"
        )

        return metrics
