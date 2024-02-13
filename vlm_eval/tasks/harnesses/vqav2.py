"""
vqav2.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the VQA-v2 visual question answering
dataset. Only loads and processes the VQAv2 Validation Split (test split is hidden / leaderboard-only).

Note =>> in VQA-v2, each question is associated with 10 answers; for the purposes of this script, we use the defined
         "multiple_choice_answer" field (corresponding to the most frequent answer amongst the 10 ground-truth answers)
         for our evaluations. See the original dataset for more: https://visualqa.org/download.html
"""
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import Compose
from tqdm import tqdm

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks.registry import DATASET_REGISTRY
from vlm_eval.util.evaluation.vqav2.eval import run_vqa_evaluation
from vlm_eval.util.interfaces import VLM, ImageProcessor

# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)


# === Dataset Indexing / Building Utilities ===
def build_vqav2_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21) -> List[Path]:
    """Parse VQAv2 raw files --> build & write index files w/ necessary VQA keys + additional dataset-specific data."""
    paths = DATASET_REGISTRY["vqa-v2"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    # Short-Circuit (if index files have already been built)
    index_files = [dataset_dir / "metadata.json"] + (
        []
        if slim_dataset_sizes is None
        else [dataset_dir / f"metadata-slim-{n_slim}.json" for n_slim in slim_dataset_sizes]
    )
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    # Otherwise, load the raw questions/annotations from the downloaded VQA-v2 raw data
    with open(root_dir / paths["questions"], "r") as qfile, open(root_dir / paths["answers"]) as afile:
        question_json, answer_json = json.load(qfile), json.load(afile)

    # Build Mapping `qid --> question_dict`
    qid2question = {question["question_id"]: question for question in question_json["questions"]}

    # Build Full Metadata Structure
    index = {}
    for annotation in tqdm(answer_json["annotations"], desc="=> Processing VQA-v2 Raw Dataset:", leave=False):
        qid: int = int(annotation["question_id"])

        # Get Image Path
        img_path = paths["images"] / f"COCO_val2014_{annotation['image_id']:012d}.jpg"
        assert (root_dir / img_path).exists(), f"Image `{img_path}` for Question ID `{qid}` does not exist!"

        # Build Metadata Entry
        # fmt: off
        index[qid] = {
            # [Required] VQA Task Keys
            "question_id": qid,
            "question": qid2question[qid]["question"],
            "img_path": str(img_path),
            "answer": annotation["multiple_choice_answer"],  # Majority answer across all 10 annotators

            # [Dataset-Specific] Additional Keys
            "question_type": annotation["question_type"],
            "answer_type": annotation["answer_type"],
            "all_answers": [ans["answer"] for ans in annotation["answers"]],
        }
        # fmt: on

    # Assertions on known quantities from VQAv2 Validation Set
    assert len(index) == 214354, "Expected 214,354 unique question/answers for VQAv2 Val!"
    assert len({v["img_path"] for v in index.values()}) == 40504, "Expected 40,504 unique images for VQAv2 Val!"

    # IMPORTANT =>> Shuffle Question ID order *once* then slice into when building slim datasets
    #               This allows us to 1) have balanced images / shards for the full-scale validation dataset and
    #                                 2) have slim datasets that build off each other (enables caching / testing)
    all_qids = list(index.keys())
    Random(seed).shuffle(all_qids)  # Python `random.shuffle` is an in-place operation for... reasons...

    # Write `metadata.json` (for the complete validation set)
    for index_file in index_files:
        if index_file.name == "metadata.json":
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in all_qids}, f)

            # Dump All Questions & Annotations to `dataset_dir` in the exact same format as `question/answer.json`
            gt_question_json, gt_answer_json = deepcopy(question_json), deepcopy(answer_json)
            with open(dataset_dir / "questions-vqa-v2-full.json", "w") as qfile:
                json.dump(gt_question_json, qfile)

            with open(dataset_dir / "annotations-vqa-v2-full.json", "w") as afile:
                json.dump(gt_answer_json, afile)

        elif index_file.name.startswith("metadata-slim-"):
            n_slim = int(re.search("-slim-(.+?).json", index_file.name).group(1))
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in all_qids[:n_slim]}, f)

            # Dump Sampled Questions & Annotations to `dataset_dir` in the exact same format as `question/answer.json`
            slim_question_json, slim_answer_json = deepcopy(question_json), deepcopy(answer_json)
            slim_qids = set(all_qids[:n_slim])
            slim_question_json["questions"] = [
                q for q in slim_question_json["questions"] if q["question_id"] in slim_qids
            ]
            slim_answer_json["annotations"] = [
                a for a in slim_answer_json["annotations"] if a["question_id"] in slim_qids
            ]

            with open(dataset_dir / f"questions-vqa-v2-slim-{n_slim}.json", "w") as qfile:
                json.dump(slim_question_json, qfile)

            with open(dataset_dir / f"annotations-vqa-v2-slim-{n_slim}.json", "w") as afile:
                json.dump(slim_answer_json, afile)

        else:
            raise ValueError(f"Received unexpected index file `{index_file}`")

    return index_files


# === Index (Metadata-Only) Dataset Declarations ==
class VQAv2IndexDataset(Dataset):
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
class VQAv2MapDataset(Dataset):
    def __init__(
        self, root_dir: Path, index_file: Path, prompt_fn: Callable[[str], str], image_processor: ImageProcessor = None
    ) -> None:
        """
        Constructs a fully-fledged PyTorch Map-Style Dataset for evaluating on splits of the VQAv2 Validation Set. In
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

    def __getitem__(self, idx: int) -> Tuple[int, str, torch.Tensor, str, str]:
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

        return ex["question_id"], question_prompt, pixel_values, ex["question"], ex["answer"]

    def __len__(self) -> int:
        return len(self.examples)


# === VQAv2 Task Runner ===
class VQAv2TaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
    ) -> None:
        """Task Runner for the VQAv2 Dataset; loads data, then runs (distributed) VLM evaluation and writes results."""
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor

        # === Unfortunate Pattern =>> Accelerate injects a lot of additional stuff into env; minimize collateral ===
        from accelerate import PartialState

        self.distributed_state = PartialState()

        # Short-Circuit (if results/metrics already exist)
        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"VQA-v2 Metrics for Model `{self.model_id}` already exist =>> Exiting!", ctx_level=1)
            exit(0)

        # Build (Map/Iterable) Dataset, using Model-Specific Prompt & Image Processor
        overwatch.info(f"Assembling VQAv2 Map-Style Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = VQAv2MapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor)

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

                gen_answers = vlm.generate_answer(pixel_values, question_prompts)
                for question_id, gen_answer, question, answer in zip(
                    question_ids, gen_answers, questions, answers, strict=True
                ):
                    qid = int(question_id.item())
                    result_qa_pairs[qid] = {
                        "question_id": qid,
                        "question": question,
                        "model_output": gen_answer,
                        "ground_truth_answer": answer,
                    }

        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_qa_pairs, f, indent=2)

        # Block on all processes before returning!
        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


# === Official Score Function (Calls the lightly modified official VQA evaluation script in `util/evaluation/vqav2` ===
class VQAv2Scorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_qa_pairs: Dict[str, Dict],
        annotations_file: Path,
        questions_file: Optional[Path] = None,
        split: str = "val",
    ) -> None:
        """Wrapper around the official VQAv2 evaluation script; handles converting results to/from VQAv2 format."""
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.questions_file, self.split = annotations_file, questions_file, split
        self.full_result_qa_pairs = full_result_qa_pairs

        # Convert Results to Official GQA Format
        self.convert_results()

    def convert_results(self) -> None:
        """VQAv2 Evaluation Script expects List[{"question_id": int, "answer": str}] (normalizes automatically)."""

        # Dump Full Results to JSON (for later inspection)
        with open(self.task_results_dir / "full-results.json", "w") as f:
            json.dump(self.full_result_qa_pairs, f, indent=2)

        # Convert to VQAv2 Expected Format
        predictions = []
        for example in self.full_result_qa_pairs.values():
            predictions.append({"question_id": example["question_id"], "answer": example["model_output"]})

        # Write Predictions to Disk
        with open(self.task_results_dir / "vqa-v2-formatted-predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

    def score(self, model_id: str) -> Dict[str, float]:
        """Call wrapped functions in `vlm_eval.util.evaluation.vqav2.eval`; returns accuracy/metrics."""
        metrics = run_vqa_evaluation(
            self.questions_file, self.annotations_file, self.task_results_dir / "vqa-v2-formatted-predictions.json"
        )

        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (Split = {self.split})\n"
            f"          => Accuracy (Official): {metrics['accuracy']:.3f}"
        )

        return metrics
