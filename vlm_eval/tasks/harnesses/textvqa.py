"""
textvqa.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the Text VQA visual question answering
dataset. Only loads and processes the Text VQA Validation Split (test split is hidden / leaderboard-only).

Note =>> in Text VQA, each question is associated with 10 answers; and we use all 10 in the official scoring.
See the original dataset for more: https://textvqa.org/
"""
import json
import os
import re
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional, Tuple

import jsonlines
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import Compose
from tqdm import tqdm

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks.registry import DATASET_REGISTRY
from vlm_eval.util.evaluation.textvqa.m4c_evaluators import TextVQAAccuracyEvaluator
from vlm_eval.util.interfaces import VLM, ImageProcessor

# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)


# === Dataset Indexing / Building Utilities ===
def build_textvqa_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21) -> List[Path]:
    """Parse Text VQA raw files --> build & write index files w/ VQA keys + additional dataset-specific data."""
    paths = DATASET_REGISTRY["text-vqa"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    # Short-Circuit (if index files have already been built)
    index_files = [dataset_dir / "metadata.json"] + (
        []
        if slim_dataset_sizes is None
        else [dataset_dir / f"metadata-slim-{n_slim}.json" for n_slim in slim_dataset_sizes]
    )
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    # Otherwise, load the raw questions/annotations from the downloaded Text VQA raw data
    with jsonlines.open(root_dir / paths["questions"]) as qreader, open(root_dir / paths["answers"]) as afile:
        question_json = [blob for blob in qreader]
        qa_json = json.load(afile)["data"]

        # Both `question_json` and `qa_json` are lists with the same ordering (but random IDs); the only difference
        # between the two is that `question_json[i]["text"]` has the OCR dump. To keep things simple, we overwrite
        # `qa_json[i]["question"]` with `question_json[i]["text"]` and proceed as normal.
        assert len(question_json) == len(qa_json) == 5000, "TextVQA Val should have 5000 examples!"
        for idx in range(len(qa_json)):
            qa_json[idx]["question"] = question_json[idx]["text"]

    # Build Mapping `qid --> annotation_dict`
    qid2question = {question["question_id"]: question for question in qa_json}

    # Build Full Metadata Structure
    index = {}
    for annotation in tqdm(qa_json, desc="=> Processing Text VQA Raw Dataset:", leave=False):
        qid: int = int(annotation["question_id"])

        # Get Image Path
        img_path = paths["images"] / f"{annotation['image_id']}.jpg"
        assert (root_dir / img_path).exists(), f"Image `{img_path}` for Question ID `{qid}` does not exist!"

        # Build Metadata Entry
        # fmt: off
        index[qid] = {
            # [Required] Text VQA Task Keys
            "question_id": qid,
            "question": qid2question[qid]["question"],
            "img_path": str(img_path),

            # [Dataset-Specific] Similar to VQAv2, each question has multiple valid answers!
            "answers": annotation["answers"],
        }
        # fmt: on

    # Assertions on known quantities from Text VQA Validation Set
    assert len(index) == 5000, "Expected 5000 unique question/answers for Text VQA Val!"
    assert len({v["img_path"] for v in index.values()}) == 3166, "Expected 3166 unique images for Text VQA Val!"

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

            # Dump All Questions/Answers to `dataset_dir` in the exact same format as `paths["questions_answers"]`
            gt_qid2question = {str(qid): qid2question[qid] for qid in all_qids}
            with open(dataset_dir / "annotations-textvqa-full.json", "w") as f:
                json.dump(gt_qid2question, f)

        elif index_file.name.startswith("metadata-slim-"):
            n_slim = int(re.search("-slim-(.+?).json", index_file.name).group(1))
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in all_qids[:n_slim]}, f)

            # Dump Sampled Questions/Answers to `dataset_dir` in the exact same format as `paths["questions_answers"]`
            slim_qid2question = {str(qid): qid2question[qid] for qid in all_qids[:n_slim]}
            with open(dataset_dir / f"annotations-textvqa-slim-{n_slim}.json", "w") as f:
                json.dump(slim_qid2question, f)

        else:
            raise ValueError(f"Received unexpected index file `{index_file}`")

    return index_files


# === Index (Metadata-Only) Dataset Declarations ==
class TextVQAIndexDataset(Dataset):
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
class TextVQAMapDataset(Dataset):
    def __init__(
        self, root_dir: Path, index_file: Path, prompt_fn: Callable[[str], str], image_processor: ImageProcessor = None
    ) -> None:
        """
        Constructs a fully-fledged PyTorch Map-Style Dataset for evaluating on splits of the Text VQA Validation Set.
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

    def __getitem__(self, idx: int) -> Tuple[int, str, str, torch.Tensor, str, str]:
        """Return (qid: int, qprompt_ocr: str, qprompt_no_ocr: str, pixel_values: torch.Tensor, q: str, ans: str)."""
        ex = self.examples[idx]
        qprompt_ocr = self.prompt_fn(ex["question"])
        qprompt_no_ocr = self.prompt_fn(ex["question"].split("\nReference OCR token:")[0])

        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
            pixel_values = self.image_processor(Image.open(self.root_dir / ex["img_path"]).convert("RGB"))

        else:
            # Assume `image_transform` is an HF ImageProcessor...
            pixel_values = self.image_processor(
                Image.open(self.root_dir / ex["img_path"]).convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]

        return ex["question_id"], qprompt_ocr, qprompt_no_ocr, pixel_values, ex["question"], " or ".join(ex["answers"])

    def __len__(self) -> int:
        return len(self.examples)


# === Text VQA Task Runner ===
class TextVQATaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
    ) -> None:
        """
        Task Runner for the Text VQA Dataset;
        loads data, then runs (distributed) VLM evaluation and writes results.
        """
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor

        # === Unfortunate Pattern =>> Accelerate injects a lot of additional stuff into env; minimize collateral ===
        from accelerate import PartialState

        self.distributed_state = PartialState()

        # Short-Circuit (if results/metrics already exist)
        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"Text VQA Metrics for Model `{self.model_id}` already exist =>> Exiting!", ctx_level=1)
            return

        # Build (Map/Iterable) Dataset, using Model-Specific Prompt & Image Processor
        overwatch.info(f"Assembling Text VQA Map-Style Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = TextVQAMapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor)

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
            for question_ids, qprompts_ocr, qprompts_no_ocr, pixel_values, questions, answers in tqdm(
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

                gen_answers_ocr = vlm.generate_answer(pixel_values, qprompts_ocr)
                gen_answers_no_ocr = vlm.generate_answer(pixel_values, qprompts_no_ocr)

                for question_id, gen_answer_ocr, gen_answer_no_ocr, question, answer in zip(
                    question_ids, gen_answers_ocr, gen_answers_no_ocr, questions, answers, strict=True
                ):
                    qid = int(question_id.item())
                    result_qa_pairs[qid] = {
                        "question_id": qid,
                        "question": question,
                        "model_output_ocr": gen_answer_ocr,
                        "model_output_no_ocr": gen_answer_no_ocr,
                        "ground_truth_answer": answer,
                    }

        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_qa_pairs, f, indent=2)

        # Block on all processes before returning!
        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


# === Official Score Function (Calls the lightly modified official VQA evaluation script in `util/evaluation/vqatext` ===
class TextVQAScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_qa_pairs: Dict[str, Dict],
        annotations_file: Path,
        questions_file: Optional[Path] = None,
        split: str = "val",
    ) -> None:
        """Wrapper around the official Text VQA evaluation script; handles converting results to/from Text VQA format."""
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.questions_file, self.split = annotations_file, questions_file, split
        self.full_result_qa_pairs = full_result_qa_pairs

        # Convert Results to Official Text VQA Format
        self.convert_results(with_ocr_tokens=True)
        self.convert_results(with_ocr_tokens=False)

    def convert_results(self, with_ocr_tokens: bool = True) -> None:
        """TextVQA Evaluation Script expects List[{"pred_answer": str, "gt_answers": List[str] (lower, no punkt)}]."""

        # Dump Full Results to JSON (for later inspection)
        with open(self.task_results_dir / "full-results.json", "w") as f:
            json.dump(self.full_result_qa_pairs, f, indent=2)

        with open(self.annotations_file, "r") as f:
            annotations = json.load(f)

        # Convert to Text VQA Expected Format --> with answer formatting (strip punctuation & lowercase)
        predictions = []
        for example in self.full_result_qa_pairs.values():
            qid = example["question_id"]
            answers = annotations[str(qid)]["answers"]
            if with_ocr_tokens:
                predictions.append({"pred_answer": example["model_output_ocr"], "gt_answers": answers})
            else:
                predictions.append({"pred_answer": example["model_output_no_ocr"], "gt_answers": answers})

        # Write Predictions to Disk
        suffix = "with-ocr" if with_ocr_tokens else "without-ocr"
        with open(self.task_results_dir / f"text-vqa-formatted-predictions-{suffix}.json", "w") as f:
            json.dump(predictions, f, indent=2)

    def score(self, model_id: str) -> Dict[str, float]:
        """Call wrapped functions in `vlm_eval.util.evaluation.textvqa.eval`; returns accuracy/metrics."""
        with open(self.task_results_dir / "text-vqa-formatted-predictions-with-ocr.json", "r") as f:
            pred_list_with_ocr = json.load(f)

        with open(self.task_results_dir / "text-vqa-formatted-predictions-without-ocr.json", "r") as f:
            pred_list_without_ocr = json.load(f)

        evaluator_ocr, evaluator_no_ocr = TextVQAAccuracyEvaluator(), TextVQAAccuracyEvaluator()
        accuracy_ocr = evaluator_ocr.eval_pred_list(pred_list_with_ocr)
        accuracy_no_ocr = evaluator_no_ocr.eval_pred_list(pred_list_without_ocr)

        metrics = {"accuracy__TextVQA-OCR": accuracy_ocr, "accuracy__TextVQA-Pure": accuracy_no_ocr}

        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (Split = {self.split})\n"
            f"          => TextVQA-OCR  Accuracy (Official): {metrics['accuracy__TextVQA-OCR']:.3f}\n"
            f"          => TextVQA-Pure Accuracy (Official): {metrics['accuracy__TextVQA-Pure']:.3f}"
        )

        return metrics
