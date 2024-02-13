"""
vizwiz.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the VizWiz visual question answering
dataset. Only loads and processes the VizWiz val set (`val.json`) -- the default VizWiz validation split.
"""
import json
import os
import re
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
from vlm_eval.util.evaluation.vizwiz.eval import VQAEval
from vlm_eval.util.interfaces import VLM, ImageProcessor

# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)


# === Dataset Indexing / Building Utilities ===
def build_vizwiz_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21) -> List[Path]:
    """Parse VizWiz --> build & write index files w/ necessary VQA keys + additional dataset-specific data."""
    paths = DATASET_REGISTRY["vizwiz"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    # Short-Circuit (if index files have already been built)
    index_files = [dataset_dir / "metadata.json"] + (
        []
        if slim_dataset_sizes is None
        else [dataset_dir / f"metadata-slim-{n_slim}.json" for n_slim in slim_dataset_sizes]
    )
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    # Otherwise, load the raw annotations (questions & answers) from the downloaded VizWiz raw data
    with open(root_dir / paths["questions_answers"], "r") as f:
        qid2question = json.load(f)

    # Build Full Metadata Structure
    index = {}
    for question_id, example in tqdm(enumerate(qid2question), desc="=> Processing VizWiz Raw Dataset:", leave=False):
        qid: int = int(question_id)

        # Get Image Path
        img_path = paths["images"] / f"{example['image']}"
        assert (root_dir / img_path).exists(), f"Image `{img_path}` for Question ID `{qid}` does not exist!"

        # Build Metadata Entry
        # fmt: off
        index[qid] = {
            # [Required] VQA Task Keys
            "question_id": qid,
            "question": example["question"],
            "img_path": str(img_path),
            "answers": [example["answers"]],

            # [Dataset-Specific] Additional Keys
            "answerable": int(example["answerable"]),
        }
        # fmt: on

    # Assertions on known quantities from VizWiz Val Set
    assert len(index) == 4319, "Expected 4319 unique questions/answer lists for VizWiz Val!"
    assert len({v["img_path"] for v in index.values()}) == 4319, "Expected 4319 unique images for VizWiz Val!"

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
            gt_questions = {str(qid): qid2question[qid]["question"] for qid in all_qids}
            gt_answers = {str(qid): qid2question[qid]["answers"] for qid in all_qids}

            with open(dataset_dir / "questions-vizwiz.json", "w") as f:
                json.dump(gt_questions, f)

            with open(dataset_dir / "annotations-vizwiz.json", "w") as f:
                json.dump(gt_answers, f)

        elif index_file.name.startswith("metadata-slim-"):
            n_slim = int(re.search("-slim-(.+?).json", index_file.name).group(1))
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in all_qids[:n_slim]}, f)

            # Dump Sampled Questions/Answers to `dataset_dir` in the exact same format as `paths["questions_answers"]`
            slim_gt_questions = {str(qid): qid2question[qid]["question"] for qid in all_qids[:n_slim]}
            slim_gt_answers = {str(qid): qid2question[qid]["answers"] for qid in all_qids[:n_slim]}

            with open(dataset_dir / f"questions-vizwiz-slim-{n_slim}.json", "w") as f:
                json.dump(slim_gt_questions, f)

            with open(dataset_dir / f"annotations-vizwiz-slim-{n_slim}.json", "w") as f:
                json.dump(slim_gt_answers, f)

        else:
            raise ValueError(f"Received unexpected index file `{index_file}`")

    return index_files


# === Index (Metadata-Only) Dataset Declarations ==
class VizWizIndexDataset(Dataset):
    def __init__(self, root_dir: Path, index_file: Path) -> None:
        """Constructs a lightweight PyTorch Dataset that loads from an index file and just returns metadata."""
        self.root_dir, self.index_file = root_dir, index_file

        # Load from `index_file` --> Dict :: qid -> { question / answer / image data } --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, Path, str]:
        """Return (question_id: int, question: int, img_path: Path, answer: str) for an example."""
        ex = self.examples[idx]
        return (
            ex["question_id"],
            ex["question"],
            Path(self.root_dir / ex["img_path"]),
            ex["answer"],
            ex["answerable"],
        )

    def __len__(self) -> int:
        return len(self.examples)


# === Map/Iterable Dataset Declarations ===
class VizWizMapDataset(Dataset):
    def __init__(
        self, root_dir: Path, index_file: Path, prompt_fn: Callable[[str], str], image_processor: ImageProcessor = None
    ) -> None:
        """
        Constructs a fully-fledged PyTorch Map-Style Dataset for evaluating on splits of the VizWiz Val Set. In
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
            ex["answers"],
            ex["answerable"],
        )

    def __len__(self) -> int:
        return len(self.examples)


# === VizWiz Task Runner ===
class VizWizTaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
    ) -> None:
        """Task Runner for the VizWiz Dataset; loads data, then runs (distributed) VLM evaluation and writes results."""
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor

        # === Unfortunate Pattern =>> Accelerate injects a lot of additional stuff into env; minimize collateral ===
        from accelerate import PartialState

        self.distributed_state = PartialState()

        # Short-Circuit (if results/metrics already exist)
        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"VizWiz Metrics for Model `{self.model_id}` already exist =>> Exiting!", ctx_level=1)
            return

        # Build (Map/Iterable) Dataset, using Model-Specific Prompt & Image Processor
        overwatch.info(f"Assembling VizWiz Map-Style Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = VizWizMapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor)

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
            for question_ids, question_prompts, pixel_values, questions, answers, answerables in tqdm(
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

                for question_id, gen_answer, question, answer, answerable in zip(
                    question_ids, gen_answers, questions, answers, answerables, strict=True
                ):
                    qid = int(question_id.item())
                    answerable = int(answerable.item())
                    answer = [{k: v[0] for k, v in a.items()} for a in answer]
                    result_qa_pairs[qid] = {
                        "question_id": qid,
                        "question": question,
                        "model_output": gen_answer,
                        "ground_truth_answer": answer,
                        "answerable": answerable,
                    }

        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_qa_pairs, f, indent=2)

        # Block on all processes before returning!
        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


# === Official Score Function (Calls the lightly modified VizWiz evaluation script in `util/evaluation/vizwiz` ===
class VizWizScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_qa_pairs: Dict[str, Dict],
        annotations_file: Path,
        questions_file: Optional[Path] = None,
        split: str = "val",
    ) -> None:
        """Wrapper around the official VizWiz evaluation script; handles converting results to/from VizWiz format."""
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.questions_file, self.split = annotations_file, questions_file, split
        self.full_result_qa_pairs = full_result_qa_pairs

        # Convert Results to Official VizWiz Format
        self.convert_results()

    def convert_results(self) -> None:
        res = {
            k: {key: v[key] for key in ["question_id", "question", "model_output", "answerable"]}
            for k, v in self.full_result_qa_pairs.items()
        }
        res_key_mapping = {"question_id": "image", "question": "question", "model_output": "answer"}
        gts = {
            k: {key: v[key] for key in ["question_id", "question", "ground_truth_answer", "answerable"]}
            for k, v in self.full_result_qa_pairs.items()
        }
        gt_key_mapping = {
            "question_id": "image",
            "question": "question",
            "ground_truth_answer": "answers",
            "answerable": "answerable",
        }
        self.gts = {k: {gt_key_mapping.get(key, key): value for key, value in v.items()} for k, v in gts.items()}
        self.res = {k: {res_key_mapping.get(key, key): value for key, value in v.items()} for k, v in res.items()}

    def score(self, model_id: str) -> Dict[str, float]:
        # Evaluate VQA results for VizWiz using official script
        vqaEval = VQAEval(self.gts, self.res, n=2)
        vqaEval.evaluate()
        vqaEval.evaluate_unanswerability()

        unanswerable_idxs = [idx for idx in self.gts if self.gts[idx]["answerable"] == 0]
        answerable_idxs = [idx for idx in self.gts if self.gts[idx]["answerable"] == 1]
        unanswerable_gts = {key: self.gts[key] for key in unanswerable_idxs if key in self.gts}
        unanswerable_res = {key: self.res[key] for key in unanswerable_idxs if key in self.res}
        answerable_gts = {key: self.gts[key] for key in answerable_idxs if key in self.gts}
        answerable_res = {key: self.res[key] for key in answerable_idxs if key in self.res}

        vqaEvalUnanswerable = VQAEval(unanswerable_gts, unanswerable_res, n=2)
        vqaEvalUnanswerable.evaluate()
        vqaEvalAnswerable = VQAEval(answerable_gts, answerable_res, n=2)
        vqaEvalAnswerable.evaluate()

        # Create Metrics Dictionary & Log
        overall_accuracy = vqaEval.accuracy["overall"]
        unanswerability_avg_precision = vqaEval.unanswerability["average_precision"]
        unanswerability_f1_score = vqaEval.unanswerability["f1_score"]
        unanswerable_accuracy = vqaEvalUnanswerable.accuracy["overall"]
        answerable_accuracy = vqaEvalAnswerable.accuracy["overall"]

        accuracies = {
            "accuracy__VizWiz-Overall": overall_accuracy,
            "accuracy__VizWiz-Answerable": answerable_accuracy,
            "accuracy__VizWiz-Unanswerable": unanswerable_accuracy,
            "accuracy__VizWiz-Unanswerable-AvgPR": unanswerability_avg_precision,
            "accuracy__VizWiz-Unanswerable-F1": unanswerability_f1_score,
        }

        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (Split = {self.split})\n"
            f"   => VizWiz Overall      Accuracy (Official): {accuracies['accuracy__VizWiz-Overall']:.3f}\n"
            f"   => VizWiz Answerable   Accuracy (Official): {accuracies['accuracy__VizWiz-Answerable']:.3f}\n"
            f"   => VizWiz Unanswerable Accuracy (Official): {accuracies['accuracy__VizWiz-Unanswerable']:.3f}\n"
            f"   => Unanswerable Avg Precision   (Official): {accuracies['accuracy__VizWiz-Unanswerable-AvgPR']:.3f}\n"
            f"   => Unanswerable F1 Score        (Official): {accuracies['accuracy__VizWiz-Unanswerable-F1']:.3f}\n"
        )

        return accuracies
