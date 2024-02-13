"""
ocidref.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the OCID-Ref Minimum/Medium/Maximum
Clutter referring expression grounding (bounding box prediction) datasets. Only loads & processes the *validation sets*
-- the various test splits are left alone (for our evaluation).
"""
import ast
import json
import os
import re
from bisect import bisect_left
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import Compose
from tqdm import tqdm

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks.registry import DATASET_REGISTRY
from vlm_eval.util.interfaces import VLM, ImageProcessor

# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)


# === Define Clutter Split and Mappings (from OCID) ===
#   The `take_id` corresponds to the index of images taken on a given table-top over time, with the table
#   getting more cluttered over time. There are three splits that OCID-REF uses:
#   - "free" [split = 0] --> clearly separated objects, indices [0 --> 9] (inclusive)
#   - "touching" [split = 1] --> touching objects (moderate clutter), indices [10 --> 16] (inclusive)
#   - "stacked" [split = 2] --> stacked, touching objects (max clutter), indices [17 --> 20] (inclusive)
#
# => Ref: https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/
def get_split(take_id: int) -> str:
    assert 0 <= take_id <= 20, "Bounds violation - `take_id` must be in [0, 20]!"
    return {0: "Minimum Clutter", 1: "Medium Clutter", 2: "Maximum Clutter"}[bisect_left([9, 16, 20], take_id)]


# === Bounding Box Utilities ===
def box_xyxy2normalized(bbox_xyxy: List[int], img_wh: Tuple[int, int]) -> List[float]:
    width, height = img_wh

    # Validate
    assert bbox_xyxy[0] < bbox_xyxy[2] <= width, "Invalid BBox Width!"
    assert bbox_xyxy[1] < bbox_xyxy[3] <= height, "Invalid BBox Height!"

    # Handle Normalization
    bbox_xyxy = [bbox_xyxy[0] / width, bbox_xyxy[1] / height, bbox_xyxy[2] / width, bbox_xyxy[3] / height]

    # Return Box Coordinates rounded to 2 decimal places!
    return [round(coord, 2) for coord in bbox_xyxy]


# === Dataset Indexing / Building Utilities ===
def build_ocidref_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21) -> List[Path]:
    """Parse OCID-Ref validation sets --> build & write index files w/ necessary keys + additional metadata."""
    paths = DATASET_REGISTRY["ocid-ref"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    # Short-Circuit (if index files have already been built)
    index_files = [dataset_dir / "metadata-full.json"] + (
        []
        if slim_dataset_sizes is None
        else [dataset_dir / f"metadata-slim-{n_slim}.json" for n_slim in slim_dataset_sizes]
    )
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    # Otherwise, load the raw val annotations (expressions & bounding boxes) from the downloaded OCID-Ref data
    with open(root_dir / paths["referring_expressions"], "r") as f:
        id2expressions = json.load(f)

    # All OCID Images are 640 x 480 Pixels!
    image_size_wh = (640, 480)

    # Build Full Metadata Structure
    index = {}
    for example_id, example in tqdm(id2expressions.items(), desc="=> Processing OCID-Ref Raw Dataset:", leave=False):
        ex_id = int(example_id)

        # Get Image Path & Full Bounding Box as [x, y, x, y]
        img_path = paths["images"] / example["scene_path"]
        assert (root_dir / img_path).exists(), f"Image `{img_path}` for Example ID `{ex_id}` does not exist!"
        bbox_xyxy = json.loads(example["bbox"])

        # Compute Normalized Bounding Box
        normalized_box_xyxy = box_xyxy2normalized(bbox_xyxy, image_size_wh)

        # Grab Referring Expression
        ref_expression = example["sentence"]

        # Compute Split
        clutter_split = get_split(example["take_id"])

        # Build Metadata Entry
        # fmt: off
        index[ex_id] = {
            # [Required] OCID-Ref Localization Task Keys
            "example_id": ex_id,
            "ref_expression": ref_expression,
            "img_path": str(img_path),
            "bbox": normalized_box_xyxy,

            # Additional Metadata
            "clutter_split": clutter_split,
            "split": "val",
        }
        # fmt: on

    # IMPORTANT =>> Shuffle Example ID order *once* then slice into when building slim datasets
    #               This allows us to 1) have balanced images / shards for the full-scale validation dataset and
    #                                 2) have slim datasets that build off each other (enables caching / testing)
    all_ex_ids = list(index.keys())
    Random(seed).shuffle(all_ex_ids)  # Python `random.shuffle` is an in-place operation for... reasons...

    # Write `metadata.json` (for the complete evaluation set)
    for index_file in index_files:
        if index_file.name == "metadata-full.json":
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in all_ex_ids}, f)

        elif index_file.name.startswith("metadata-slim-"):
            n_slim = int(re.search("-slim-(.+?).json", index_file.name).group(1))

            # Take the first `n_slim` examples per `refer_dataset` in all_qids
            slim_ex_ids, counts = [], {"Minimum Clutter": 0, "Medium Clutter": 0, "Maximum Clutter": 0}
            for ex_id in all_ex_ids:
                clutter_split = index[ex_id]["clutter_split"]
                if counts[clutter_split] < n_slim:
                    slim_ex_ids.append(ex_id)
                    counts[clutter_split] += 1

                # Termination Condition
                if all([c == n_slim for c in counts.values()]):
                    break

            # Dump Sampled Examples
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in slim_ex_ids}, f)

    return index_files


# === Index (Metadata-Only) Dataset Declarations ===
class OCIDRefIndexDataset(Dataset):
    def __init__(self, root_dir: Path, index_file: Path) -> None:
        """Constructs a lightweight PyTorch Dataset that loads from an index file and just returns metadata."""
        self.root_dir, self.index_file = root_dir, index_file

        # Load from `index_file` --> Dict :: example_id -> { ref_expr / bbox / image data } --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, Path, List[float]]:
        """Return (example_id: int, ref_expression: str, img_path: Path, bbox: List[float]) for an example."""
        ex = self.examples[idx]
        return ex["example_id"], ex["ref_expression"], Path(self.root_dir / ex["img_path"]), np.asarray(ex["bbox"])

    def __len__(self) -> int:
        return len(self.examples)


# === Map/Iterable Dataset Declarations ===
class OCIDRefMapDataset(Dataset):
    def __init__(
        self, root_dir: Path, index_file: Path, prompt_fn: Callable[[str], str], image_processor: ImageProcessor
    ) -> None:
        """
        Constructs a fully-fledged PyTorch Map-Style Dataset for evaluating on splits of the OCID-Ref Validation Sets.
        In addition to the path to the dataset `index_file` to load from, requires a `prompt_fn` for formatting
        individual expressions (model-specific), and an `image_processor` for applying any required image transforms.

        :param root_dir: Absolute path to the project's default root directory with downloads/task data
        :param prompt_fn: Callable that maps a question with the expected prompt template (model-specific)
        :param image_processor: Callable that applies the expected image transforms before yielding (model-specific)
        """
        self.prompt_fn, self.image_processor = prompt_fn, image_processor
        self.root_dir, self.index_file = root_dir, index_file

        # Load from `index_file` --> Dict :: example_id -> { ref_expr / bbox / image data } --> flatten
        with open(self.root_dir / self.index_file, "r") as f:
            self.examples = list(json.load(f).values())

    def __getitem__(self, idx: int) -> Tuple[int, str, torch.Tensor, str, List[float]]:
        """Return (example_id: int, ref_expr_prompt: str, pixel_values: Tensor, ref_expr: str, bbox: List[float])."""
        ex = self.examples[idx]
        ref_expr_prompt = self.prompt_fn(ex["ref_expression"])

        if isinstance(self.image_processor, Compose) or hasattr(self.image_processor, "is_prismatic"):
            # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
            pixel_values = self.image_processor(Image.open(self.root_dir / ex["img_path"]).convert("RGB"))
        else:
            # Assume `image_transform` is an HF ImageProcessor...
            pixel_values = self.image_processor(
                Image.open(self.root_dir / ex["img_path"]).convert("RGB"), return_tensors="pt"
            )["pixel_values"][0]

        return ex["example_id"], ref_expr_prompt, pixel_values, ex["ref_expression"], np.asarray(ex["bbox"])

    def __len__(self) -> int:
        return len(self.examples)


# === OCIDRef Task Runner ===
class OCIDRefTaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
    ) -> None:
        """Task Runner for the OCID-Ref Dataset; loads data, then runs (distributed) VLM evaluation & writes results."""
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor

        # === Unfortunate Pattern =>> Accelerate injects a lot of additional stuff into env; minimize collateral ===
        from accelerate import PartialState

        self.distributed_state = PartialState()

        # Short-Circuit (if results/metrics already exist)
        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"OCID-Ref Metrics for Model `{self.model_id}` already exist =>> Exiting!", ctx_level=1)
            return

        # Build (Map/Iterable) Dataset, using Model-Specific Prompt & Image Processor
        overwatch.info(f"Assembling OCID-Ref Map-Style Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = OCIDRefMapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor)

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
        result_sent_bbox_pairs = {}
        try:
            overwatch.info(f"Distributing Evaluation across {self.distributed_state.num_processes} GPUs", ctx_level=1)
            for example_ids, ref_exp_prompts, pixel_values, ref_exps, bboxes in tqdm(
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

                gen_bboxes = vlm.generate_answer(pixel_values, ref_exp_prompts)
                for example_id, gen_bbox, ref_exp, bbox_gt in zip(
                    example_ids, gen_bboxes, ref_exps, bboxes, strict=True
                ):
                    ex_id = int(example_id.item())
                    result_sent_bbox_pairs[ex_id] = {
                        "example_id": ex_id,
                        "ref_exp": ref_exp,
                        "model_output": gen_bbox,
                        "ground_truth_bbox": bbox_gt.numpy().tolist(),
                    }

        finally:
            with open(self.task_results_dir / f"results+rank-{self.distributed_state.process_index}.json", "w") as f:
                json.dump(result_sent_bbox_pairs, f, indent=2)

            # Block on all processes before returning!
        self.distributed_state.wait_for_everyone()
        overwatch.info("Done Evaluating =>> Exiting!", ctx_level=1)


# === Official Score Function Utilities =>> Just computes Acc@0.25 IOU ===
# Note =>> 0.25 vs. 0.5 IOU - this is from OCID-Ref Paper: https://arxiv.org/abs/2103.07679


def parse_bbox(gen_bbox: str) -> Optional[List[float]]:
    try:
        bbox_xyxy = ast.literal_eval(gen_bbox)
        assert isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4, "Invalid BBox"
        assert all(0 <= coord <= 1 for coord in bbox_xyxy), "Invalid Normalized BBox"
        assert (bbox_xyxy[0] < bbox_xyxy[2]) and (bbox_xyxy[1] < bbox_xyxy[3]), "Invalid BBox Format - should be XYXY"
        return bbox_xyxy
    except (AssertionError, ValueError, SyntaxError, TypeError):
        return None


def compute_iou(pred_bbox: List[float], gt_bbox: List[float]) -> float:
    """Computes IOU between two bboxes in xyxy format."""
    int_x1, int_y1 = max(pred_bbox[0], gt_bbox[0]), max(pred_bbox[1], gt_bbox[1])
    int_x2, int_y2 = min(pred_bbox[2], gt_bbox[2]), min(pred_bbox[3], gt_bbox[3])

    # Compute Box Areas
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

    # Compute Intersection Area
    intersection_area = max(0, int_x2 - int_x1) * max(0, int_y2 - int_y1)

    # Compute Union Area
    union_area = pred_area + gt_area - intersection_area

    # Return IOU
    return intersection_area / union_area


class OCIDRefScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_sent_bbox_pairs: Dict[str, Dict],
        annotations_file: Path,
        split: str = "val",
        **_: str,
    ) -> None:
        """Computes Acc @ 0.25 IOU --> standard metric for OCID-Ref splits (Min/Med/Max Clutter)."""
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.split = annotations_file, split
        self.full_result_sent_bbox_pairs = full_result_sent_bbox_pairs

        # Load Annotations File to Get Split Information
        with open(self.annotations_file, "r") as f:
            self.annotations = json.load(f)

    def score(self, model_id: str) -> Dict[str, float]:
        """Run Acc @ 0.25 IOU scoring on the predicted normalized boxes [x1 y1 x2 y2]; invalid outputs are failures."""
        ref_scores = {
            d: {"correct": 0, "invalid": 0, "incorrect": 0, "total": 0}
            for d in ["Minimum Clutter", "Medium Clutter", "Maximum Clutter"]
        }
        for example_id, example in tqdm(self.full_result_sent_bbox_pairs.items(), "=> Scoring Box Predictions:"):
            clutter_split = self.annotations[example_id]["clutter_split"]
            pred_bbox_xyxy = parse_bbox(example["model_output"])
            if pred_bbox_xyxy is None:
                ref_scores[clutter_split]["invalid"] += 1
                ref_scores[clutter_split]["total"] += 1
                continue

            # Otherwise, compute IOU between boxes!
            iou = compute_iou(pred_bbox_xyxy, example["ground_truth_bbox"])
            if iou >= 0.25:
                ref_scores[clutter_split]["correct"] += 1
                ref_scores[clutter_split]["total"] += 1
            else:
                ref_scores[clutter_split]["incorrect"] += 1
                ref_scores[clutter_split]["total"] += 1

        # Create Metrics Dictionary & Log =>> Additionally compute aggregate (full dataset) accuracy!
        accuracies = {f"accuracy__OCIDRef-{k.split()[0]}": v["correct"] / v["total"] for k, v in ref_scores.items()}
        accuracies["accuracy__OCIDRef-All"] = sum([v["correct"] for v in ref_scores.values()]) / sum(
            [v["total"] for v in ref_scores.values()]
        )
        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (Minimum/Medium/Maximum) (Split = Val)\n"
            f"          => OCIDRef-Minimum Accuracy (Official): {accuracies['accuracy__OCIDRef-Minimum']:.3f}\n"
            f"          => OCIDRef-Medium  Accuracy (Official): {accuracies['accuracy__OCIDRef-Medium']:.3f}\n"
            f"          => OCIDRef-Maximum Accuracy (Official): {accuracies['accuracy__OCIDRef-Maximum']:.3f}\n"
            f"          => OCIDRef-All     Accuracy (Official): {accuracies['accuracy__OCIDRef-All']:.3f}"
        )

        return accuracies
