"""
refcoco.py

Task Runner, Dataset Definitions, Builder Functions, and Evaluation Logic for the RefCOCO / RefCOCO+ / RefCOCOg
referring expression grounding (bounding box prediction) datasets. Only loads & processes the RefCOCO* *validation sets*
-- the various test splits (testA/testB) are left alone (for our evaluation).
"""
import ast
import json
import os
import re
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
from vlm_eval.util.loading.refer import REFER

# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)


# === Bounding Box Utilities ===
def box_xywh2xyxy(bbox_xywh: List[float], img_wh: Tuple[int, int], do_normalize: bool = True) -> List[float]:
    bbox_xyxy = [bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]
    width, height = img_wh

    # Validate
    assert bbox_xyxy[0] < bbox_xyxy[2] <= width, "Invalid BBox Width!"
    assert bbox_xyxy[1] < bbox_xyxy[3] <= height, "Invalid BBox Height!"

    # Handle Normalization
    if do_normalize:
        bbox_xyxy = [bbox_xyxy[0] / width, bbox_xyxy[1] / height, bbox_xyxy[2] / width, bbox_xyxy[3] / height]

    # Return Box Coordinates rounded to 2 decimal places!
    return [round(coord, 2) for coord in bbox_xyxy]


# === Dataset Indexing / Building Utilities ===


# ruff: noqa: C901
def build_refcoco_indices(root_dir: Path, slim_dataset_sizes: Optional[Tuple[int, ...]], seed: int = 21) -> List[Path]:
    """Parse RefCOCO* validation sets --> build & write index files w/ necessary keys + additional metadata."""
    paths = DATASET_REGISTRY["refcoco"]["paths"]
    os.makedirs(dataset_dir := root_dir / paths["dataset_dir"], exist_ok=True)

    # Short-Circuit (if index files have already been built)
    index_files = [dataset_dir / "metadata-full.json"] + (
        []
        if slim_dataset_sizes is None
        else [dataset_dir / f"metadata-slim-{n_slim}.json" for n_slim in slim_dataset_sizes]
    )
    if all([index_file.exists() for index_file in index_files]):
        return index_files

    # Otherwise, use the REFER API to load the raw expressions & annotations from the three splits
    download_dir = str(root_dir / "download" / "refcoco")
    refcoco = REFER(download_dir, "refcoco", splitBy="unc")
    refcocop = REFER(download_dir, "refcoco+", splitBy="unc")
    refcocog = REFER(download_dir, "refcocog", splitBy="umd")

    # Build Full Metadata Structure
    index = {}
    for refer_dataset, refer in [("RefCOCO", refcoco), ("RefCOCO+", refcocop), ("RefCOCOg", refcocog)]:
        overwatch.info(f"Processing {refer_dataset} - Validation Split!")

        # Get Ref IDs for "val" Split =>> Iterate
        ref_ids = refer.getRefIds(split="val")
        assert len(ref_ids) == (
            count := {"RefCOCO": 3811, "RefCOCO+": 3805, "RefCOCOg": 2573}[refer_dataset]
        ), f"Expected {count} refs in {refer_dataset}!"
        for ref_id in tqdm(ref_ids, desc=f"=> Processing {refer_dataset} Val Set:"):
            ref, annotation = refer.Refs[ref_id], refer.refToAnn[ref_id]

            # Get Image Path & Full Bounding Box (as [x, y, w, h] =>> convert to [x, y, x, y])
            img_path = paths["images"] / f"COCO_train2014_{ref['image_id']:012d}.jpg"
            assert (root_dir / img_path).exists(), f"Image `{img_path}` for Ref ID `{ref_id}` does not exist!"
            bbox_xywh, img_size_wh = annotation["bbox"], Image.open(root_dir / img_path).size

            # Compute Normalized Bounding Box
            normalized_box_xyxy = box_xywh2xyxy(bbox_xywh, img_size_wh, do_normalize=True)

            # Iterate through Sentences tied to Ref =>> Add to Index
            for sent_blob in ref["sentences"]:
                example_id: int = hash(f"{refer_dataset}-{ref_id}-{sent_blob['sent_id']}")
                assert example_id not in index, "Hash collision -- do something smarter!"

                # Build Metadata Entry
                # fmt: off
                index[example_id] = {
                    # [Required] RefCOCO Localization Task Keys
                    "example_id": example_id,
                    "ref_expression": sent_blob["sent"],
                    "img_path": str(img_path),
                    "bbox": normalized_box_xyxy,

                    # Additional Metadata
                    "refer_dataset": refer_dataset,
                    "split": "val",
                    "ref_id": ref_id,
                    "sent_id": sent_blob["sent_id"]
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
            slim_ex_ids, counts = [], {"RefCOCO": 0, "RefCOCO+": 0, "RefCOCOg": 0}
            for ex_id in all_ex_ids:
                refer_dataset = index[ex_id]["refer_dataset"]
                if counts[refer_dataset] < n_slim:
                    slim_ex_ids.append(ex_id)
                    counts[refer_dataset] += 1

                # Termination Condition
                if all([c == n_slim for c in counts.values()]):
                    break

            # Dump Sampled Examples
            with open(index_file, "w") as f:
                json.dump({k: index[k] for k in slim_ex_ids}, f)

        else:
            raise ValueError(f"Received unexpected index file `{index_file}`")

    return index_files


# === Index (Metadata-Only) Dataset Declarations ===
class RefCOCOIndexDataset(Dataset):
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
class RefCOCOMapDataset(Dataset):
    def __init__(
        self, root_dir: Path, index_file: Path, prompt_fn: Callable[[str], str], image_processor: ImageProcessor
    ) -> None:
        """
        Constructs a fully-fledged PyTorch Map-Style Dataset for evaluating on splits of the RefCOCO Validation Sets. In
        addition to the path to the dataset `index_file` to load from, requires a `prompt_fn` for formatting individual
        expressions (model-specific), and an `image_processor` for applying any required image transforms.

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


# === RefCOCO Task Runner ===
class RefCOCOTaskRunner:
    def __init__(
        self,
        root_dir: Path,
        index_file: Path,
        task_results_dir: Path,
        model_id: str,
        prompt_fn: Callable[[str], str],
        image_processor: ImageProcessor,
    ) -> None:
        """Task Runner for the RefCOCO Dataset; loads data, then runs (distributed) VLM evaluation & writes results."""
        self.root_dir, self.index_file, self.task_results_dir = root_dir, index_file, task_results_dir
        self.model_id, self.prompt_fn, self.image_processor = model_id, prompt_fn, image_processor

        # === Unfortunate Pattern =>> Accelerate injects a lot of additional stuff into env; minimize collateral ===
        from accelerate import PartialState

        self.distributed_state = PartialState()

        # Short-Circuit (if results/metrics already exist)
        os.makedirs(self.task_results_dir, exist_ok=True)
        if (self.task_results_dir / "metrics.json").exists():
            overwatch.info(f"RefCOCO Metrics for Model `{self.model_id}` already exist =>> Exiting!", ctx_level=1)
            return

        # Build (Map/Iterable) Dataset, using Model-Specific Prompt & Image Processor
        overwatch.info(f"Assembling RefCOCO Map-Style Dataset from {self.root_dir / self.index_file}", ctx_level=1)
        self.dataset = RefCOCOMapDataset(self.root_dir, self.index_file, self.prompt_fn, self.image_processor)

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


# === Official Score Function =>> Just computes Acc@0.5 IOU ===
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


class RefCOCOScorer:
    def __init__(
        self,
        dataset_id: str,
        task_results_dir: Path,
        full_result_sent_bbox_pairs: Dict[str, Dict],
        annotations_file: Path,
        split: str = "val",
        **_: str,
    ) -> None:
        """Computes Acc @ 0.5 IOU --> standard RefCOCO / RefCOCO+ / RefCOCOg metric."""
        self.dataset_id, self.task_results_dir = dataset_id, task_results_dir
        self.annotations_file, self.split = annotations_file, split
        self.full_result_sent_bbox_pairs = full_result_sent_bbox_pairs

        # Load Annotations File to Get Split Information
        with open(self.annotations_file, "r") as f:
            self.annotations = json.load(f)

    def score(self, model_id: str) -> Dict[str, float]:
        """Run Acc @ 0.5 IOU scoring on the predicted normalized boxes [x1 y1 x2 y2]; invalid outputs are failures."""
        ref_scores = {
            d: {"correct": 0, "invalid": 0, "incorrect": 0, "total": 0} for d in ["RefCOCO", "RefCOCO+", "RefCOCOg"]
        }
        for example_id, example in tqdm(self.full_result_sent_bbox_pairs.items(), "=> Scoring Box Predictions:"):
            dataset = self.annotations[example_id]["refer_dataset"]
            pred_bbox_xyxy = parse_bbox(example["model_output"])
            if pred_bbox_xyxy is None:
                ref_scores[dataset]["invalid"] += 1
                ref_scores[dataset]["total"] += 1
                continue

            # Otherwise, compute IOU between boxes!
            iou = compute_iou(pred_bbox_xyxy, example["ground_truth_bbox"])
            if iou >= 0.5:
                ref_scores[dataset]["correct"] += 1
                ref_scores[dataset]["total"] += 1
            else:
                ref_scores[dataset]["incorrect"] += 1
                ref_scores[dataset]["total"] += 1

        # Create Metrics Dictionary & Log
        accuracies = {f"accuracy__{k}": v["correct"] / v["total"] for k, v in ref_scores.items()}
        overwatch.info(
            f"Results for Model `{model_id}` on {self.dataset_id} (RefCOCO/RefCOCO+/RefCOCOg) (Split = Val)\n"
            f"          => RefCOCO  Accuracy (Official): {accuracies['accuracy__RefCOCO']:.3f}\n"
            f"          => RefCOCO+ Accuracy (Official): {accuracies['accuracy__RefCOCO+']:.3f}\n"
            f"          => RefCOCOg Accuracy (Official): {accuracies['accuracy__RefCOCOg']:.3f}"
        )

        return accuracies
