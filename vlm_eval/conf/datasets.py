"""
datasets.py

Draccus Dataclass Definitions for a DatasetConfig type, with various registered subclasses for each dataset type and
variant thereof (e.g., the "slim-1024" variant of the "vqa-v2") dataset.
"""
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional

from draccus import ChoiceRegistry


@dataclass
class DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_family: str                     # Dataset family (e.g., "vqa-v2") to evaluate
    dataset_id: str                         # Unique identifier for a given split (e.g., <dataset>-slim)
    split: str                              # Split of the original dataset (e.g., "val" | "testdev-balanced" | ...)

    expected_examples: int                  # Number of expected examples in the dataset (for verification)

    root_dir: Path                          # Path to root directory for storing datasets & results (on local disk)
    index_file: Path                        # File specifying the dataset variant to load (`metadata-slim-{k}.json`)
    annotations_file: Path                  # File with the raw annotations in the "official format" (for eval scripts)
    questions_file: Optional[Path] = None   # File with the raw questions in "official format" (optional)
    ocr: Optional[bool] = False             # Whether to use OCR in question or not (only relevant for Text VQA)
    # fmt: on


# === VQA-v2 Datasets =>> Note: "Slim" defaults to k = 1024 examples ===


@dataclass
class VQAv2FullDatasetConfig(DatasetConfig):
    dataset_family: str = "vqa-v2"
    dataset_id: str = "vqa-v2-full"
    split: str = "val"

    expected_examples: int = 214354

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/vqa-v2/metadata.json")
    annotations_file: Path = Path("datasets/vqa-v2/annotations-vqa-v2-full.json")
    questions_file: Path = Path("datasets/vqa-v2/questions-vqa-v2-full.json")


@dataclass
class VQAv2SubSampledDatasetConfig(DatasetConfig):
    dataset_family: str = "vqa-v2"
    dataset_id: str = "vqa-v2-subsampled"
    split: str = "val"

    expected_examples: int = 16384
    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/vqa-v2/metadata-slim-16384.json")
    annotations_file: Path = Path("datasets/vqa-v2/annotations-vqa-v2-slim-16384.json")
    questions_file: Path = Path("datasets/vqa-v2/questions-vqa-v2-slim-16384.json")


@dataclass
class VQAv2SlimDatasetConfig(DatasetConfig):
    dataset_family: str = "vqa-v2"
    dataset_id: str = "vqa-v2-slim"
    split: str = "val"

    expected_examples: int = 1024

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/vqa-v2/metadata-slim-1024.json")
    annotations_file: Path = Path("datasets/vqa-v2/annotations-vqa-v2-slim-1024.json")
    questions_file: Path = Path("datasets/vqa-v2/questions-vqa-v2-slim-1024.json")


# === GQA Datasets =>> Note: "Slim" defaults to k = 1024 examples ===
@dataclass
class GQAFullDatasetConfig(DatasetConfig):
    dataset_family: str = "gqa"
    dataset_id: str = "gqa-full"
    split: str = "testdev_balanced"

    expected_examples: int = 12578

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/gqa/metadata-full.json")
    annotations_file: Path = Path("datasets/gqa/annotations-gqa-full.json")


@dataclass
class GQASlimDatasetConfig(DatasetConfig):
    dataset_family: str = "gqa"
    dataset_id: str = "gqa-slim"
    split: str = "testdev_balanced"

    expected_examples: int = 1024

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/gqa/metadata-slim-1024.json")
    annotations_file: Path = Path("datasets/gqa/annotations-gqa-slim-1024.json")


# === VizWiz Datasets =>> Note: "Slim" defaults to k = 1024 examples ===
@dataclass
class VizWizFullDatasetConfig(DatasetConfig):
    dataset_family: str = "vizwiz"
    dataset_id: str = "vizwiz-full"
    split: str = "val"

    expected_examples: int = 4319

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/vizwiz/metadata.json")
    annotations_file: Path = Path("datasets/vizwiz/annotations-vizwiz-full.json")
    questions_file: Path = Path("datasets/vizwiz/questions-vizwiz-full.json")


@dataclass
class VizWizSlimDatasetConfig(DatasetConfig):
    dataset_family: str = "vizwiz"
    dataset_id: str = "vizwiz-slim"
    split: str = "val"

    expected_examples: int = 1024

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/vizwiz/metadata-slim-1024.json")
    annotations_file: Path = Path("datasets/vizwiz/annotations-vizwiz-slim-1024.json")
    questions_file: Path = Path("datasets/vizwiz/questions-vizwiz-slim-1024.json")


# === Text VQA Datasets =>> Note: "Slim" defaults to k = 1024 examples ===
@dataclass
class TextVQAFullDatasetConfig(DatasetConfig):
    dataset_family: str = "text-vqa"
    dataset_id: str = "text-vqa-full"
    split: str = "val"

    expected_examples: int = 5000

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/text-vqa/metadata.json")
    annotations_file: Path = Path("datasets/text-vqa/annotations-textvqa-full.json")


@dataclass
class TextVQAFullOCRDatasetConfig(DatasetConfig):
    dataset_family: str = "text-vqa"
    dataset_id: str = "text-vqa-ocr-full"
    split: str = "val"

    expected_examples: int = 5000

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/text-vqa/metadata.json")
    annotations_file: Path = Path("datasets/text-vqa/annotations-textvqa-full.json")

    ocr: bool = True


@dataclass
class TextVQASlimDatasetConfig(DatasetConfig):
    dataset_family: str = "text-vqa"
    dataset_id: str = "text-vqa-slim"
    split: str = "val"

    expected_examples: int = 1024

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/text-vqa/metadata-slim-1024.json")
    annotations_file: Path = Path("datasets/text-vqa/annotations-textvqa-slim-1024.json")


# === NoCaps Captioning Datasets =>> Note: "Slim" defaults to k = 1024 examples ===
@dataclass
class NoCapsFullDatasetConfig(DatasetConfig):
    dataset_family: str = "nocaps"
    dataset_id: str = "nocaps-full"
    split: str = "val"

    expected_examples: int = 4500

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/nocaps/metadata-full.json")
    annotations_file: Path = Path("datasets/nocaps/metadata-full.json")


@dataclass
class NoCapsSlimDatasetConfig(DatasetConfig):
    dataset_family: str = "nocaps"
    dataset_id: str = "nocaps-slim"
    split: str = "val"

    expected_examples: int = 1024

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/nocaps/metadata-slim-1024.json")
    annotations_file: Path = Path("datasets/nocaps/metadata-slim-1024.json")


# === Visual Spatial Reasoning (True/False) Datasets =>> Note: Using the "zero-shot" test split as in InstructBLIP ===
@dataclass
class VSRFullDatasetConfig(DatasetConfig):
    dataset_family: str = "vsr"
    dataset_id: str = "vsr-full"
    split: str = "zeroshot-test"

    expected_examples: int = 1222

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/vsr/metadata-full.json")
    annotations_file: Path = Path("datasets/vsr/metadata-full.json")


# === RefCOCO / RefCOCO+ / RefCOCOg (BBox Prediction) Datasets =>> Note: Using the "validation" sets *only* ===
@dataclass
class RefCOCOFullDatasetConfig(DatasetConfig):
    dataset_family: str = "refcoco"
    dataset_id: str = "refcoco-full"
    split: str = "val"

    expected_examples: int = 26488

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/refcoco/metadata-full.json")
    annotations_file: Path = Path("datasets/refcoco/metadata-full.json")


@dataclass
class RefCOCOSlimDatasetConfig(DatasetConfig):
    dataset_family: str = "refcoco"
    dataset_id: str = "refcoco-slim"
    split: str = "val"

    # Examples =>> n = 1024 for each of RefCOCO/RefCOCO+/RefCOCOg = 3 * 1024 = 3072
    expected_examples: int = 3072

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/refcoco/metadata-slim-1024.json")
    annotations_file: Path = Path("datasets/refcoco/metadata-slim-1024.json")


# === OCID-Ref Datasets =>> "Minimum Clutter" | "Medium Clutter" | "Max Clutter" Splits (using Validation *only*) ===
@dataclass
class OCIDRefFullDatasetConfig(DatasetConfig):
    dataset_family: str = "ocid-ref"
    dataset_id: str = "ocid-ref-full"
    split: str = "val"

    expected_examples: int = 18342

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/ocid-ref/metadata-full.json")
    annotations_file: Path = Path("datasets/ocid-ref/metadata-full.json")


@dataclass
class OCIDRefSlimDatasetConfig(DatasetConfig):
    dataset_family: str = "ocid-ref"
    dataset_id: str = "ocid-ref-slim"
    split: str = "val"

    # Examples =>> n = 1024 for each of the Min/Med/Max Clutter Splits = 3 * 1024 = 3072
    expected_examples: int = 3072

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/ocid-ref/metadata-slim-1024.json")
    annotations_file: Path = Path("datasets/ocid-ref/metadata-slim-1024.json")


@dataclass
class TallyQAFullDatasetConfig(DatasetConfig):
    dataset_family: str = "tally-qa"
    dataset_id: str = "tally-qa-full"
    split: str = "test"

    expected_examples: int = 38589

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/tally-qa/metadata-full.json")
    annotations_file: Path = Path("datasets/tally-qa/metadata-full.json")


@dataclass
class TallyQASubsampledDatasetConfig(DatasetConfig):
    dataset_family: str = "tally-qa"
    dataset_id: str = "tally-qa-subsampled"
    split: str = "test"

    # Examples =>> n = 8192 for each of "Simple" and "Complex" = 2 * 8192 = 16384
    expected_examples: int = 16384

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/tally-qa/metadata-slim-8192.json")
    annotations_file: Path = Path("datasets/tally-qa/metadata-slim-8192.json")


@dataclass
class TallyQASlimDatasetConfig(DatasetConfig):
    dataset_family: str = "tally-qa"
    dataset_id: str = "tally-qa-slim"
    split: str = "test"

    # Examples =>> n = 1024 for each of "Simple" and "Complex" = 2 * 1024 = 2048
    expected_examples: int = 2048

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/tally-qa/metadata-slim-1024.json")
    annotations_file: Path = Path("datasets/tally-qa/metadata-slim-1024.json")


# === Pope Datasets =>> Note: "Slim" defaults to k = 1024 examples ===
@dataclass
class PopeFullDatasetConfig(DatasetConfig):
    dataset_family: str = "pope"
    dataset_id: str = "pope-full"
    split: str = "eval"

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/pope/metadata-full.json")
    annotations_file: Path = Path("datasets/pope/metadata-full.json")

    # Examples = n = 3000 for each of adversarial/popular/random splits = 3 * 3000 = 9000
    expected_examples: int = 9000


@dataclass
class PopeSlimDatasetConfig(DatasetConfig):
    dataset_family: str = "pope"
    dataset_id: str = "pope-slim"
    split: str = "eval"

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/pope/metadata-slim-1024.json")
    annotations_file: Path = Path("datasets/pope/metadata-slim-1024.json")

    # Examples = n = 1024 for each of adversarial/popular/random splits = 3 * 1024 = 3072
    expected_examples: int = 3072


# === AI2D Datasets =>> Note: "Slim" defaults to k = 1024 examples ===
@dataclass
class AI2DFullDatasetConfig(DatasetConfig):
    dataset_family: str = "ai2d"
    dataset_id: str = "ai2d-full"
    split: str = "eval"

    expected_examples: int = 15501

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/ai2d/metadata-full.json")
    annotations_file: Path = Path("datasets/ai2d/metadata-full.json")


@dataclass
class AI2DSlimDatasetConfig(DatasetConfig):
    dataset_family: str = "ai2d"
    dataset_id: str = "ai2d-slim"
    split: str = "eval"

    expected_examples: int = 2048

    root_dir: Path = Path("../../datasets/vlm-evaluation")
    index_file: Path = Path("datasets/ai2d/metadata-slim-1024.json")
    annotations_file: Path = Path("datasets/ai2d/metadata-slim-1024.json")


# === Define a Dataset Registry Enum for Reference / Validation =>> all *new* datasets must be added here! ===
@unique
class DatasetRegistry(Enum):
    # VQA-v2
    VQAv2_FULL = VQAv2FullDatasetConfig
    VQAv2_SUBSAMPLED = VQAv2SubSampledDatasetConfig
    VQAv2_SLIM = VQAv2SlimDatasetConfig

    # GQA
    GQA_FULL = GQAFullDatasetConfig
    GQA_SLIM = GQASlimDatasetConfig

    # VizWiz
    VIZWIZ_FULL = VizWizFullDatasetConfig
    VIZWIZ_SLIM = VizWizSlimDatasetConfig

    # VQAText
    TEXTVQA_FULL = TextVQAFullDatasetConfig
    TEXTVQA_FULL_OCR = TextVQAFullOCRDatasetConfig
    TEXTVQA_SLIM = TextVQASlimDatasetConfig

    # NoCaps
    NOCAPS_FULL = NoCapsFullDatasetConfig
    NOCAPS_SLIM = NoCapsSlimDatasetConfig

    # VSR (True/False)
    VSR_FULL = VSRFullDatasetConfig

    # Pope (Yes/No)
    POPE_FULL = PopeFullDatasetConfig
    POPE_SLIM = PopeSlimDatasetConfig

    # RefCOCO / RefCOCO+ / RefCOCOg (BBox Prediction)
    REFCOCO_FULL = RefCOCOFullDatasetConfig
    REFCOCO_SLIM = RefCOCOSlimDatasetConfig

    # OCID-Ref - Min/Med/Max Clutter Splits (BBox Prediction)
    OCIDREF_FULL = OCIDRefFullDatasetConfig
    OCIDREF_SLIM = OCIDRefSlimDatasetConfig

    # TallyQA - Simple & Complex Splits
    TALLYQA_FULL = TallyQAFullDatasetConfig
    TALLYQA_SUBSAMPLED = TallyQASubsampledDatasetConfig
    TALLYQA_SLIM = TallyQASlimDatasetConfig

    # AI2D
    AI2D_FULL = AI2DFullDatasetConfig
    AI2D_SLIM = AI2DSlimDatasetConfig

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in DatasetRegistry:
    DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
