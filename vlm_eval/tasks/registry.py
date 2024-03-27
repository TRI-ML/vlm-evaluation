"""
registry.py

Centralized file for storing metadata/download/extraction information for each of the evaluation datasets, in a way
that's hopefully transparent enough for folks trying to add new evaluations or replicate parts of this codebase.
"""
from enum import Enum
from pathlib import Path
from typing import Dict


# === Define Variants via String Enums for some type safety (note that __contains__ won't work) ===
class EvaluationDatasetType(str, Enum):
    vqa = "vqa"
    true_false = "true_false"
    captioning = "captioning"
    contrast = "contrast"
    refer = "refer"
    multiple_choice = "multiple_choice"


# === Full Task Registry (Evaluation Datasets & URLs & Local Paths) ===
# fmt: off
DATASET_REGISTRY: Dict[str, Dict] = {
    "vqa-v2": {
        "dataset_type": EvaluationDatasetType.vqa,
        "download": [
            {
                "name": "v2_val2014_questions.json",
                "extract": True,
                "extract_type": "file",
                "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
                "do_rename": True,
            },
            {
                "name": "v2_val2014_annotations.json",
                "extract": True,
                "extract_type": "file",
                "url": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
                "do_rename": True,
            },
            {
                "name": "images_val2014",
                "extract": True,
                "extract_type": "directory",
                "url": "http://images.cocodataset.org/zips/val2014.zip",
                "do_rename": True,
            },
        ],
        "paths": {
            # Raw Downloaded Data --> derived from the extraction parameters/names above (redundant, but hackable)
            "download_dir": Path("download/vqa-v2"),
            "questions": Path("download/vqa-v2/v2_val2014_questions.json"),
            "answers": Path("download/vqa-v2/v2_val2014_annotations.json"),
            "images": Path("download/vqa-v2/images_val2014"),

            # Dataset Directory --> stores index / metadata JSON file(s) for full and "slim" datasets
            #   Each `metadata-*.json` file is a complete, single source of truth for dataset initialization
            "dataset_dir": Path("datasets/vqa-v2"),
            "index_files": [
                Path("datasets/vqa-v2/metadata.json"),
                Path("datasets/vqa-v2/metadata-slim-512.json"),
                Path("datasets/vqa-v2/metadata-slim-1024.json"),
                Path("datasets/vqa-v2/metadata-slim-2048.json"),
            ],
        },
    },

    "gqa": {
        "dataset_type": EvaluationDatasetType.vqa,
        "download": [
            {
                "name": "testdev_balanced_questions.json",
                "extract": True,
                "extract_type": "directory",
                "url": "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip",
                "do_rename": False,
            },
            {
                "name": "images",
                "extract": True,
                "extract_type": "directory",
                "url": "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip",
                "do_rename": True,
            }
        ],
        "paths": {
            # Raw Downloaded Data --> derived from the extraction parameters/names above (redundant, but hackable)
            "download_dir": Path("download/gqa"),
            "questions_answers": Path("download/gqa/testdev_balanced_questions.json"),
            "images": Path("download/gqa/images"),

            # Dataset Directory --> stores index / metadata JSON file(s) for full and "slim" datasets
            #   Each `metadata-*.json` file is a complete, single source of truth for dataset initialization
            "dataset_dir": Path("datasets/gqa"),
            "index_files": [
                Path("datasets/gqa/metadata.json"),
                Path("datasets/gqa/metadata-slim-512.json"),
                Path("datasets/gqa/metadata-slim-1024.json"),
                Path("datasets/gqa/metadata-slim-2048.json"),
            ],
        }
    },

    "vizwiz": {
        "dataset_type": EvaluationDatasetType.vqa,
        "download": [
            {
                "name": "val_images",
                "extract": True,
                "extract_type": "directory",
                "url": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip",
                "do_rename": True,
            },
            {
                "name": "annotations",
                "extract": True,
                "extract_type": "directory",
                "url": "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip",
                "do_rename": False,
            },
        ],
        "paths": {
            # Raw Downloaded Data --> derived from the extraction parameters/names above (redundant, but hackable)
            "download_dir": Path("download/vizwiz"),
            "questions_answers": Path("download/vizwiz/val.json"),
            "images": Path("download/vizwiz/val_images"),

            # Dataset Directory --> stores index / metadata JSON file(s) for full and "slim" datasets
            #   Each `metadata-*.json` file is a complete, single source of truth for dataset initialization
            "dataset_dir": Path("datasets/vizwiz"),
            "index_files": [
                Path("datasets/vizwiz/metadata.json"),
                Path("datasets/vizwiz/metadata-slim-512.json"),
                Path("datasets/vizwiz/metadata-slim-1024.json"),
                Path("datasets/vizwiz/metadata-slim-2048.json"),
            ],
        }
    },


    "text-vqa": {
        "dataset_type": EvaluationDatasetType.vqa,
        "download": [
            {
                # IMPORTANT =>> See note: https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#scripts
                #   Turns out that for _all_ the evaluations, LLaVa authors provide different "custom" preprocessed
                #   versions of the questions. Usually this is benign (e.g., just appending "Answer with a short
                #   word or phrase" to the question to cut out on additional processing), but for TextVQA, it's drastic!
                #
                #   Specifically --> the preprocessed versions of the questions are further extended with the output of
                #   running an OCR system on the image (getting a list of extracted strings)! Maybe defeats the purpose
                #   of "TextVQA" --> but not sure what's standard for this evaluation.
                "name": "textvqa_val_v051_ocr.jsonl",
                "extract": False,
                "extract_type": "file",
                "url": ("https://gist.githubusercontent.com/siddk/74a709cb5e292099bae3cbcec1d0ce8a/raw/74e367e79a23b1043eb469b36168945fbac01601/textvqa_val_v051_ocr.jsonl"),
                "do_rename": True,
            },
            {
                "name": "TextVQA_0.5.1_val.json",
                "extract": False,
                "extract_type": "file",
                "url": "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json",
                "do_rename": True,
            },
            {
                "name": "train_val_images",
                "extract": True,
                "extract_type": "directory",
                "url": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
                "do_rename": True,
            },
        ],
        "paths": {
            # Raw Downloaded Data --> derived from the extraction parameters/names above (redundant, but hackable)
            "download_dir": Path("download/text-vqa"),
            "questions": Path("download/text-vqa/textvqa_val_v051_ocr.jsonl"),
            "answers": Path("download/text-vqa/TextVQA_0.5.1_val.json"),
            "images": Path("download/text-vqa/train_val_images"),

            # Dataset Directory --> stores index / metadata JSON file(s) for full and "slim" datasets
            #   Each `metadata-*.json` file is a complete, single source of truth for dataset initialization
            "dataset_dir": Path("datasets/text-vqa"),
            "index_files": [
                Path("datasets/text-vqa/metadata.json"),
                Path("datasets/text-vqa/metadata-slim-512.json"),
                Path("datasets/text-vqa/metadata-slim-1024.json"),
                Path("datasets/text-vqa/metadata-slim-2048.json"),
            ],
        },
    },

    "nocaps": {
        "dataset_type": EvaluationDatasetType.vqa,
        "download": [
            {
                "name": "nocaps_val_4500_captions.json",
                "extract": False,
                "extract_type": "file",
                "url": "https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json",
                "do_rename": True,
            },
            {
                "name": "validation",
                "extract": False,
                "extract_type": "directory",
                "url": "s3://open-images-dataset/validation",
                "do_rename": True,
                "aws": True
             },
        ],
        "paths": {
            # Raw Downloaded Data --> derived from the extraction parameters/names above (redundant, but hackable)
            "download_dir": Path("download/nocaps"),
            "images": Path("download/nocaps/validation"),
            "captions": Path("download/nocaps/nocaps_val_4500_captions.json"),

            # Dataset Directory --> stores index / metadata JSON file(s) for full and "slim" datasets
            #   Each `metadata-*.json` file is a complete, single source of truth for dataset initialization
            "dataset_dir": Path("datasets/nocaps"),
            "index_files": [
                Path("datasets/nocaps/metadata.json"),
                Path("datasets/nocaps/metadata-slim-512.json"),
                Path("datasets/nocaps/metadata-slim-1024.json"),
                Path("datasets/nocaps/metadata-slim-2048.json"),
            ],
          },
     },

    "pope": {
        "dataset_type": EvaluationDatasetType.vqa,
        "download": [
            {
                "name": "val_images",
                "extract": True,
                "extract_type": "directory",
                "url": "http://images.cocodataset.org/zips/val2014.zip",
                "do_rename": True,
            },
            {
                "name": "coco_pope_adversarial.json",
                "extract": False,
                "extract_type": "file",
                "url": ("https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/"
                        "output/coco/coco_pope_adversarial.json"),
                "do_rename": False,
            },
            {
                "name": "coco_pope_popular.json",
                "extract": False,
                "extract_type": "file",
                "url": ("https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/"
                        "output/coco/coco_pope_popular.json"),
                "do_rename": False,
            },
            {
                "name": "coco_pope_random.json",
                "extract": False,
                "extract_type": "file",
                "url": ("https://raw.githubusercontent.com/AoiDragon/POPE/e3e39262c85a6a83f26cf5094022a782cb0df58d/"
                        "output/coco/coco_pope_random.json"),
                "do_rename": False,
            },
        ],
        "paths": {
            # Raw Downloaded Data --> derived from the extraction parameters/names above (redundant, but hackable)
            "download_dir": Path("download/pope"),
            "qa_adversarial": Path("download/pope/coco_pope_adversarial.json"),
            "qa_popular": Path("download/pope/coco_pope_popular.json"),
            "qa_random": Path("download/pope/coco_pope_random.json"),
            "images": Path("download/pope/val_images"),

            # Dataset Directory --> stores index / metadata JSON file(s) for full and "slim" datasets
            #   Each `metadata-*.json` file is a complete, single source of truth for dataset initialization
            "dataset_dir": Path("datasets/pope"),
            "index_files": [
                Path("datasets/pope/metadata.json"),
            ],
        }
    },

    "vsr": {
        "dataset_type": EvaluationDatasetType.true_false,
        "download": [
            {
                "name": "test.jsonl",
                "extract": False,
                "extract_type": "file",
                "url": "https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot/raw/main/test.jsonl",
                "do_rename": True,
            },
            {
                "name": "images",
                "extract": True,
                "extract_type": "directory",
                "url": "https://www.dropbox.com/s/0s3bj25s62crjh2/vsr_images.zip?dl=1",
                "do_rename": False,
            }
        ],
        "paths": {
            # Raw Downloaded Data --> derived from the extraction parameters/names above (redundant, but hackable)
            "download_dir": Path("download/vsr"),
            "questions_answers": Path("download/vsr/test.jsonl"),
            "images": Path("download/vsr/images"),

            # Dataset Directory --> stores index / metadata JSON file for full dataset (no slim variant; small set)
            "dataset_dir": Path("datasets/vsr"),
            "index_files": [
                Path("datasets/vsr/metadata.json"),
            ]
        }
    },

    "refcoco": {
        "dataset_type": EvaluationDatasetType.refer,
        "download": [
            {
                "name": "refcoco",
                "extract": True,
                "extract_type": "directory",
                "url": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
                "do_rename": False,
            },
            {
                "name": "refcoco+",
                "extract": True,
                "extract_type": "directory",
                "url": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip",
                "do_rename": False,
            },
            {
                "name": "refcocog",
                "extract": True,
                "extract_type": "directory",
                "url": "https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip",
                "do_rename": False,
            },
            {
                "name": "train2014",
                "extract": True,
                "extract_type": "directory",
                "url": "http://images.cocodataset.org/zips/train2014.zip",
                "do_rename": False,
            },
        ],
        "paths": {
            # Raw Downloaded Data --> derived from the extraction parameters/names above (redundant, but hackable)
            "download_dir": Path("download/refcoco"),
            "images": Path("download/refcoco/train2014"),

            # Dataset Directory --> stores index / metadata JSON file for full dataset (no slim variant; small set)
            "dataset_dir": Path("datasets/refcoco"),
            "index_files": [
                Path("datasets/refcoco/metadata.json"),
            ]
        }
    },

    "ocid-ref": {
        "dataset_type": EvaluationDatasetType.refer,
        "download": [
            # Not sure how reliable these links are =>> the JSON file is a Google Drive link, and the raw dataset
            #   link seems like it could rotate (i.e., different link each day).
            {
                "name": "val_expressions.json",
                "extract": False,
                "extract_type": "file",
                "url": "https://drive.google.com/uc?export=download&id=19dD65ABxWiOLrpNea7mUW1RjJDBLo3Yp",
                "do_rename": False,
            },
            {

                "name": "OCID-dataset",
                "extract": True,
                "extract_type": "directory",
                "url": "https://data.acin.tuwien.ac.at/index.php/s/g3EkcgcPioolQmJ/download",
                "do_rename": True,
            },
        ],
        "paths": {
            # Raw Downloaded Data --> derived from the extraction parameters/names above (redundant, but hackable)
            "download_dir": Path("download/ocid-ref"),
            "referring_expressions": Path("download/ocid-ref/val_expressions.json"),
            "images": Path("download/ocid-ref/OCID-dataset"),

            # Dataset Directory --> stores index / metadata JSON file for full dataset (no slim variant; small set)
            "dataset_dir": Path("datasets/ocid-ref"),
            "index_files": [
                Path("datasets/refcoco/metadata.json"),
            ]
        },
    },

    "tally-qa": {
        "dataset_type": EvaluationDatasetType.multiple_choice,
        "download": [
            {
                "name": "test.json",
                "extract": True,
                "extract_type": "directory",
                "url": "https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true",
                "do_rename": False,
            },
            {
                "name": "VG_100K",
                "extract": True,
                "extract_type": "directory",
                "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
                "do_rename": False,
            },
            {
                "name": "VG_100K_2",
                "extract": True,
                "extract_type": "directory",
                "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
                "do_rename": False,
            },
        ],
        "paths": {
            # Raw Downloaded Data --> derived from the extraction parameters/names above (redundant, but hackable)
            "download_dir": Path("download/tally-qa"),
            "questions_answers": Path("download/tally-qa/test.json"),
            "images": Path("download/tally-qa"),

            # Dataset Directory --> stores index / metadata JSON file(s) for full and "slim" datasets
            #   Each `metadata-*.json` file is a complete, single source of truth for dataset initialization
            "dataset_dir": Path("datasets/tally-qa"),
            "index_files": [
                Path("datasets/tally-qa/metadata.json"),
            ],
        },
    },

    "ai2d": {
        "dataset_type": EvaluationDatasetType.multiple_choice,
        "download": [
            {
                "name": "val_qa",
                "extract": True,
                "extract_type": "directory",
                "url": "http://ai2-website.s3.amazonaws.com/data/ai2d-all.zip",
                "do_rename": True,
            },
            {
                "name": "test_ids",
                "extract": False,
                "extract_type": "file",
                "url": "https://s3-us-east-2.amazonaws.com/prior-datasets/ai2d_test_ids.csv",
                "do_rename": False,
            },
        ],
        "paths": {
            # Raw Downloaded Data --> derived from the extraction parameters/names above (redundant, but hackable)
            "download_dir": Path("download/ai2d"),
            "questions_answers": Path("download/ai2d/val_qa/questions"),
            "images": Path("download/ai2d/val_qa/images"),
            "test_ids": Path("download/ai2d/ai2d_test_ids.csv"),

            # Dataset Directory --> stores index / metadata JSON file(s) for full and "slim" datasets
            #   Each `metadata-*.json` file is a complete, single source of truth for dataset initialization
            "dataset_dir": Path("datasets/ai2d"),
            "index_files": [
                Path("datasets/ai2d/metadata.json"),
                Path("datasets/ai2d/metadata-slim-512.json"),
                Path("datasets/ai2d/metadata-slim-1024.json"),
                Path("datasets/ai2d/metadata-slim-2048.json"),
            ],
        }
    },

}
# fmt: on
