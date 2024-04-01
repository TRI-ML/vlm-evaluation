"""
prepare.py

Entry point for dataset downloading & preparation -- handles all aspects of the raw data acquisition, extraction, and
verification process, writing both WebDataset and Mosaic Streaming (MDS) versions of the data.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import draccus

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks import build_index_datasets, download_extract

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class DatasetPreparationConfig:
    # fmt: off
    dataset_family: str = "ai2d"                                # Dataset family to prepare

    # Processing Parameters
    create_slim_dataset: bool = True                            # Whether to create "slim" (minified) dataset(s)
    slim_dataset_sizes: Tuple[int, ...] = (                     # Number of examples for the slim dataset(s)
        1024, 8192
    )
    export_formats: Tuple[str, ...] = (                         # Formats for export (always writes a "Map" Dataset)
        "webdataset",
        "mosaic-streaming",
    )

    # Format-Specific Parameters
    max_shard_size_bytes: int = 64000000                        # Maximum size for a shard in bytes (default: 64 MB)
    wds_examples_per_shard: int = 1024                          # [WebDataset] Number of examples per `tar` shard
    mds_hashes: Tuple[str, str] = ("sha1", "xxh64")             # [Mosaic] Pair of (crypto, non-crypto) hash functions

    # Path Parameters
    root_dir: Path = Path(                                      # Path to root directory for storing datasets
        # "datasets/vlm-evaluation"
        "/mnt/fsx/skaramcheti/datasets/vlm-evaluation"
    )

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Env Variable or Path to HF Token

    # Randomness
    seed: int = 21                                              # Random Seed (for slim datasets, augmentations)
    # fmt: on


@draccus.wrap()
def prepare(cfg: DatasetPreparationConfig) -> None:
    overwatch.info(f"Downloading and Preparing VLM Evaluation Dataset `{cfg.dataset_family}`")

    # Phase 1 :: Download & Extract Raw Data to `cfg.data_dir` / cfg.dataset_id / "download"
    overwatch.info(f"Phase 1 =>> Downloading & Extracting `{cfg.dataset_family}` to {cfg.root_dir / 'download'}")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    download_extract(cfg.dataset_family, cfg.root_dir, hf_token)

    # Phase 2 :: Assemble Index Dataset(s) (always builds metadata from local disk, then used to export other formats)
    overwatch.info(f"Phase 2 =>> Building Index Dataset(s) for `{cfg.dataset_family}` at {cfg.root_dir / 'datasets'}")
    index_datasets = build_index_datasets(
        cfg.dataset_family,
        cfg.root_dir,
        slim_dataset_sizes=cfg.slim_dataset_sizes if cfg.create_slim_dataset else None,
        seed=cfg.seed,
    )

    # Phase 3 :: Build Streaming / Iterable Datasets in the desired format(s)
    return index_datasets


if __name__ == "__main__":
    prepare()
