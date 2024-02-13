"""
download.py

Utility functions for downloading and extracting various datasets to (local) disk.
"""
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from zipfile import ZipFile

import requests
from rich.progress import BarColumn, DownloadColumn, MofNCompleteColumn, Progress, TextColumn, TransferSpeedColumn

from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks.registry import DATASET_REGISTRY

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


def download_with_progress(url: str, download_dir: Path, chunk_size_bytes: int = 1024, hf_token=None) -> Path:
    """Utility function for downloading files from the internet, with a handy Rich-based progress bar."""

    # Fire an HTTP Request, with `stream = True` => we at least want the Request Headers to validate!
    if hf_token is None:
        response = requests.get(url, stream=True)
    else:
        response = requests.get(url, headers={"Authorization": f"Bearer {hf_token}"}, stream=True)

    # Handle Filename Parsing (if not clear from URL)
    dest_path = download_dir / Path(url).name.split("?")[0] if "drive.google" not in url else ""
    if dest_path == "":
        # Parse Content-Headers --> "Content-Disposition" --> filename
        filename = re.findall('filename="(.+)"', response.headers["content-disposition"])[0]
        dest_path = download_dir / filename

    # Download / Short-Circuit if exists
    overwatch.info(f"Downloading {dest_path} from `{url}`", ctx_level=1)
    if dest_path.exists():
        return dest_path

    # Download w/ Transfer-Aware Progress
    #   => Reference: https://github.com/Textualize/rich/blob/master/examples/downloader.py
    with Progress(
        TextColumn("[bold]{task.description} - {task.fields[fname]}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        transient=True,
    ) as dl_progress:
        total_raw = response.headers.get("content-length", None)
        total = int(total_raw) if total_raw is not None else None
        dl_tid = dl_progress.add_task("Downloading", fname=dest_path.name, total=total)
        with open(dest_path, "wb") as f:
            for data in response.iter_content(chunk_size=chunk_size_bytes):
                dl_progress.advance(dl_tid, f.write(data))

    return dest_path


def extract_with_progress(archive_path: Path, download_dir: Path, extract_type: str, cleanup: bool = True) -> Path:
    """Utility function for extracting compressed archives, with a handy Rich-based progress bar."""
    ## Semi-hacky naming fix for Ocid-ref because the download file is named differently
    if "ocid-ref" in download_dir.as_posix():
        renamed = "/".join(archive_path.as_posix().split("/")[:-1]) + "/OCID-dataset.tar.gz"
        os.rename(archive_path.as_posix(), renamed)
        archive_path = Path(renamed)
        
    assert archive_path.suffix in {".gz", ".tar", ".zip"}, f"Invalid compressed archive `{archive_path}`!"
    overwatch.info(f"Extracting {archive_path.name} to `{download_dir}`", ctx_level=1)

    # Extract w/ Progress
    with Progress(
        TextColumn("[bold]{task.description} - {task.fields[aname]}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        MofNCompleteColumn(),
        transient=True,
    ) as progress:
        if archive_path.suffix == ".zip":
            with ZipFile(archive_path) as zf:
                tid = progress.add_task("Extracting", aname=archive_path.name, total=len(members := zf.infolist()))
                extract_path = Path(zf.extract(members[0], download_dir))
                if extract_type == "file":
                    assert (
                        len(members) == 1
                    ), f"Archive `{archive_path}` with extract type `{extract_type} has > 1 member!"
                elif extract_type == "directory":
                    for member in members[1:]:
                        zf.extract(member, download_dir)
                        progress.advance(tid)
                else:
                    raise ValueError(f"Extract type `{extract_type}` for archive `{archive_path}` is not defined!")

        elif archive_path.suffix in {".tar", ".gz"}:
            assert extract_type == "directory", f"Unexpected `{extract_type = }` for `tar` archive!"
            extract_path = download_dir / archive_path.stem.split(".")[0]
            with tarfile.open(archive_path) as tf:
                tid = progress.add_task("Extracting", aname=archive_path.name, total=len(members := tf.getmembers()))
                for member in members:
                    tf.extract(member=member, path=download_dir)
                    progress.advance(tid)

    # Cleanup (if specified)
    if cleanup:
        archive_path.unlink()

    return extract_path



def download_extract(dataset_family: str, root_dir: Path, hf_token: str) -> None:
    """Download all files for a given dataset (querying registry above), extracting archives if necessary."""
    os.makedirs(download_dir := root_dir / "download" / dataset_family, exist_ok=True)

    # Download Files => Single-Threaded, with Progress Bar
    dl_tasks = [d for d in DATASET_REGISTRY[dataset_family]["download"] if not (download_dir / d["name"]).exists()]
    for dl_task in dl_tasks:
        if dl_task.get("hf_download", False):
            dl_path = download_with_progress(dl_task["url"], download_dir, hf_token=hf_token)
        else:
            dl_path = download_with_progress(dl_task["url"], download_dir)

        # Extract Files (if specified)
        if dl_task["extract"]:
            dl_path = extract_with_progress(dl_path, download_dir, dl_task["extract_type"])

        # Rename Path --> dl_task["name"]
        if dl_task["do_rename"]:
            shutil.move(dl_path, download_dir / dl_task["name"])
