# VLM Evaluation
![](./images/03-evaluation-suite-med-res.png)

VLM Evaluation: Benchmark for VLMs, spanning text generation tasks from VQA to Captioning.

Built with [PyTorch](https://pytorch.org/), using sane quality defaults (`black`, `ruff`, `pre-commit`).

---

## Installation

This repository is built on top of PyTorch; while specified as a dependency for the package, we highly recommend that
you install the desired version of PyTorch (e.g., with accelerator support) for your given hardware and dependency
manager (e.g., `conda`). Otherwise, the default installed version be incompatible.

PyTorch installation instructions [can be found here](https://pytorch.org/get-started/locally/). This repository
requires PyTorch >= 2.1.0, but has only been thoroughly tested with PyTorch 2.1.0, Torchvision 0.16.0, Torchaudio 2.1.0.

Once PyTorch has been properly installed, you can install this package locally via an editable installation:

```bash
git clone https://github.com/TRI-ML/vlm-evaluation
cd vlm-evaluation
pip install -e .
```

Finally, make sure to copy your HuggingFace token to `.hf_token`.

## Usage

Prepare datasets for eval: `scripts/datasets/prepare.py`; model and evaluation dataset configs are defined in `vlm_eval/conf`

Entry Point: `scripts/evaluate.py`; model and evaluation dataset configs are defined in `vlm_eval/conf`. This script evaluates
a given model on the specified dataset

Interactive GUI: `scripts/interactive_demo.py` loads a trained model and creates a gradio style interactive demo.

Scoring: `scripts/score.py` Score an evaluated model.

## Example

First make sure you create the folders for evaluation datasets and results. For example:
`/home/ubuntu/datasets/vlm-evaluation`, `/home/ubuntu/prismatic-vlms/results`

(1) Prepare datasets for Text VQA:

`python scripts/datasets/prepare.py --dataset_family text-vqa`

where `dataset_family` can be selected from `[vqa-v2, gqa, vizwiz, text-vqa, refcoco, ocid-ref, tally-qa, pope, vsr]`

(2) Evaluate LLaVa 1.5 (7B) model and Prism 7B on Text VQA slim dataset:

`python scripts/evaluate.py --model_family llava-v15 --model_id llava-v1.5-7b --model_dir liuhaotian/llava-v1.5-7b --dataset.type text-vqa-slim --dataset.root_dir /home/ubuntu/datasets/vlm-evaluation`

For prismatic models you can either pass just a `model_id`:

`python scripts/evaluate.py --model_id prism-dinosiglip+7b --dataset.type text-vqa-slim --dataset.root_dir /home/ubuntu/datasets/vlm-evaluation`

or you can provide a path to a model directory with `--model_dir`. If a `model_dir` is provided, `model_id` will be ignored.

If you have multiple GPUs available:

`accelerate launch --num_processes=<NUM_GPUS> scripts/evaluate.py --model_family llava-v15 --model_id llava-v1.5-7b --model_dir liuhaotian/llava-v1.5-7b --dataset.type text-vqa-slim --dataset.root_dir /home/ubuntu/datasets/vlm-evaluation`

`accelerate launch --num_processes=<NUM_GPUS> scripts/evaluate.py --model_id prism-dinosiglip+7b --dataset.type text-vqa-slim --dataset.root_dir /home/ubuntu/datasets/vlm-evaluation`

You can evaluate any models trained in the accompanying prismatic-vlms codebase by modifying the model_dir, model_family, and model_id above accordingly

(3) Score LLaVa 1.5 (7B) Model and Prism 7B on Text VQA

`python scripts/score.py --model_id llava-v1.5-7b --dataset.type text-vqa-slim --dataset.root_dir /home/ubuntu/datasets/vlm-evaluation --results_dir /home/ubuntu/prismatic-vlms/results`

`python scripts/score.py --model_id prism-dinosiglip+7b --dataset.type text-vqa-slim --dataset.root_dir /home/ubuntu/datasets/vlm-evaluation --results_dir /home/ubuntu/prismatic-vlms/results`

(4) Play with Prism 7B Model in interactive GUI
Run the following scripts in separate terminals:

Launch gradio controller: 

`python -m vlm_eval.serve.controller --host 0.0.0.0 --port 10000`

Launch web server: 

`python -m vlm_eval.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload --share`

Launch interactive demo for Prism 7B Model: 

`python -m scripts.interactive_demo --port 40000 --model_id prism-dinosiglip+7b`

Launch interactive demo for LLaVA 1.5 7B Model: 

`python -m scripts.interactive_demo --port 40001 --model_family llava-v15 --model_id llava-v1.5-7b --model_dir liuhaotian/llava-v1.5-7b`

When running the demo, the following parameters are adjustable:
+ Temperature
+ Max output tokens

The default interaction mode is Chat, which is the main way to use our models. However, we also support a number of other 
interaction modes for more specific use cases:
+ Captioning: Here, you can simply upload an image with no provided prompt and the selected model will output a caption. Even if a prompt
is input by the user, it will not be used in producing the caption.
+ Bounding Box Prediction: After uploading an image, simply specify a portion of the image for which bounding box coordinates are desired
in the prompt and the selected model will output corresponding coordinates.
+ Visual Question Answering: Selecting this option is best when the user wants short, succint answers to a specific question provided in the
prompt.
+ True/False Question Answering: Selecting this option is best when the user wants a True/False answer to a specific question provided in the 
prompt.

## Contributing

Before committing to the repository, *make sure to set up your dev environment!*

Here are the basic development environment setup guidelines:

+ Fork/clone the repository, performing an editable installation. Make sure to install with the development dependencies
  (e.g., `pip install -e ".[dev]"`); this will install `black`, `ruff`, and `pre-commit`.

+ Install `pre-commit` hooks (`pre-commit install`).

+ Branch for the specific feature/issue, issuing PR against the upstream repository for review.

## Repository Structure

High-level overview of repository/project file-tree:

+ `vlm_eval` - Package source code; has all core utilities for task specification, model loading, and scoring.
+ `scripts/` - Standalone scripts for various functionality (e.g., training).
+ `.gitignore` - Default Python `.gitignore`.
+ `.pre-commit-config.yaml` - Pre-commit configuration file (sane defaults + `black` + `ruff`).
+ `LICENSE` - By default, research code is made available under the MIT License; if changing, think carefully about why!
+ `pyproject.toml` - Following PEP 621, this file has all project configuration details (including dependencies), as
                     well as tool configurations (for `black` and `ruff`).
+ `README.md` - You are here!
