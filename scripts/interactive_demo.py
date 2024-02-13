"""
interactive_demo.py

Entry point for all VLM-Evaluation interactive demos; specify model and get a gradio UI where you can chat with it!

This file is heavily adapted from the script used to serve models in the LLaVa repo:
https://github.com/haotian-liu/LLaVA/blob/main/llava/serve/model_worker.py. It is
modified to ensure compatibility with our Prismatic models.
"""
import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Union

import draccus
import requests
import torch
import uvicorn
from accelerate.utils import set_seed
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import StreamingResponse
from llava.constants import WORKER_HEART_BEAT_INTERVAL
from llava.mm_utils import load_image_from_base64
from llava.utils import server_error_msg
from torchvision.transforms import Compose

from vlm_eval.models import load_vlm
from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.serve import INTERACTION_MODES_MAP, MODEL_ID_TO_NAME

GB = 1 << 30
worker_id = str(uuid.uuid4())[:6]
global_counter = 0
model_semaphore = None


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class ModelWorker:
    def __init__(self, controller_addr, worker_addr, worker_id, no_register, vlm, model_name):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_name = model_name
        self.vlm = vlm

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        url = self.controller_addr + "/register_worker"
        data = {"worker_name": self.worker_addr, "check_heart_beat": True, "worker_status": self.get_status()}
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url, json={"worker_name": self.worker_addr, "queue_length": self.get_queue_length()}, timeout=5
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException:
                pass
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return (
                limit_model_concurrency
                - model_semaphore._value
                + (len(model_semaphore._waiters) if model_semaphore._waiters is not None else 0)
            )

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)

        temperature = params.get("temperature", 0.2)
        max_new_tokens = params.get("max_new_tokens", 2048)
        interaction_mode = INTERACTION_MODES_MAP[params.get("interaction_mode", "Chat")]

        if temperature != 0:
            self.vlm.set_generate_kwargs(
                {"do_sample": True, "max_new_tokens": max_new_tokens, "temperature": temperature}
            )
        else:
            self.vlm.set_generate_kwargs({"do_sample": False, "max_new_tokens": max_new_tokens})

        if images is not None and len(images) == 1:
            images = [load_image_from_base64(image) for image in images]
        else:
            raise NotImplementedError("Only supports queries with one image for now")

        if interaction_mode == "chat":
            question_prompt = [prompt]
        else:
            prompt_fn = self.vlm.get_prompt_fn(interaction_mode)
            if interaction_mode != "captioning":
                question_prompt = [prompt_fn(prompt)]
            else:
                question_prompt = [prompt_fn()]

        if isinstance(self.vlm.image_processor, Compose) or hasattr(self.vlm.image_processor, "is_prismatic"):
            # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
            pixel_values = self.vlm.image_processor(images[0].convert("RGB"))
        else:
            # Assume `image_transform` is a HF ImageProcessor...
            pixel_values = self.vlm.image_processor(images[0].convert("RGB"), return_tensors="pt")["pixel_values"][0]

        if type(pixel_values) is dict:
            for k in pixel_values.keys():
                pixel_values[k] = torch.unsqueeze(pixel_values[k].cuda(), 0)
        else:
            pixel_values = torch.unsqueeze(pixel_values.cuda(), 0)

        generated_text = self.vlm.generate_answer(pixel_values, question_prompt)[0]
        generated_text = generated_text.split("USER")[0].split("ASSISTANT")[0]
        yield json.dumps({"text": ori_prompt + generated_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)


@dataclass
class DemoConfig:
    # fmt: off

    # === Model Parameters =>> Prismatic ===
    model_family: str = "prismatic"           # Model family to load from in < `prismatic` | `llava-v15` | ... >
    model_id: str = "prism-dinosiglip+7b"     # Model ID to load and run (instance of `model_family`)
    model_dir: str = None                     # Can optionally supply model_dir instead of model_id

    # === Model Parameters =>> Official LLaVa ===
    # model_family: str = "llava-v15"
    # model_id: str = "llava-v1.5-13b"
    # model_dir: Path = "liuhaotian/llava-v1.5-13b"

    # === Model Parameters =>> Official InstructBLIP ===
    # model_family: str = "instruct-blip"
    # model_id: str = "instructblip-vicuna-7b"
    # model_dir: Path = "Salesforce/instructblip-vicuna-7b"

    # Model Worker Parameters
    host: str = "0.0.0.0"
    port: int = 40000
    controller_address: str = "http://localhost:10000"
    limit_model_concurrency: int = 5
    stream_interval: int = 1
    no_register: bool = False

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")  # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                  # Random Seed (for reproducibility)

    def __post_init__(self) -> None:
        self.run_dir = self.model_dir
        self.model_name = MODEL_ID_TO_NAME[str(self.model_id)]
        self.worker_address = f"http://localhost:{self.port}"

    # fmt: on


@draccus.wrap()
def interactive_demo(cfg: DemoConfig):
    # overwatch.info(f"Starting Evaluation for Dataset `{cfg.dataset.dataset_id}` w/ Model `{cfg.model_id}`")
    set_seed(cfg.seed)

    # Build the VLM --> Download/Load Pretrained Model from Checkpoint
    overwatch.info("Initializing VLM =>> Bundling Models, Image Processors, and Tokenizer")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    vlm = load_vlm(cfg.model_family, cfg.model_id, cfg.run_dir, hf_token=hf_token)

    global worker
    global limit_model_concurrency
    limit_model_concurrency = cfg.limit_model_concurrency
    worker = ModelWorker(
        cfg.controller_address, cfg.worker_address, worker_id, cfg.no_register, vlm, cfg.model_name
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    interactive_demo()
