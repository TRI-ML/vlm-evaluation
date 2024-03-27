"""
prismatic.py

Class definition for wrapping arbitrary "Prismatic VLMs" (including replications of LLaVa, BLIP-2, etc.), wrapping
utilities for VQA, image captioning, and (WIP) conditional likelihood estimation.
"""
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import PartialState
from PIL.Image import Image
from prismatic import load

from vlm_eval.util.interfaces import VLM, ImageProcessor, Tokenizer


class PrismaticVLM(VLM):
    def __init__(
        self,
        model_family: str,
        model_id: str,
        run_dir: Path,
        hf_token: str,
        load_precision: str = "bf16",
        ocr: bool = False,
        max_length: int = 128,
        temperature: float = 1.0,
        **_: str,
    ) -> None:
        self.model_family, self.model_id, self.run_dir = model_family, model_id, run_dir
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[load_precision]
        self.hf_token = hf_token
        self.ocr = ocr

        # Get Distributed State
        self.distributed_state = PartialState()

        # Load Model on GPU(s)
        self.model, self.tokenizer, self.image_processor = self.load()

        # Set Default VQA Generation Configuration
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {"do_sample": False, "max_new_tokens": self.max_length, "temperature": self.temperature}

    def load(self) -> Tuple[nn.Module, Tokenizer, ImageProcessor]:
        """Load a Prismatic/Quartz Model using the default `prisma.load_pretrained_vlm` initializer."""

        if self.run_dir is not None:
            load_from = self.run_dir
        elif self.model_id is not None:
            load_from = self.model_id
        else:
            raise ValueError("Model Dir and ID cannot both be None")

        # Get Fully Initialized VLM Instance (+ handle `load_precision`)
        vlm = load(load_from, hf_token=self.hf_token)
        vlm.to(self.distributed_state.device, dtype=self.dtype)

        # Get Tokenizer and Image Processor
        tokenizer, image_transform = vlm.llm_backbone.tokenizer, vlm.vision_backbone.image_transform
        return vlm, tokenizer, image_transform

    def set_generate_kwargs(self, generate_kwargs):
        self.generate_kwargs = generate_kwargs

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> Any:
        return self.model.get_prompt_builder(system_prompt)

    def get_prompt_fn(self, dataset_family: str = "vqa-v2") -> Callable[[str], str]:
        vqa_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=False)
        vqa_uncertain_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=True)
        true_false_prompt_fn = self.get_true_false_chat_prompt_fn()
        contrast_caption_prompt_fn = self.get_contrast_caption_chat_prompt_fn()
        bbox_refer_prompt_fn = self.get_bbox_refer_chat_prompt_fn()
        text_vqa_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=False)
        captioning_prompt_fn = self.get_captioning_prompt_fn()
        tally_qa_prompt_fn = self.get_mc_prompt_fn()
        ai2d_prompt_fn = self.get_mc_prompt_fn()

        return {
            "vqa-v2": vqa_prompt_fn,
            "gqa": vqa_prompt_fn,
            "vizwiz": vqa_uncertain_prompt_fn,
            "text-vqa": text_vqa_prompt_fn,
            "vsr": true_false_prompt_fn,
            "pope": vqa_prompt_fn,
            "tally-qa": tally_qa_prompt_fn,
            "refcoco": bbox_refer_prompt_fn,
            "ocid-ref": bbox_refer_prompt_fn,
            "ai2d": ai2d_prompt_fn,
            # Generic for GUI
            "captioning": captioning_prompt_fn,
            "bbox_pred": bbox_refer_prompt_fn,
            "vqa": vqa_prompt_fn,
            "true_false": true_false_prompt_fn,
        }[dataset_family]

    def get_captioning_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for captioning tasks."""
        prompt_builder_fn = self.model.get_prompt_builder

        def captioning_prompt_fn() -> str:
            # Use Default Prompt (same as LLaVa-v1.5)
            prompt_builder = prompt_builder_fn()
            cap_prompt = "\nProvide a short image description."

            # Add to Prompt Builder
            prompt_builder.add_turn(role="human", message=cap_prompt)

            return prompt_builder.get_prompt()

        return captioning_prompt_fn

    def get_vqa_chat_prompt_fn(self, uncertainty_aware: bool = False) -> Callable[[str], str]:
        """Generates the full reference prompt for VQA tasks."""
        prompt_builder_fn = self.model.get_prompt_builder

        def vqa_prompt_fn(question: str) -> str:
            # Use Default Prompt (same as LLaVa-v1.5)
            prompt_builder = prompt_builder_fn()
            q_prompt = f"\n{question}"

            # For some evaluation such as VizWiz, models are expected to output "unanswerable" when questions are
            # ambiguous --> LLaVa 1.5 handles this by injecting the following "trigger phrase" into the prompt.
            if uncertainty_aware:
                q_prompt += "\nWhen the provided information is insufficient, respond with 'Unanswerable'."
                q_prompt += "\nAnswer the question using a single word or phrase."

            # Otherwise, LLaVa-1.5 encourages short VQA responses by default.
            else:
                q_prompt += "\nAnswer the question using a single word or phrase."

            # Add to Prompt Builder
            prompt_builder.add_turn(role="human", message=q_prompt)

            return prompt_builder.get_prompt()

        return vqa_prompt_fn

    def get_true_false_chat_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a True/False captioning task."""
        prompt_builder_fn = self.model.get_prompt_builder

        def true_false_prompt_fn(caption: str) -> str:
            prompt_builder = prompt_builder_fn()
            cap_prompt = f'\nBased on the image, is this statement "True" or "False"? {caption}'
            cap_prompt += '\nRespond with "True" or "False" directly.'

            # Add to Prompt Builder
            prompt_builder.add_turn(role="human", message=cap_prompt)

            return prompt_builder.get_prompt()

        return true_false_prompt_fn

    def get_contrast_caption_chat_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a multi-pair contrast captioning task (e.g., WinoGround)."""
        prompt_builder_fn = self.model.get_prompt_builder

        def contrast_caption_prompt_fn(caption: str) -> str:
            # Use Default Prompt (same as LLaVa-v1.5)
            prompt_builder = prompt_builder_fn()
            cap_prompt = f'\nDoes the following caption match the image? Caption: "{caption}"'
            cap_prompt += '\nRespond with "True" or "False" directly.'

            # Add to Prompt Builder
            prompt_builder.add_turn(role="human", message=cap_prompt)

            return prompt_builder.get_prompt()

        return contrast_caption_prompt_fn

    def get_mc_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a multiple choice question-answering task."""
        prompt_builder_fn = self.model.get_prompt_builder

        def mc_prompt_fn(question: str, choices: List[str]) -> str:
            # Create Choice String
            assert len(choices) <= 26, "Too many answer choices vs. possible letters in the alphabet!"
            choice_str = "\n".join([f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)])

            # Use Default Prompt (same as LLaVa-v1.5)
            prompt_builder = prompt_builder_fn()
            q_prompt = f"\n{question}\n{choice_str}"

            # Multiple Choice Trigger
            q_prompt += "\nAnswer with the option's letter from the given choices directly."

            # Add to Prompt Builder
            prompt_builder.add_turn(role="human", message=q_prompt)

            return prompt_builder.get_prompt()

        return mc_prompt_fn

    def get_bbox_refer_chat_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a referring expression localization task."""
        prompt_builder_fn = self.model.get_prompt_builder

        def bbox_refer_prompt_fn(sentence: str) -> str:
            # Use Default Prompt (same as LLaVa-v1.5)
            prompt_builder = prompt_builder_fn()
            detect_prompt = (
                f'\nPlease provide the bounding box coordinate of the region this sentence describes: "{sentence}"'
            )

            # Add to Prompt Builder
            prompt_builder.add_turn(role="human", message=detect_prompt)

            return prompt_builder.get_prompt()

        return bbox_refer_prompt_fn

    @torch.inference_mode()
    def generate_answer(
        self,
        pixel_values: torch.Tensor,
        question_prompts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
    ) -> Union[List[str], Tuple[List[str], List[List[float]]]]:
        return self.model.generate_batch(
            pixel_values, question_prompts, return_string_probabilities, **self.generate_kwargs
        )

    @torch.inference_mode()
    def generate(
        self,
        image: Image,
        input_text: str,
        do_sample: bool = False,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        min_length: int = 1,
    ) -> str:
        return self.model.generate(
            image,
            input_text,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
        )
