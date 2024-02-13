"""
instructblip.py

Class definition for the InstructBLIP VLM, wrapping utilities for VQA, image captioning, and (WIP) conditional
likelihood estimation. Only supports the Vicuna LLM backbones (no FLAN-T5).
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate import PartialState
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from vlm_eval.util.interfaces import VLM, ImageProcessor, Tokenizer

# Define InstructBLIP Mapping from Model ID --> HF Hub Path
INSTRUCTBLIP_MODELS = {"instructblip-vicuna-7b": "Salesforce/instructblip-vicuna-7b"}


class InstructBLIP(VLM):
    def __init__(
        self,
        model_family: str,
        model_id: str,
        run_dir: Path,
        load_precision: str = "bf16",
        ocr: bool = False,
        max_length: int = 128,
        temperature: float = 0.2,
        **_: str,
    ) -> None:
        self.model_family, self.model_id, self.hub_path = model_family, model_id, run_dir
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[load_precision]
        self.ocr = ocr

        # Get Distributed State
        self.distributed_state = PartialState()

        # Load Model on GPU(s) --> download if necessary via HF Hub
        self.model, self.text_img_processor, self.image_processor = self.load()

        # For Fair Evaluation against LLaVa/Quartz/IDEFICS --> greedy decoding:
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {"do_sample": False, "max_new_tokens": self.max_length, "temperature": self.temperature}

        # InstructBLIP Default Generation Configuration =>> Uses Beam Search (very slow!)
        #   => Ref: https://huggingface.co/Salesforce/instructblip-vicuna-7b#intended-uses--limitations
        # self.generate_kwargs = {
        #     "do_sample": False,
        #     "num_beams": 5,
        #     "max_length": self.max_length,
        #     "min_length": 1,
        #     "repetition_penalty": 1.5,
        #     "length_penalty": 1.0,
        #     "temperature": 1,
        # }

        # For computing likelihoods --> get tokens corresponding to "true", "false" and "yes", "no"
        self.string2idx = {}
        for trigger_string in ["true", "false", "yes", "no"]:
            token_idx_list = self.text_img_processor.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    def load(self) -> Tuple[nn.Module, Tokenizer, ImageProcessor]:
        """
        Loads model and processors (InstructBLIPProcessor contains Vicuna Tokenizer, Q-Former Tokenizer, and an
        ImageProcessor) using the HF `InstructBLIP*.from_pretrained()` functionality.
        """
        with self.distributed_state.main_process_first():
            text_img_processor = InstructBlipProcessor.from_pretrained(self.hub_path)
            model = InstructBlipForConditionalGeneration.from_pretrained(self.hub_path)

        # Lift `image_processor` for use in evaluation harnesses
        image_processor = text_img_processor.image_processor

        # Place Model on Device
        model = model.to(self.distributed_state.device, dtype=self.dtype)
        model.eval()

        return model, text_img_processor, image_processor

    def set_generate_kwargs(self, generate_kwargs):
        self.generate_kwargs = generate_kwargs

    def get_prompt_fn(self, dataset_family: str = "vqa-v2") -> Callable[[str], str]:
        vqa_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=False)
        vqa_uncertain_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=True)
        true_false_prompt_fn = self.get_true_false_chat_prompt_fn()
        contrast_caption_prompt_fn = self.get_contrast_caption_chat_prompt_fn()
        bbox_refer_prompt_fn = self.get_bbox_refer_chat_prompt_fn()
        text_vqa_prompt_fn = self.get_vqa_chat_prompt_fn(uncertainty_aware=False, ocr_handling=True)
        captioning_prompt_fn = self.get_captioning_prompt_fn()

        return {
            "vqa-v2": vqa_prompt_fn,
            "gqa": vqa_prompt_fn,
            "vizwiz": vqa_uncertain_prompt_fn,
            "text-vqa": text_vqa_prompt_fn,
            "vsr": true_false_prompt_fn,
            "pope": vqa_prompt_fn,
            "refcoco": bbox_refer_prompt_fn,
            "ocid-ref": bbox_refer_prompt_fn,
            # Generic for GUI
            "captioning": captioning_prompt_fn,
            "bbox_pred": bbox_refer_prompt_fn,
            "vqa": vqa_prompt_fn,
            "true_false": true_false_prompt_fn,
        }[dataset_family]

    @staticmethod
    def get_captioning_prompt_fn() -> Callable[[str], str]:
        """Generates the full reference prompt for captioning tasks."""

        def captioning_prompt_fn() -> str:
            return "A short image description:"

        return captioning_prompt_fn

    @staticmethod
    def get_vqa_chat_prompt_fn(uncertainty_aware: bool = False, ocr_handling: bool = False) -> Callable[[str], str]:
        """Generates the full reference prompt for VQA tasks."""

        def vqa_prompt_fn(question: str) -> str:
            if not ocr_handling:
                return f"Question: {question} Short answer:"
            else:
                q_maybe_ocr = question.split("\nReference OCR token: ")
                if len(q_maybe_ocr) == 1:
                    return f"Question: {q_maybe_ocr[0]} Short answer:"
                else:
                    q, ocr_tokens = q_maybe_ocr
                    return f"OCR tokens: {ocr_tokens}. Question: {q} Short answer:"

        return vqa_prompt_fn

    @staticmethod
    def get_true_false_chat_prompt_fn() -> Callable[[str], str]:
        """Generates the full reference prompt for a True/False captioning task."""

        def true_false_prompt_fn(caption: str) -> str:
            return f'Based on the image, is this statement true or false? "{caption}" Answer:'

        return true_false_prompt_fn

    @staticmethod
    def get_contrast_caption_chat_prompt_fn() -> Callable[[str], str]:
        """Generates the full reference prompt for a multi-pair contrast captioning task (e.g., WinoGround)."""

        def contrast_caption_prompt_fn(caption: str) -> str:
            return f'Does the following caption match the image (true or false)? Caption: "{caption}" Answer:'

        return contrast_caption_prompt_fn

    @staticmethod
    def get_bbox_refer_chat_prompt_fn() -> Callable[[str], str]:
        """Generates the full reference prompt for a referring expression localization task."""

        def bbox_refer_prompt_fn(sentence: str) -> str:
            return f'Please provide the bounding box coordinate of the region this sentence describes: "{sentence}":'

        return bbox_refer_prompt_fn

    @torch.inference_mode()
    def generate_answer(
        self, pixel_values: torch.Tensor, questions: List[str], return_string_probabilities: Optional[List[str]] = None
    ) -> Union[List[str], List[List[float]]]:
        with torch.cuda.amp.autocast(dtype=torch.float32):
            batch_input_ids = [
                self.text_img_processor(text=q, return_tensors="pt").to(pixel_values.device) for q in questions
            ]

            # InstructBLIP "Custom" Decoding
            gen_texts, gen_probabilities = [], []
            for idx in range(len(batch_input_ids)):
                if return_string_probabilities is None:
                    full_out_ids = self.model.generate(
                        pixel_values=pixel_values[idx][None, ...],
                        qformer_input_ids=batch_input_ids[idx]["qformer_input_ids"],
                        qformer_attention_mask=batch_input_ids[idx]["qformer_attention_mask"],
                        input_ids=batch_input_ids[idx]["input_ids"],
                        attention_mask=batch_input_ids[idx]["attention_mask"],
                        **self.generate_kwargs,
                    )
                    gen_ids = full_out_ids[0]

                    # Decode `gen_ids` and strip and <EOS> tokens
                    gen_texts.append(self.text_img_processor.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    # InstructBLIP will by default output lowercase answers...
                    return_string_probabilities = [s.lower() for s in return_string_probabilities]

                    full_out_dict = self.model.generate(
                        pixel_values=pixel_values[idx][None, ...],
                        qformer_input_ids=batch_input_ids[idx]["qformer_input_ids"],
                        qformer_attention_mask=batch_input_ids[idx]["qformer_attention_mask"],
                        input_ids=batch_input_ids[idx]["input_ids"],
                        attention_mask=batch_input_ids[idx]["attention_mask"],
                        output_scores=True,
                        return_dict_in_generate=True,
                        **self.generate_kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for true/false and yes/no generations
                    gen_ids = full_out_dict.sequences[0]

                    # Decode `gen_ids` and strip and <EOS> tokens
                    gen_texts.append(self.text_img_processor.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_string_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities
