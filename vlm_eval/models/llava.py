"""
llava.py

Class definition for the LLaVa VLM, wrapping utilities for VQA, image captioning, and (WIP) conditional likelihood
estimation.

Reference: https://github.com/haotian-liu/LLaVA/tree/main
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from accelerate import PartialState
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from PIL.Image import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from vlm_eval.util.interfaces import VLM, ImageProcessor, Tokenizer

# Define LLaVa Mapping from Model ID --> HF Hub Path
LLAVA_MODELS = {
    # fmt: off

    # LLaVa v1 (no "academic VQA" pretraining; lower-resolution ViT-L/14 @ 224x224 OpenAI CLIP Backbone)
    "llava-lightning+clip+llama-2-13b": "liuhaotian/llava-llama-2-13b-chat-lightning-preview",

    # LLaVa v1.5 ("academic VQA" + ViT-L/14 @ 336x336 OpenAI CLIP Backbone)
    "llava-v1.5-7b": "liuhaotian/llava-v1.5-7b",
    "llava-v1.5-13b": "liuhaotian/llava-v1.5-13b",
    # fmt: on
}


# === LLaVa Letterbox/Pad Image Transforms ===
@dataclass
class LetterboxSquarePad:
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return F.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")


@dataclass
class LLaVaV15ImageTransform:
    default_image_processor: CLIPImageProcessor

    def __post_init__(self) -> None:
        self.letterbox = LetterboxSquarePad(tuple([int(255 * x) for x in self.default_image_processor.image_mean]))

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        padded_img = self.letterbox(img)
        return self.default_image_processor(padded_img, **kwargs)


class LLaVa(VLM):
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
        self.model, self.tokenizer, self.image_processor = self.load()

        # LLaVa is a chat-based model --> Load Chat-Specific VQA Prompts following LLaVa SciQA
        #   Ref: https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py#L29
        self.conv_mode = {
            "llava-v1.5-7b": "vicuna_v1",
            "llava-v1.5-13b": "vicuna_v1",
            "llava-v1.5-7b+x7": "vicuna_v1",
            "llava-v1.5-13b+x7": "vicuna_v1",
            "llava-v1.5-7b+x9": "vicuna_v1",
            "llava-v1.5-13b+x9": "vicuna_v1",
            "llava-v1.5-7b+x11": "vicuna_v1",
            "llava-v1.5-13b+x11": "vicuna_v1",
        }[self.model_id]
        self.conv = conv_templates[self.conv_mode].copy()

        # Set Default Generation Configuration --> again from the Github Repository!
        self.max_length = max_length
        self.temperature = temperature
        self.generate_kwargs = {"do_sample": False, "max_new_tokens": self.max_length, "temperature": self.temperature}

        # For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    def load(self) -> Tuple[nn.Module, Tokenizer, ImageProcessor]:
        """
        Loads model using a combination of `transformers.AutoModelForCausalLM` along with the special
        `LlavaLlamaForCausalLM` class defined in the `LLaVa` package.

        Using this instead of the default `LLaVa.load_pretrained_model` to remove bloat & patch image processor.

        Reference: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/builder.py
        """

        # Download & Configure Tokenizer, Model
        #   =>> Note :: Set Tokenizer `legacy=True` for LLaVa since it used Transformer pre-PR #24565
        #               Basically, the "old" version of Transformers has an issue where extra spaces/tokens get inserted
        #               after special tokens (in a consistent way). This is *wrong* but in a *minor* way; however,
        #               the HF Transformers folks have made the decision to keep this wrong behavior as default, and
        #               throw a scary warning...
        #                   => See: https://github.com/huggingface/transformers/pull/24565#issuecomment-1680314450
        #
        #               TL;DR --> Set `legacy=True` for *existing* LLaVa models (just ignore the warning)!
        with self.distributed_state.main_process_first():
            tokenizer = AutoTokenizer.from_pretrained(self.hub_path, use_fast=False, legacy=True)
            model = LlavaLlamaForCausalLM.from_pretrained(self.hub_path, torch_dtype=self.dtype)

            # Handle Vision Tower
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
                vision_tower.to(dtype=self.dtype)

        # Get Image Processor (can only happen after Vision Tower has been loaded) =>> Handle "pad"
        image_processor = vision_tower.image_processor
        if self.model_id in {
            "llava-v1.5-7b",
            "llava-v1.5-7b+x7",
            "llava-v1.5-7b+x9",
            "llava-v1.5-7b+x11",
            "llava-v1.5-13b",
            "llava-v1.5-13b+x7",
            "llava-v1.5-13b+x9",
            "llava-v1.5-13b+x11",
        }:
            image_processor = LLaVaV15ImageTransform(default_image_processor=image_processor)

        # Handle Special Tokens (for Inference)
        tokenizer.padding_side = "left"
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", False)

        # Assert both False --> no change for LLaVa-LLaMa-13B!
        #   => For other models, we might need to handle differently...
        assert not mm_use_im_start_end and not mm_use_im_patch_token, f"Check special tokens for `{self.model_id}`"

        # Load both the `model` and `vision_tower` onto the correct devices/in the correct precision!
        model, vision_tower = model.to(self.distributed_state.device), vision_tower.to(self.distributed_state.device)
        model.eval()

        return model, tokenizer, image_processor

    def set_generate_kwargs(self, generate_kwargs):
        self.generate_kwargs = generate_kwargs

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

        # Conversation manager `self.conv` is not stateless! Need to reset on each construction!
        self.conv = conv_templates[self.conv_mode].copy()

        # Different LLaVa Models handle <IMAGE> token insertion differently; we support both LLaVa v1 and v1.5!
        #   => Ref (v1): https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py#L53
        #   => Ref (v1.5): https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#evaluate-on-custom-datasets
        q_prompt = DEFAULT_IMAGE_TOKEN + "\n"
        if self.model_id.startswith("llava-v1.5"):
            q_prompt += "\nProvide a short image description."

        # Derive the full `vqa_prompt` following the logic from LLaVa/LLaMa (insert <SYS> and <INST> role tags)
        self.conv.append_message(self.conv.roles[0], q_prompt)
        self.conv.append_message(self.conv.roles[1], None)

        # Get full chat prompt template function --> insert question with `template.format(question=<QUESTION>)`
        prompt_template = self.conv.get_prompt()

        def llava_cap_prompt_fn() -> str:
            return prompt_template

        return llava_cap_prompt_fn

    def get_vqa_chat_prompt_fn(self, uncertainty_aware: bool = False) -> Callable[[str], str]:
        """Generates the full reference prompt for VQA tasks."""

        # Conversation manager `self.conv` is not stateless! Need to reset on each construction!
        self.conv = conv_templates[self.conv_mode].copy()

        # Different LLaVa Models handle <IMAGE> token insertion differently; we support both LLaVa v1 and v1.5!
        #   => Ref (v1): https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py#L53
        #   => Ref (v1.5): https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#evaluate-on-custom-datasets
        q_prompt = DEFAULT_IMAGE_TOKEN + "\n" + "{question}"
        if self.model_id.startswith("llava-v1.5"):
            # For some evaluation such as VizWiz, models are expected to output "unanswerable" when questions are
            # ambiguous --> LLaVa 1.5 handles this by injecting the following "trigger phrase" into the prompt.
            if uncertainty_aware:
                q_prompt += "\nWhen the provided information is insufficient, respond with 'Unanswerable'."
                q_prompt += "\nAnswer the question using a single word or phrase."

            # Otherwise, LLaVa-1.5 encourages short VQA responses by default.
            else:
                q_prompt += "\nAnswer the question using a single word or phrase."

        # Derive the full `vqa_prompt` following the logic from LLaVa/LLaMa (insert <SYS> and <INST> role tags)
        self.conv.append_message(self.conv.roles[0], q_prompt)
        self.conv.append_message(self.conv.roles[1], None)

        # Get full chat prompt template function --> insert question with `template.format(question=<QUESTION>)`
        prompt_template = self.conv.get_prompt()

        def llava_vqa_prompt_fn(question: str) -> str:
            return prompt_template.format(question=question)

        return llava_vqa_prompt_fn

    def get_true_false_chat_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a True/False captioning task."""

        # Conversation manager `self.conv` is not stateless! Need to reset on each construction!
        self.conv = conv_templates[self.conv_mode].copy()

        # Construct True/False Prompt =>> Following InstructBLIP
        cap_prompt = DEFAULT_IMAGE_TOKEN + '\nBased on the image, is this statement "True" or "False"?' + " {caption}"
        cap_prompt += '\nRespond with "True" or "False" directly.'

        # Derive the full prompt following the logic from LLaVa/LLaMa (insert <SYS> and <INST> role tags)
        self.conv.append_message(self.conv.roles[0], cap_prompt)
        self.conv.append_message(self.conv.roles[1], None)

        # Get full chat prompt template function --> insert caption with `template.format(caption=<CAPTION>)`
        prompt_template = self.conv.get_prompt()

        def llava_true_false_prompt_fn(caption: str) -> str:
            return prompt_template.format(caption=caption)

        return llava_true_false_prompt_fn

    def get_contrast_caption_chat_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a multi-pair contrast captioning task (e.g., WinoGround)."""

        # Conversation manager `self.conv` is not stateless! Need to reset on each construction!
        self.conv = conv_templates[self.conv_mode].copy()

        # Construct True/False Prompt =>> Following InstructBLIP
        cap_prompt = DEFAULT_IMAGE_TOKEN + '\nDoes the following caption match the image? Caption: "{caption}"'
        cap_prompt += '\nRespond with "True" or "False" directly.'

        # Derive the full prompt following the logic from LLaVa/LLaMa (insert <SYS> and <INST> role tags)
        self.conv.append_message(self.conv.roles[0], cap_prompt)
        self.conv.append_message(self.conv.roles[1], None)

        # Get full chat prompt template function --> insert caption with `template.format(caption=<CAPTION>)`
        prompt_template = self.conv.get_prompt()

        def llava_contrast_caption_prompt_fn(caption: str) -> str:
            return prompt_template.format(caption=caption)

        return llava_contrast_caption_prompt_fn

    def get_mc_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a multiple-choice question-answer task."""

        # Conversation manager `self.conv` is not stateless! Need to reset on each construction!
        self.conv = conv_templates[self.conv_mode].copy()

        # Different LLaVa Models handle <IMAGE> token insertion differently; we support both LLaVa v1 and v1.5!
        #   => Ref (v1): https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py#L53
        #   => Ref (v1.5): https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md#evaluate-on-custom-datasets
        q_prompt = DEFAULT_IMAGE_TOKEN + "\n" + "{question}\n{choice_str}"
        if self.model_id.startswith("llava-v1.5"):
            q_prompt += "\nAnswer with the option's letter from the given choices directly."

        # Derive the full `vqa_prompt` following the logic from LLaVa/LLaMa (insert <SYS> and <INST> role tags)
        self.conv.append_message(self.conv.roles[0], q_prompt)
        self.conv.append_message(self.conv.roles[1], None)

        # Get full chat prompt template function --> insert question with `template.format(question=<QUESTION>)`
        prompt_template = self.conv.get_prompt()

        def llava_mc_prompt_fn(question: str, choices: List[str]) -> str:
            assert len(choices) <= 26, "Too many answer choices vs. possible letters in the alphabet!"
            choice_str = "\n".join([f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices)])

            return prompt_template.format(question=question, choice_str=choice_str)

        return llava_mc_prompt_fn

    def get_bbox_refer_chat_prompt_fn(self) -> Callable[[str], str]:
        """Generates the full reference prompt for a referring expression localization task."""

        # Conversation manager `self.conv` is not stateless! Need to reset on each construction!
        self.conv = conv_templates[self.conv_mode].copy()

        # Construct Detection Prompt =>> Following LLaVa-1.5 Paper
        detect_prompt = (
            DEFAULT_IMAGE_TOKEN
            + '\nPlease provide the bounding box coordinate of the region this sentence describes: "{sentence}"'
        )

        # Derive the full prompt following the logic from LLaVa/LLaMa (insert <SYS> and <INST> role tags)
        self.conv.append_message(self.conv.roles[0], detect_prompt)
        self.conv.append_message(self.conv.roles[1], None)

        # Get full chat prompt template function --> insert sentence with `template.format(sentence=<SENTENCE>)`
        prompt_template = self.conv.get_prompt()

        def llava_bbox_refer_prompt_fn(sentence: str) -> str:
            return prompt_template.format(sentence=sentence)

        return llava_bbox_refer_prompt_fn

    @torch.inference_mode()
    def generate_answer(
        self, pixel_values: torch.Tensor, questions: List[str], return_string_probabilities: Optional[List[str]] = None
    ) -> Union[List[str], List[List[float]]]:
        # By default, LLaVa code only neatly handles processing a single example at a time, due to the way the <image>
        # tokens are interleaved with the text; this code just loops over inputs (naive padding doesn't work...)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            batch_input_ids = [
                tokenizer_image_token(q, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(pixel_values.device)
                for q in questions
            ]

            # Greedy Decoding
            gen_texts, gen_probabilities = [], []
            for idx, input_ids in enumerate(batch_input_ids):
                if return_string_probabilities is None:
                    full_out_ids = self.model.generate(
                        input_ids[None, ...], images=pixel_values[idx][None, ...], **self.generate_kwargs
                    )
                    gen_ids = full_out_ids[0, input_ids.shape[0] :]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = self.model.generate(
                        input_ids[None, ...],
                        images=pixel_values[idx][None, ...],
                        output_scores=True,
                        return_dict_in_generate=True,
                        **self.generate_kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[0] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_string_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.inference_mode()
    def generate(
        self,
        image: Image,
        input_text: str,
        do_sample: bool = True,
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        min_length: int = 1,
        length_penalty: float = 0,
    ) -> str:
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0].to(
            self.distributed_state.device
        )
        with torch.cuda.amp.autocast(dtype=self.dtype):
            input_ids = tokenizer_image_token(input_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(
                pixel_values.device
            )

            # Generate Subject to Keyword Arguments
            gen_ids = self.model.generate(
                input_ids[None, ...],
                images=pixel_values[None, ...],
                do_sample=do_sample,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                length_penalty=length_penalty,
            )
            gen_text = self.tokenizer.decode(gen_ids[0, input_ids.shape[0] :], skip_special_tokens=True).strip()

        return gen_text
