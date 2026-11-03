"""InternVL3 (HF ``-hf`` checkpoints) wrapper."""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, InternVLForConditionalGeneration

from ..alignment import FrameTokens
from .base import EncodedSample, SocialMLLM


class InternVL3Social(SocialMLLM):
    """InternVL3-hf with social-aware attention.

    Every frame is resized to 448x448 and encoded into a 16x16 grid of 256 visual
    tokens (no dynamic tiling), using the ``Frame{i}: <img>...</img>`` video
    prompt format of InternVL.
    """

    default_system_prompt = (
        "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
    )

    # zero-shot defaults: lambda=1e-4 activates ~20% of the heads in layers 10-19 of InternVL3-8B
    default_bias_config = {"alpha": 0.5, "head_threshold": 1e-4, "layer_min": 10, "layer_max": 19}

    def __init__(self, model_name_or_path: str = "OpenGVLab/InternVL3-8B-hf", num_frames: int = 4, **kwargs):
        self.num_frames = num_frames  # frames actually fed to the model (uniformly sub-sampled)
        super().__init__(model_name_or_path, **kwargs)

    def _load(self, model_name_or_path):
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        model = InternVLForConditionalGeneration.from_pretrained(
            model_name_or_path, dtype=self.dtype, attn_implementation="sdpa"
        ).to(self.device)
        return processor, model

    @property
    def text_model(self):
        return self.model.model.language_model

    def encode(self, frames: Sequence[Image.Image], question: str) -> EncodedSample:
        frames = list(frames)
        idx = list(range(len(frames)))
        if self.num_frames and self.num_frames < len(frames):
            idx = [round(i * (len(frames) - 1) / (self.num_frames - 1)) for i in range(self.num_frames)]
            frames = [frames[i] for i in idx]
        pixel_values = self.processor.image_processor(images=frames, crop_to_patches=False, return_tensors="pt")[
            "pixel_values"
        ]
        proc = self.processor
        seq = proc.image_seq_length  # 256 tokens per frame
        grid = int(round(seq**0.5))
        tok = self.tokenizer
        img_start, img_ctx, img_end = proc.start_image_token_id, tok.convert_tokens_to_ids(proc.image_token), proc.end_image_token_id

        prefix = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n"
        vision_ids, frame_tokens, pos = [], [], len(tok(prefix, add_special_tokens=False)["input_ids"])
        nl = tok("\n", add_special_tokens=False)["input_ids"]
        for i in range(len(frames)):
            head = tok(f"Frame{i + 1}: ", add_special_tokens=False)["input_ids"]
            if i > 0:
                vision_ids += nl
                pos += len(nl)
            vision_ids += head + [img_start]
            pos += len(head) + 1
            frame_tokens.append(FrameTokens(pos, grid, grid, frame_index=idx[i]))
            vision_ids += [img_ctx] * seq + [img_end]
            pos += seq + 1
        text_block = "\n" + question
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        input_ids, _, offsets, text_token_offset = self._tokenize_chat(prefix, text_block, suffix, vision_ids)
        model_inputs = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
            "pixel_values": pixel_values.to(self.dtype),
        }
        return EncodedSample(model_inputs, frame_tokens, offsets, text_token_offset, text_char_offset=1)
