"""Qwen2.5-VL wrapper."""

from __future__ import annotations

from typing import List, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ..alignment import FrameTokens
from ..data import resize_max_pixels
from .base import EncodedSample, SocialMLLM


class Qwen25VLSocial(SocialMLLM):
    """Qwen2.5-VL-Instruct with social-aware attention.

    Frames are fed as one video.  Qwen2.5-VL groups ``temporal_patch_size`` (=2)
    frames into one temporal patch, so 8 frames give 4 temporal patches; with
    per-frame boxes, temporal patch ``t`` uses the box of frame ``2t``.
    """

    default_system_prompt = "You are a helpful assistant."
    # zero-shot defaults: lambda=2e-4 activates ~20% of the heads in layers 10-19 of Qwen2.5-VL-7B
    default_bias_config = {"alpha": 0.5, "head_threshold": 2e-4, "layer_min": 10, "layer_max": 19}

    def __init__(self, model_name_or_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", max_pixels: int = 256 * 256, fps: float = 2.0, **kwargs):
        self.max_pixels = max_pixels  # per-frame pixel budget (controls the number of visual tokens)
        self.fps = fps  # nominal frame rate of the sampled frames (used by M-RoPE)
        super().__init__(model_name_or_path, **kwargs)

    def _load(self, model_name_or_path):
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, dtype=self.dtype, attn_implementation="sdpa"
        ).to(self.device)
        return processor, model

    @property
    def text_model(self):
        return self.model.model.language_model

    def encode(self, frames: Sequence[Image.Image], question: str) -> EncodedSample:
        frames = [resize_max_pixels(f, self.max_pixels) for f in frames]
        vp = self.processor.video_processor
        video_inputs = vp(
            videos=[frames],
            min_pixels=vp.size["shortest_edge"],
            max_pixels=self.max_pixels,
            do_sample_frames=False,
            return_tensors="pt",
        )
        grid_t, grid_h, grid_w = video_inputs["video_grid_thw"][0].tolist()
        merge = self.processor.image_processor.merge_size
        tok_h, tok_w = grid_h // merge, grid_w // merge
        n_video_tokens = grid_t * tok_h * tok_w

        cfg = self.model.config
        prefix = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n"
        vision_ids = [cfg.vision_start_token_id] + [cfg.video_token_id] * n_video_tokens + [cfg.vision_end_token_id]
        text_block = "\n" + question
        suffix = "<|im_end|>\n<|im_start|>assistant\n"
        input_ids, vision_start, offsets, text_token_offset = self._tokenize_chat(prefix, text_block, suffix, vision_ids)

        first_video_token = vision_start + 1
        frame_tokens = [
            FrameTokens(first_video_token + t * tok_h * tok_w, tok_h, tok_w, frame_index=t * vp.temporal_patch_size)
            for t in range(grid_t)
        ]
        model_inputs = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
            "pixel_values_videos": video_inputs["pixel_values_videos"],
            "video_grid_thw": video_inputs["video_grid_thw"],
            "second_per_grid_ts": [vp.temporal_patch_size / self.fps],
        }
        return EncodedSample(model_inputs, frame_tokens, offsets, text_token_offset, text_char_offset=1)
