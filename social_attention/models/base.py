"""Common wrapper logic shared by the supported MLLMs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

from ..alignment import FrameTokens, build_bias_and_mask, char_spans_to_token_spans, speaker_visual_positions
from ..attention import SocialAttentionState, SocialBiasConfig, enable_social_attention


@dataclass
class EncodedSample:
    """Model inputs plus the bookkeeping needed to build the attention bias."""

    model_inputs: Dict[str, torch.Tensor]
    frames: List[FrameTokens]  # visual tokens of each frame / temporal patch
    text_offsets: List[tuple]  # offset_mapping of the tokenised text block
    text_token_offset: int  # position of the text block's first token in input_ids
    text_char_offset: int  # position of the question inside the text block


class SocialMLLM:
    """Base class: loads a model, builds inputs, attaches the social-aware bias and generates."""

    def __init__(
        self,
        model_name_or_path: str,
        bias_config: Optional[SocialBiasConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        system_prompt: Optional[str] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.bias_config = bias_config or SocialBiasConfig(**self.default_bias_config)
        self.system_prompt = system_prompt if system_prompt is not None else self.default_system_prompt
        self.processor, self.model = self._load(model_name_or_path)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.state: SocialAttentionState = enable_social_attention(self.text_model, self.bias_config)

    # ---- to be implemented by subclasses -------------------------------------------------
    default_system_prompt: str = ""
    default_bias_config: dict = {}  # zero-shot defaults (alpha, head_threshold, ...) calibrated per model

    def _load(self, model_name_or_path: str):
        raise NotImplementedError

    @property
    def text_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def encode(self, frames: Sequence[Image.Image], question: str) -> EncodedSample:
        raise NotImplementedError

    # ---- shared ---------------------------------------------------------------------------
    def _tokenize_chat(self, prefix: str, text_block: str, suffix: str, vision_ids: List[int]):
        """Tokenise ``prefix + <vision tokens> + text_block + suffix`` piecewise.

        Returns ``(input_ids, vision_start, text_offsets, text_token_offset)``.
        Tokenising the pieces separately keeps the character offsets of the text
        block exact; the pieces are separated by special tokens so BPE merges
        cannot cross the boundaries.
        """
        tok = self.tokenizer
        prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
        enc = tok(text_block, add_special_tokens=False, return_offsets_mapping=True)
        suffix_ids = tok(suffix, add_special_tokens=False)["input_ids"]
        vision_start = len(prefix_ids)
        text_token_offset = vision_start + len(vision_ids)
        input_ids = prefix_ids + vision_ids + enc["input_ids"] + suffix_ids
        return input_ids, vision_start, enc["offset_mapping"], text_token_offset

    def build_bias(self, encoded: EncodedSample, speaker_boxes: Dict, speaker_spans: Dict):
        seq_len = encoded.model_inputs["input_ids"].shape[-1]
        visual = speaker_visual_positions(speaker_boxes, encoded.frames)
        tokens = {
            sid: char_spans_to_token_spans(
                spans, encoded.text_offsets, encoded.text_token_offset, encoded.text_char_offset
            )
            for sid, spans in speaker_spans.items()
        }
        if self.bias_config.visual_region == "all":  # every visual token
            region = [p for f in encoded.frames for p in range(f.start, f.end)]
        else:  # V_all of the paper: union of all speaker boxes
            region = sorted({p for ps in visual.values() for p in ps})
        bias, mm_mask = build_bias_and_mask(seq_len, visual, tokens, region, self.bias_config.alpha, self.device)
        return bias, mm_mask, tokens

    def decode_token_spans(self, encoded: EncodedSample, token_spans: Dict[str, List[tuple]]) -> Dict[str, List[str]]:
        """Debug helper: the text covered by each speaker's token spans."""
        ids = encoded.model_inputs["input_ids"][0].tolist()
        return {sid: [self.tokenizer.decode(ids[s:e]) for s, e in spans] for sid, spans in token_spans.items()}

    @torch.no_grad()
    def generate(
        self,
        frames: Sequence[Image.Image],
        question: str,
        speaker_boxes: Optional[Dict] = None,
        speaker_spans: Optional[Dict] = None,
        use_bias: bool = True,
        max_new_tokens: int = 16,
        return_debug: bool = False,
        **generate_kwargs,
    ):
        """Answer ``question`` about ``frames``; with ``use_bias`` the social-aware bias is applied."""
        encoded = self.encode(frames, question)
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in encoded.model_inputs.items()}
        debug = {}
        if use_bias and (not speaker_boxes or not speaker_spans):
            warnings.warn("sample has no speaker boxes/spans; running without the bias")
            use_bias = False
        if use_bias:
            bias, mm_mask, token_spans = self.build_bias(encoded, speaker_boxes, speaker_spans)
            self.state.set(bias, mm_mask)
            if return_debug:
                debug["token_spans"] = self.decode_token_spans(encoded, token_spans)
        else:
            self.state.clear()
        try:
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, **generate_kwargs
            )
        finally:
            if self.bias_config.record_stats:
                debug["head_stats"] = list(self.state.stats)
            self.state.clear()
        new_tokens = out[0, inputs["input_ids"].shape[-1] :]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return (answer, debug) if return_debug else answer
