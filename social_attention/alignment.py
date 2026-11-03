"""Map speaker annotations (bounding boxes, transcript spans) to token positions.

Everything here is model-agnostic.  A model wrapper only has to tell us

* where the visual tokens of every frame live in ``input_ids`` and the token
  grid size of that frame (``FrameTokens``),
* the character-offset mapping of the question text and the position of its
  first token in ``input_ids``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

Span = Tuple[int, int]


@dataclass
class FrameTokens:
    """Visual tokens of one frame (or one temporal patch) inside ``input_ids``."""

    start: int  # position of the first visual token of this frame
    grid_h: int  # number of token rows
    grid_w: int  # number of token columns
    frame_index: int = 0  # index of the source frame (for per-frame boxes), e.g. 2t for Qwen2.5-VL's temporal patch t

    @property
    def end(self) -> int:  # exclusive
        return self.start + self.grid_h * self.grid_w


def box_to_cells(box: Sequence[float], grid_h: int, grid_w: int) -> Tuple[int, int, int, int]:
    """Normalised ``[x1, y1, x2, y2]`` box -> inclusive cell range ``(x1, x2, y1, y2)`` on a token grid."""
    x1, y1, x2, y2 = box
    cx1 = max(0, min(int(math.floor(x1 * grid_w)), grid_w - 1))
    cx2 = max(0, min(int(math.ceil(x2 * grid_w)) - 1, grid_w - 1))
    cy1 = max(0, min(int(math.floor(y1 * grid_h)), grid_h - 1))
    cy2 = max(0, min(int(math.ceil(y2 * grid_h)) - 1, grid_h - 1))
    return cx1, cx2, cy1, cy2


def speaker_visual_positions(
    speaker_boxes: Dict[str, Sequence[float]], frames: Sequence[FrameTokens]
) -> Dict[str, List[int]]:
    """Visual token positions (in ``input_ids``) covered by each speaker's box in every frame.

    ``speaker_boxes`` maps a speaker id to one normalised ``[x1, y1, x2, y2]``
    box (used for all frames) or to a list with one box per *source* frame,
    indexed by ``FrameTokens.frame_index``.  An empty or all-zero box means the
    speaker is not visible.
    """
    result: Dict[str, List[int]] = {}
    for speaker, box in speaker_boxes.items():
        if not box:
            continue
        per_frame = isinstance(box[0], (list, tuple))
        positions: List[int] = []
        for frame in frames:
            b = box[frame.frame_index] if per_frame else box
            if not b or all(v == 0 for v in b):
                continue
            cx1, cx2, cy1, cy2 = box_to_cells(b, frame.grid_h, frame.grid_w)
            for y in range(cy1, cy2 + 1):
                for x in range(cx1, cx2 + 1):
                    positions.append(frame.start + y * frame.grid_w + x)
        if positions:
            result[speaker] = positions
    return result


def char_spans_to_token_spans(
    char_spans: Sequence[Span], offsets: Sequence[Span], token_offset: int, char_offset: int = 0
) -> List[Span]:
    """Convert character spans of the question text into token spans of ``input_ids``.

    ``offsets`` is the ``offset_mapping`` returned by a fast tokenizer for the
    tokenised text block, ``token_offset`` the position of that block's first
    token in ``input_ids`` and ``char_offset`` the position of the question text
    inside the tokenised block (e.g. 1 if the block starts with a newline).
    """
    token_spans: List[Span] = []
    for start, end in char_spans:
        if start < 0 or end <= start:
            continue
        start, end = start + char_offset, end + char_offset
        tok_start = tok_end = None
        for i, (s, e) in enumerate(offsets):
            if tok_start is None and e > start:
                tok_start = i
            if tok_start is not None and s >= end:
                tok_end = i
                break
        if tok_start is None:
            continue
        if tok_end is None:
            tok_end = len(offsets)
        token_spans.append((tok_start + token_offset, tok_end + token_offset))
    return token_spans


def build_bias_and_mask(
    seq_len: int,
    speaker_visual: Dict[str, List[int]],
    speaker_tokens: Dict[str, List[Span]],
    all_visual_positions: Sequence[int],
    alpha: float,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the ``(1, 1, L, L)`` bias and cross-modal mask tensors for one sample.

    * ``bias[i, j] = alpha`` when text token ``i`` belongs to speaker ``s`` and
      visual token ``j`` lies inside ``s``'s box, else 0.
    * ``mm_mask[i, j] = True`` when ``i`` is a speaker text token and ``j`` is
      in ``all_visual_positions`` (the region over which head activity and the
      adaptive maximum are measured: all visual tokens, or the union of the
      speaker boxes).
    """
    bias = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    mm_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    visual = torch.as_tensor(list(all_visual_positions), dtype=torch.long)
    for speaker, spans in speaker_tokens.items():
        text_idx = [t for s, e in spans for t in range(s, min(e, seq_len))]
        if not text_idx:
            continue
        text_idx = torch.as_tensor(text_idx, dtype=torch.long)
        mm_mask[text_idx[:, None], visual[None, :]] = True
        vis = speaker_visual.get(speaker)
        if vis:
            vis = torch.as_tensor(vis, dtype=torch.long)
            bias[text_idx[:, None], vis[None, :]] = alpha
    return bias[None, None].to(device), mm_mask[None, None].to(device)
