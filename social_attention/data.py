"""Sample format and loading helpers.

A sample is a JSON object::

    {
      "id": "mmsi_043_0009",
      "task": "T",                       # free-form tag, used for per-task accuracy
      "frames": ["frames/f1.jpg", ...],  # uniformly sampled video frames, in order
      "question": "Watch this video ... Your answer is:",
      "answer": "Speaker 0",             # optional ground truth
      "speaker_boxes": {"0": [x1, y1, x2, y2], ...},   # normalised xyxy, one per speaker
      "speaker_spans": {"0": [[start, end], ...], ...}  # character spans in ``question``
    }

Relative frame paths are resolved against the directory of the JSON file.
``speaker_spans`` tells which characters of the question belong to which speaker
(the speaker's own utterances including the "Speaker k :" prefix).  For the
common "Speaker k : utterance" transcript format they can be generated with
:func:`speaker_spans_from_transcript`.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence

from PIL import Image


def load_samples(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    for sample in samples:
        frames = sample.get("frames")
        if frames is None and "frames_dir" in sample:
            d = sample["frames_dir"] if os.path.isabs(sample["frames_dir"]) else os.path.join(base, sample["frames_dir"])
            frames = [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        sample["frames"] = [p if os.path.isabs(p) else os.path.join(base, p) for p in frames]
    return samples


def load_frames(paths: Sequence[str], num_frames: Optional[int] = None) -> List[Image.Image]:
    """Load frames and uniformly sub-sample them to ``num_frames`` if requested."""
    paths = list(paths)
    if num_frames is not None and num_frames < len(paths):
        idx = [round(i * (len(paths) - 1) / (num_frames - 1)) for i in range(num_frames)] if num_frames > 1 else [0]
        paths = [paths[i] for i in idx]
    return [Image.open(p).convert("RGB") for p in paths]


def resize_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    """Shrink ``image`` (keeping the aspect ratio) so that it has at most ``max_pixels`` pixels."""
    if image.width * image.height > max_pixels:
        f = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((max(1, int(image.width * f)), max(1, int(image.height * f))))
    return image


def speaker_spans_from_transcript(
    question: str, speaker_names: Optional[Dict[str, str]] = None
) -> Dict[str, List[List[int]]]:
    """Character spans of every ``"<name> : utterance"`` line in ``question``.

    ``speaker_names`` maps a speaker id (box key) to the name used in the text,
    e.g. ``{"0": "Speaker 0", "1": "Speaker 1"}`` or ``{"0": "Penny"}``.  If
    omitted, lines of the form ``"Speaker k : ..."`` are assigned to speaker
    ``k``.  Lines that also appear inside the instruction (e.g. "... of this
    sentence: Speaker 0 : ...") are matched as well.
    """
    spans: Dict[str, List[List[int]]] = {}
    if speaker_names is None:
        pattern = re.compile(r"Speaker (?P<id>\d+) : [^\n]*")
        for m in pattern.finditer(question):
            spans.setdefault(m.group("id"), []).append([m.start(), m.end()])
        return spans
    for sid, name in speaker_names.items():
        pattern = re.compile(re.escape(name) + r" : [^\n]*")
        for m in pattern.finditer(question):
            spans.setdefault(sid, []).append([m.start(), m.end()])
    return spans


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().replace("speaker", "speaker ").split())
