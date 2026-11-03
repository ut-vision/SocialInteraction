"""Social-aware attention: dynamic cross-modal head selection + adaptive attention bias.

This module implements Sec. 4 of the paper as a drop-in attention function for
HuggingFace ``transformers`` models that use the ``AttentionInterface``
(Qwen2.5-VL, InternVL3 / Qwen2 text backbones, ...).

For a (batch, head) pair the function

1. computes the ordinary attention logits ``q k^T / sqrt(d)`` (+ causal mask),
2. measures the mean cross-modal attention from speaker text tokens to the
   visual tokens of ``V_all`` (``mm_mask``: by default the union of all speaker
   boxes); heads whose mean is above ``head_threshold`` (lambda in Eq. 3) are
   *active*,
3. for active heads adds an adaptive bias to the logits between a speaker's text
   tokens and the visual tokens inside that speaker's own bounding boxes.  The
   bias value for text token ``i`` is ``alpha * max_j logits[i, j]`` over
   ``j in V_all`` (Eq. 4), i.e. the strongest cross-modal interaction the token
   already has, scaled by ``alpha``,
4. applies softmax and multiplies with ``v`` as usual.

Layers outside ``[layer_min, layer_max]``, decoding steps (query length != bias
size) and modules without a bias attached simply fall back to SDPA attention, so
the patched model behaves exactly like the original model when no bias is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

ATTENTION_NAME = "social_aware"


@dataclass
class SocialBiasConfig:
    """Hyper-parameters of the social-aware attention bias."""

    alpha: float = 1.0  # bias strength (alpha in Eq. 4)
    head_threshold: float = 5e-5  # lambda in Eq. 3; heads whose mean cross-modal attention is above it are active
    layer_min: int = 10  # first transformer layer (inclusive) that receives the bias
    layer_max: int = 19  # last transformer layer (inclusive) that receives the bias
    visual_region: str = "boxes"  # V_all for head selection / adaptive max: "boxes" (union of speaker boxes, as in the paper) or "all" visual tokens
    record_stats: bool = False  # keep per-layer active-head ratios in ``SocialAttentionState.stats``


@dataclass
class SocialAttentionState:
    """Mutable state shared by all patched attention modules of one model."""

    config: SocialBiasConfig = field(default_factory=SocialBiasConfig)
    bias: Optional[torch.Tensor] = None  # (B, 1, L, L) float, alpha at (speaker text token, speaker visual token)
    mm_mask: Optional[torch.Tensor] = None  # (B, 1, L, L) bool, True at (speaker text token, visual token in V_all)
    stats: list = field(default_factory=list)  # per layer: {"layer", "active_ratio", "mean_cross_attn", "head_cross_attn"}

    def set(self, bias: torch.Tensor, mm_mask: torch.Tensor) -> None:
        self.bias = bias
        self.mm_mask = mm_mask
        self.stats = []

    def clear(self) -> None:
        self.bias = None
        self.mm_mask = None


# ----------------------------------------------------------------------------------------------
# Core of the method.  To port it to another model, this is the only function you need.
# ----------------------------------------------------------------------------------------------
def social_aware_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
):
    """Attention function registered under ``ATTENTION_NAME``.

    Shapes follow the ``transformers`` attention interface: ``query`` is
    ``(B, H, L_q, D)``, ``key``/``value`` are ``(B, H_kv, L_k, D)``.
    """
    state: Optional[SocialAttentionState] = getattr(module, "social_state", None)
    layer_idx = getattr(module, "layer_idx", None)
    q_len, k_len = query.shape[2], key.shape[2]

    use_bias = (
        state is not None
        and state.bias is not None
        and layer_idx is not None
        and state.config.layer_min <= layer_idx <= state.config.layer_max
        and q_len == k_len == state.bias.shape[-1]  # prefill step only
    )
    if not use_bias:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )

    cfg = state.config
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    B, H, L, D = query.shape
    scale = D**-0.5 if scaling is None else scaling

    # 1. original attention logits (float32 for numerical stability)
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale  # (B, H, L, L)
    if attention_mask is None:
        causal = torch.ones(L, L, dtype=torch.bool, device=logits.device).tril()
        logits = logits.masked_fill(~causal, float("-inf"))
    else:
        mask = attention_mask[:, :, :, :L]
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask, float("-inf"))
        else:
            logits = logits + mask.float()

    # 2. dynamic cross-modal head selection (Eq. 3)
    mm_mask = state.mm_mask.to(logits.device)  # (B, 1, L, L)
    probs = torch.softmax(logits, dim=-1)
    n_pairs = mm_mask.sum(dim=(-2, -1)).clamp(min=1).float()  # (B, 1)
    cross_attn_mean = probs.masked_fill(~mm_mask, 0.0).sum(dim=(-2, -1)) / n_pairs  # (B, H)
    active = cross_attn_mean > cfg.head_threshold  # (B, H)
    if cfg.record_stats:
        state.stats.append(
            {
                "layer": layer_idx,
                "active_ratio": active.float().mean().item(),
                "mean_cross_attn": cross_attn_mean.mean().item(),
                "head_cross_attn": cross_attn_mean[0].tolist(),
            }
        )

    # 3. adaptive social-aware bias (Eq. 4): alpha * max_{visual j} logits[i, j]
    row_max = logits.masked_fill(~mm_mask, float("-inf")).amax(dim=-1).clamp(min=0.0)  # (B, H, L)
    bias = state.bias.to(device=logits.device, dtype=logits.dtype)  # (B, 1, L, L), alpha where to boost
    logits = logits + active[:, :, None, None] * row_max[..., None] * bias

    # 4. standard softmax attention
    probs = torch.softmax(logits, dim=-1).to(query.dtype)
    probs = torch.nn.functional.dropout(probs, p=dropout, training=module.training)
    attn_output = torch.matmul(probs, value).transpose(1, 2).contiguous()  # (B, L, H, D)
    return attn_output, None


AttentionInterface.register(ATTENTION_NAME, social_aware_attention_forward)


def enable_social_attention(
    text_model: torch.nn.Module, config: Optional[SocialBiasConfig] = None
) -> SocialAttentionState:
    """Route all self-attention modules of ``text_model`` through the social-aware attention.

    ``text_model`` is the decoder-only language model inside an MLLM (e.g.
    ``model.model.language_model``).  Returns the shared ``SocialAttentionState``;
    call ``state.set(bias, mm_mask)`` before a forward/generate call and
    ``state.clear()`` afterwards.
    """
    state = SocialAttentionState(config=config or SocialBiasConfig())
    text_model.config._attn_implementation = ATTENTION_NAME
    n = 0
    for module in text_model.modules():
        if hasattr(module, "layer_idx") and hasattr(module, "q_proj"):
            module.social_state = state
            n += 1
    if n == 0:
        raise RuntimeError("No attention modules found in the given text model.")
    return state
