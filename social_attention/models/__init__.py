from .base import SocialMLLM

MODELS = {
    "qwen2_5_vl": ("social_attention.models.qwen2_5_vl", "Qwen25VLSocial"),
    "internvl3": ("social_attention.models.internvl3", "InternVL3Social"),
}


def load_model(name: str, **kwargs) -> SocialMLLM:
    """Instantiate a wrapped MLLM by short name (``qwen2_5_vl`` or ``internvl3``)."""
    import importlib

    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}', choose from {list(MODELS)}")
    module, cls = MODELS[name]
    return getattr(importlib.import_module(module), cls)(**kwargs)


__all__ = ["SocialMLLM", "load_model", "MODELS"]
