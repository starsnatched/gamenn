import os
from typing import Optional

from .utils import ensure_dir


def merge_and_save(model_wrapper, tokenizer, out_dir: str) -> str:
    ensure_dir(out_dir)
    try:
        if hasattr(model_wrapper, "pretrained_model") and hasattr(model_wrapper.pretrained_model, "merge_and_unload"):
            model_wrapper.pretrained_model = model_wrapper.pretrained_model.merge_and_unload()
    except Exception:
        pass
    model_wrapper.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir

