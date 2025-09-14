import os


def resolve_hf_token() -> str | None:
    t = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t
    try:
        from huggingface_hub import get_token

        tk = get_token()
        if tk:
            return tk
    except Exception:
        pass
    try:
        from huggingface_hub import HfFolder

        tk = HfFolder.get_token()
        if tk:
            return tk
    except Exception:
        pass
    return None


def format_auth_error(model_id: str) -> str:
    return (
        f"Access to model '{model_id}' requires Hugging Face authentication. "
        f"Set 'HF_TOKEN' or 'HUGGING_FACE_HUB_TOKEN', or run 'uvx huggingface-cli login'. "
        f"Ensure access is granted on the model card."
    )

