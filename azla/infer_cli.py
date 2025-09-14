import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from .config import Config
from .auth import resolve_hf_token, format_auth_error


def build_inputs(tokenizer, prompt: str, device: str):
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        except Exception:
            pass
    return tokenizer(prompt, return_tensors="pt").to(device)


def load_model(model_id: str, prefer_value_head: bool = True):
    token = resolve_hf_token()
    tok_kwargs = {"use_fast": True}
    if token:
        tok_kwargs["token"] = token
    try:
        tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    except Exception as e:
        raise RuntimeError(format_auth_error(model_id)) from e
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if prefer_value_head:
        try:
            mdl_kwargs = {}
            if token:
                mdl_kwargs["token"] = token
            model = AutoModelForCausalLMWithValueHead.from_pretrained(model_id, **mdl_kwargs)
            return tok, model.pretrained_model
        except Exception:
            pass
    mdl_kwargs = {}
    if token:
        mdl_kwargs["token"] = token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **mdl_kwargs)
    except Exception as e:
        raise RuntimeError(format_auth_error(model_id)) from e
    return tok, model


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_model_dir", type=str, default="runs/final")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args(argv)
    if args.base_model is None:
        try:
            cfg = Config.load(args.config)
            args.base_model = cfg.model.pretrained
        except Exception:
            args.base_model = "google/gemma-3-270m-it"
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok_ft, model_ft = load_model(args.ft_model_dir)
    tok_ft.padding_side = "left"
    model_ft.to(device)
    model_ft.eval()
    tok_base, model_base = load_model(args.base_model, prefer_value_head=False)
    tok_base.padding_side = "left"
    model_base.to(device)
    model_base.eval()
    ft_in = build_inputs(tok_ft, args.prompt, device)
    base_in = build_inputs(tok_base, args.prompt, device)
    with torch.no_grad():
        out_ft = model_ft.generate(
            **ft_in,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=tok_ft.eos_token_id,
        )
        out_base = model_base.generate(
            **base_in,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=tok_base.eos_token_id,
        )
    print("=== Fine-tuned ===")
    print(tok_ft.decode(out_ft[0], skip_special_tokens=True))
    print("=== Base ===")
    print(tok_base.decode(out_base[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
