import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from azla.auth import resolve_hf_token, format_auth_error
from trl import AutoModelForCausalLMWithValueHead


def build_inputs(tokenizer, prompt: str, device: str):
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        except Exception:
            pass
    return tokenizer(prompt, return_tensors="pt").to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    token = resolve_hf_token()
    tok_kwargs = {"use_fast": True}
    if token:
        tok_kwargs["token"] = token
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir, **tok_kwargs)
    except Exception as e:
        raise RuntimeError(format_auth_error(args.model_dir)) from e
    try:
        mdl_kwargs = {}
        if token:
            mdl_kwargs["token"] = token
        model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_dir, **mdl_kwargs)
        gen_model = model.pretrained_model
    except Exception:
        mdl_kwargs = {}
        if token:
            mdl_kwargs["token"] = token
        try:
            gen_model = AutoModelForCausalLM.from_pretrained(args.model_dir, **mdl_kwargs)
        except Exception as e:
            raise RuntimeError(format_auth_error(args.model_dir)) from e
    gen_model.to(device)
    gen_model.eval()
    inputs = build_inputs(tok, args.prompt, device)
    with torch.no_grad():
        out = gen_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
