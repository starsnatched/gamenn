AZLA-MCTS: AlphaZero-Latent Agreement MCTS with TRL PPO and LoRA

Quickstart

- Create a virtualenv and install with uv:
  uv sync --all-extras

- Full fine-tuning on GPU with Gemma:
  uv run main

- Smoke test on CPU (optional):
  uv run azla -- --config configs/tiny.yaml --run smoke

Inference

- After training, the merged model is saved to `runs/final`.
- Compare fine-tuned vs base:
  uv run infer -- --prompt "Write a short greeting." --max_new_tokens 64

Structure

- azla/: core modules
- configs/: YAML configs (default, tiny)
- datasets/: sample tiny JSONL
- tests/: unit tests for LES and MCTS
