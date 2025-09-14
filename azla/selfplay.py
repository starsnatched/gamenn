from typing import Dict, List, Tuple
import numpy as np

from .mcts import Evaluator, MCTSConfigRuntime, run_mcts_next_token
from .latent_scores import compute_les


class ModelEvaluator(Evaluator):
    def __init__(self, model_manager, max_length: int, temperature: float) -> None:
        self.mm = model_manager
        self.max_length = max_length
        self.temperature = temperature

    def propose(self, prefix_ids: List[int], top_k: int, temperature: float):
        import torch

        ids = torch.tensor([prefix_ids], device=self.mm.device)
        attn = torch.ones_like(ids)
        logits = self.mm.logits_for_next(ids, attn, use_dropout=True)
        probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)[0].detach().cpu().numpy()
        idxs = np.argpartition(-probs, min(top_k, probs.shape[-1] - 1))[:top_k]
        p = probs[idxs]
        p = p / max(p.sum(), 1e-8)
        return idxs.astype(int), p.astype(float)

    def value_of_token(self, prefix_ids: List[int], token_id: int) -> float:
        import torch

        ids = torch.tensor([prefix_ids + [token_id]], device=self.mm.device)
        attn = torch.ones_like(ids)
        with self.mm.enable_dropout():
            out = self.mm.model.pretrained_model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
            logits_seq = out.logits[0, -min(ids.shape[1], 4) :, :].detach().cpu().numpy()
        seq_tokens = ids[0, -logits_seq.shape[0] :].detach().cpu().numpy()
        base_ids = torch.tensor([prefix_ids], device=self.mm.device)
        base_attn = torch.ones_like(base_ids)
        base_logits = self.mm.logits_for_next(base_ids, base_attn, use_dropout=False)[0:1].detach().cpu().numpy()
        with self.mm.enable_dropout():
            reasoned_logits = self.mm.logits_for_next(ids, attn, use_dropout=True)[0:1].detach().cpu().numpy()
        dists = []
        for _ in range(2):
            with self.mm.enable_dropout():
                p = self.mm.logits_for_next(ids, attn, use_dropout=True)[0].detach().cpu().numpy()
                p = np.exp(p - p.max())
                p = p / max(p.sum(), 1e-8)
                dists.append(p)
        v = compute_les(logits_seq, seq_tokens, base_logits, reasoned_logits, dists)
        return v


def _chat_input_ids(tokenizer, prompt: str, device: str):
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        try:
            ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
            return ids[0].tolist()
        except Exception:
            pass
    return tokenizer.encode(prompt, add_special_tokens=False)


def play_one(mm_a, mm_b, tokenizer, mcts_cfg: MCTSConfigRuntime, max_length: int, temperature: float, rng: np.random.Generator, prompt: str):
    ids_a = _chat_input_ids(tokenizer, prompt, mm_a.device)
    ids_b = ids_a.copy()
    eval_a = ModelEvaluator(mm_a, max_length, temperature)
    eval_b = ModelEvaluator(mm_b, max_length, temperature)
    traj_a: List[Tuple[int, Dict[int, float]]] = []
    traj_b: List[Tuple[int, Dict[int, float]]] = []
    for _ in range(max_length):
        tok_a, pi_a = run_mcts_next_token(eval_a, ids_a, mcts_cfg, temperature, rng)
        ids_a.append(tok_a)
        traj_a.append((tok_a, pi_a))
        tok_b, pi_b = run_mcts_next_token(eval_b, ids_b, mcts_cfg, temperature, rng)
        ids_b.append(tok_b)
        traj_b.append((tok_b, pi_b))
        if tok_a == tokenizer.eos_token_id and tok_b == tokenizer.eos_token_id:
            break
    v_a = eval_a.value_of_token(ids_a[:-1], ids_a[-1]) if ids_a else 0.0
    v_b = eval_b.value_of_token(ids_b[:-1], ids_b[-1]) if ids_b else 0.0
    winner = 0 if v_a >= v_b else 1
    return {
        "winner": winner,
        "ids_a": ids_a,
        "ids_b": ids_b,
        "pi_a": traj_a,
        "pi_b": traj_b,
        "les_a": float(v_a),
        "les_b": float(v_b),
    }
