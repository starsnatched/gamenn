from typing import Dict, List, Tuple
import math
import numpy as np

from .utils import softmax_np, top_k_indices


class Evaluator:
    def propose(self, prefix_ids: List[int], top_k: int, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def value_of_token(self, prefix_ids: List[int], token_id: int) -> float:
        raise NotImplementedError


class MCTSConfigRuntime:
    def __init__(
        self,
        num_sims: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_frac: float,
        progressive_widening_k: float,
        progressive_widening_alpha: float,
        root_top_k: int,
    ) -> None:
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.progressive_widening_k = progressive_widening_k
        self.progressive_widening_alpha = progressive_widening_alpha
        self.root_top_k = root_top_k


class RootNode:
    def __init__(self, prior_tokens: np.ndarray, priors: np.ndarray) -> None:
        self.tokens = prior_tokens.astype(int)
        self.priors = priors.astype(float)
        self.N = np.zeros_like(self.tokens, dtype=float)
        self.W = np.zeros_like(self.tokens, dtype=float)
        self.Q = np.zeros_like(self.tokens, dtype=float)


def puct_scores(Q: np.ndarray, P: np.ndarray, N_parent: float, N_child: np.ndarray, c_puct: float) -> np.ndarray:
    U = c_puct * P * math.sqrt(max(1.0, N_parent)) / (1.0 + N_child)
    return Q + U


def progressive_widening_limit(N_parent: float, k: float, alpha: float) -> int:
    return max(1, int(k * (N_parent ** alpha)))


def dirichlet_noise(priors: np.ndarray, alpha: float, frac: float, rng: np.random.Generator) -> np.ndarray:
    if alpha <= 0 or frac <= 0:
        return priors
    noise = rng.gamma(alpha, 1.0, size=priors.shape)
    noise = noise / noise.sum()
    return (1 - frac) * priors + frac * noise


def run_mcts_next_token(
    evaluator: Evaluator,
    prefix_ids: List[int],
    cfg: MCTSConfigRuntime,
    temperature: float,
    rng: np.random.Generator,
) -> Tuple[int, Dict[int, float]]:
    tokens, priors = evaluator.propose(prefix_ids, cfg.root_top_k, temperature)
    priors = priors / max(priors.sum(), 1e-8)
    priors = dirichlet_noise(priors, cfg.dirichlet_alpha, cfg.dirichlet_frac, rng)
    root = RootNode(tokens, priors)
    for _ in range(cfg.num_sims):
        limit = progressive_widening_limit(root.N.sum(), cfg.progressive_widening_k, cfg.progressive_widening_alpha)
        expanded = int((root.N > 0).sum())
        if expanded < min(limit, len(root.tokens)):
            unexpanded = np.where(root.N == 0)[0]
            if len(unexpanded) > 0:
                idx = rng.choice(unexpanded)
            else:
                idx = rng.integers(0, len(root.tokens))
        else:
            scores = puct_scores(root.Q, root.priors, root.N.sum(), root.N, cfg.c_puct)
            idx = int(np.argmax(scores))
        tok = int(root.tokens[idx])
        v = evaluator.value_of_token(prefix_ids, tok)
        root.N[idx] += 1.0
        root.W[idx] += float(v)
        root.Q[idx] = root.W[idx] / root.N[idx]
    visits = root.N
    if temperature > 1e-6:
        pi = visits ** (1.0 / temperature)
    else:
        pi = visits
    pi = pi / max(pi.sum(), 1e-8)
    choice = int(root.tokens[int(np.argmax(pi))])
    policy = {int(root.tokens[i]): float(pi[i]) for i in range(len(root.tokens)) if visits[i] > 0}
    return choice, policy

