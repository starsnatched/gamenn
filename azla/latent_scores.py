from typing import Dict, List, Tuple
import numpy as np
from scipy.special import logsumexp

from .utils import entropy_np, js_divergence, ngram_repetition_ratio, clip01


def log_probs_from_logits(logits: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    lse = logsumexp(logits, axis=-1)
    idx = tokens.reshape(-1)
    sel = logits[np.arange(len(idx)), idx]
    return sel - lse


def length_normalized_logprob(logits: np.ndarray, tokens: np.ndarray) -> float:
    if len(tokens) == 0:
        return 0.0
    lp = log_probs_from_logits(logits, tokens)
    return float(np.mean(lp))


def entropy_reduction(baseline_logits: np.ndarray, conditioned_logits: np.ndarray) -> float:
    p0 = softmax_rows(baseline_logits)
    p1 = softmax_rows(conditioned_logits)
    h0 = np.mean([entropy_np(p) for p in p0])
    h1 = np.mean([entropy_np(p) for p in p1])
    return max(0.0, h0 - h1)


def softmax_rows(x: np.ndarray) -> List[np.ndarray]:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    s = e.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return [row / s[i] for i, row in enumerate(e)]


def stability_mc_dropout(dists: List[np.ndarray]) -> float:
    if len(dists) < 2:
        return 0.0
    pairwise = []
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            pairwise.append(js_divergence(dists[i], dists[j]))
    if not pairwise:
        return 0.0
    js = float(np.mean(pairwise))
    s = 1.0 / (1.0 + js)
    return s


def repetition_penalty(tokens: np.ndarray, n: int = 2) -> float:
    return ngram_repetition_ratio(tokens, n)


def predictive_gain(baseline_next_logits: np.ndarray, reasoned_next_logits: np.ndarray) -> float:
    p0 = softmax_rows(baseline_next_logits)
    p1 = softmax_rows(reasoned_next_logits)
    h0 = np.mean([entropy_np(p) for p in p0])
    h1 = np.mean([entropy_np(p) for p in p1])
    return max(0.0, h0 - h1)


def compute_les(
    seq_logits: np.ndarray,
    seq_tokens: np.ndarray,
    baseline_next_logits: np.ndarray,
    reasoned_next_logits: np.ndarray,
    mc_dropout_dists: List[np.ndarray],
    weights: Dict[str, float] = None,
) -> float:
    w = {
        "confidence": 1.0,
        "predictive_gain": 0.8,
        "entropy_reduction": 0.5,
        "stability": 0.7,
        "repetition": 0.5,
    }
    if weights:
        w.update(weights)
    conf = length_normalized_logprob(seq_logits, seq_tokens)
    conf_n = 0.5 + 0.5 * np.tanh(conf)
    pg = predictive_gain(baseline_next_logits, reasoned_next_logits)
    pg_n = 1.0 - np.exp(-pg)
    ent_red = entropy_reduction(baseline_next_logits, reasoned_next_logits)
    ent_n = 1.0 - np.exp(-ent_red)
    stab = stability_mc_dropout(mc_dropout_dists)
    rep = repetition_penalty(seq_tokens, n=2)
    les = (
        w["confidence"] * conf_n
        + w["predictive_gain"] * pg_n
        + w["entropy_reduction"] * ent_n
        + w["stability"] * stab
        - w["repetition"] * rep
    )
    return float(clip01(les / sum(abs(v) for v in w.values())))


def compute_les_from_logits(
    logits_seq: np.ndarray,
    tokens_seq: np.ndarray,
    baseline_next_logits: np.ndarray,
    reasoned_next_logits: np.ndarray,
) -> float:
    probs_mc = [row for row in softmax_rows(reasoned_next_logits)]
    return compute_les(logits_seq, tokens_seq, baseline_next_logits, reasoned_next_logits, probs_mc)
