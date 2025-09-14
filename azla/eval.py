from typing import List

from .latent_scores import compute_les_from_logits


def evaluate_mean_les(logits_list, tokens_list, base_next_list, reason_next_list) -> float:
    vals = []
    for a, b, c, d in zip(logits_list, tokens_list, base_next_list, reason_next_list):
        vals.append(compute_les_from_logits(a, b, c, d))
    return float(sum(vals) / max(1, len(vals)))

