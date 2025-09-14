import numpy as np

from azla.latent_scores import compute_les_from_logits


def test_les_monotonicity_confidence():
    logits_low = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    logits_high = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    tokens = np.array([0, 0])
    base = np.array([[0.0, 0.0, 0.0]])
    reason = np.array([[0.1, 0.0, -0.1]])
    les_low = compute_les_from_logits(logits_low, tokens, base, reason)
    les_high = compute_les_from_logits(logits_high, tokens, base, reason)
    assert les_high >= les_low


def test_les_entropy_reduction():
    logits_seq = np.array([[0.1, 0.0, -0.1]])
    tokens = np.array([0])
    base = np.array([[0.0, 0.0, 0.0]])
    reason = np.array([[3.0, 0.0, -3.0]])
    les1 = compute_les_from_logits(logits_seq, tokens, base, base)
    les2 = compute_les_from_logits(logits_seq, tokens, base, reason)
    assert les2 >= les1

