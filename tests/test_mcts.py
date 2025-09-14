import numpy as np

from azla.mcts import Evaluator, MCTSConfigRuntime, run_mcts_next_token


class TestEvaluator(Evaluator):
    def __init__(self, values: dict, priors: dict):
        self.values = values
        self.priors = priors

    def propose(self, prefix_ids, top_k, temperature):
        items = sorted(self.priors.items(), key=lambda x: -x[1])[:top_k]
        toks = np.array([k for k, _ in items], dtype=int)
        prs = np.array([v for _, v in items], dtype=float)
        prs = prs / prs.sum()
        return toks, prs

    def value_of_token(self, prefix_ids, token_id):
        return float(self.values.get(token_id, 0.0))


def test_mcts_picks_best_token():
    values = {1: 0.1, 2: 0.5, 3: 0.3}
    priors = {1: 0.2, 2: 0.3, 3: 0.5}
    ev = TestEvaluator(values, priors)
    cfg = MCTSConfigRuntime(
        num_sims=50,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        dirichlet_frac=0.0,
        progressive_widening_k=1.0,
        progressive_widening_alpha=0.5,
        root_top_k=3,
    )
    rng = np.random.default_rng(0)
    tok, pi = run_mcts_next_token(ev, [], cfg, 0.1, rng)
    assert tok == 2
    assert pi[2] == max(pi.values())

