from typing import Deque
from collections import deque
import numpy as np

from .config import MCTSConfig


class CurriculumController:
    def __init__(self, cfg: MCTSConfig, window: int, inc_thr: float, dec_thr: float, min_sims: int, max_sims: int) -> None:
        self.cfg = cfg
        self.window = window
        self.inc_thr = inc_thr
        self.dec_thr = dec_thr
        self.min_sims = min_sims
        self.max_sims = max_sims
        self.buffer: Deque[float] = deque(maxlen=window)

    def update(self, score: float) -> None:
        self.buffer.append(score)

    def adjust(self) -> MCTSConfig:
        if not self.buffer:
            return self.cfg
        mean = float(np.mean(self.buffer))
        sims = self.cfg.num_sims
        if mean > self.inc_thr:
            sims = min(self.max_sims, sims + max(1, sims // 4))
        elif mean < self.dec_thr:
            sims = max(self.min_sims, sims - max(1, sims // 4))
        self.cfg.num_sims = sims
        return self.cfg

