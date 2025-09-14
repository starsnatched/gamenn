import os
import math
import random
import time
import json
import hashlib
from typing import Any, Dict, Optional

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def detect_device(prefer_gpu: bool = True) -> str:
    if not prefer_gpu:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def timestamp_ms() -> int:
    return int(time.time() * 1000)


def stable_hash(obj: Any) -> str:
    data = json_dumps(obj).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def clip01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, x))


def softmax_np(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        temperature = 1e-6
    z = x / temperature
    z -= np.max(z)
    e = np.exp(z)
    s = e.sum()
    return e / s if s > 0 else np.ones_like(e) / len(e)


def entropy_np(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float((p * (np.log(p) - np.log(q))).sum())


def moving_average(values: np.ndarray, alpha: float) -> float:
    if len(values) == 0:
        return 0.0
    ma = 0.0
    w = 1.0
    for v in values:
        ma = alpha * v + (1 - alpha) * ma
        w = alpha + (1 - alpha) * w
    return float(ma / w)


def top_k_indices(probs: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, probs.shape[-1]))
    return np.argpartition(-probs, k - 1)[:k]


def ngram_repetition_ratio(tokens: np.ndarray, n: int = 2) -> float:
    if len(tokens) < n + 1:
        return 0.0
    seen: Dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        t = tuple(tokens[i : i + n])
        seen[t] = seen.get(t, 0) + 1
    repeats = sum(1 for c in seen.values() if c > 1)
    return repeats / max(1, len(seen))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_device(t, device: str):
    try:
        import torch

        if isinstance(t, torch.Tensor):
            return t.to(device)
    except Exception:
        return t
    return t

