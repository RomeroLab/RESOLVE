from __future__ import annotations

import hashlib

import numpy as np
from scipy.special import ndtri

D0_N = 16
BUDGET = 64
NAMESPACE = "PRE_RESOLVE_DEV_V2"

def normal_scores(y: np.ndarray) -> np.ndarray:

    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.lexsort((np.arange(n), y))
    rank = np.empty(n, dtype=np.float64)
    rank[order] = np.arange(n, dtype=np.float64)
    return ndtri((rank + 0.5) / n)

def elite_k(n: int) -> int:
    return max(10, -(-int(n) // 100))

def elite_from_labels(y: np.ndarray) -> tuple[set[int], int]:
    y = np.asarray(y, dtype=np.float64).ravel()
    k = elite_k(y.size)
    if not 0 < k < y.size:
        raise ValueError(f"elite count {k} is outside library size {y.size}")
    order = np.lexsort((np.arange(y.size), -y))
    return {int(i) for i in order[:k]}, k

def stream_seed(source: str, arm: str, seed: int, round_index: int, role: str) -> int:

    key = "|".join([
        "PRE_RESOLVE_RNG_V1", source, arm, str(int(seed)),
        str(int(round_index)), role, NAMESPACE,
    ])
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")

def shared_d0(source: str, seed: int, n_total: int, d0: int = D0_N) -> list[int]:

    if d0 >= n_total:
        raise ValueError(f"D0={d0} does not fit in n={n_total}")
    key = "|".join([
        "PRE_RESOLVE_RNG_V1", source, "COLD_START", str(int(seed)), "0",
        "uniform_baseline", NAMESPACE,
    ])
    stream = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    rng = np.random.default_rng(stream)
    picked = rng.choice(np.arange(n_total), size=int(d0), replace=False)
    return sorted(int(i) for i in picked)
