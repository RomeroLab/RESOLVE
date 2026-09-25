from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from resolve_method.kernel import Belief

POOL_POINTS = (
    (0.0, 0.0),
    (0.25, 0.25), (0.25, 1.0), (0.25, 4.0),
    (1.0, 0.25), (1.0, 1.0), (1.0, 4.0),
    (4.0, 0.25), (4.0, 1.0), (4.0, 4.0),
)
_LOG2PI = float(np.log(2.0 * np.pi))

def fit_pool(X, y, points=POOL_POINTS) -> list[Belief]:
    return [Belief.fit(X, y, lam=float(lam), gamma=float(gamma)) for lam, gamma in points]

def ml_weights(nll: np.ndarray) -> np.ndarray:
    nll = np.asarray(nll, dtype=np.float64).ravel()
    if nll.size == 0:
        raise ValueError("empty pool")
    ok = np.isfinite(nll)
    if not ok.any():
        return np.full(nll.size, 1.0 / nll.size)
    z = np.where(ok, -nll, -np.inf)
    z = z - z[ok].max()
    w = np.where(ok, np.exp(z), 0.0)
    return w / w.sum()

def gaussian_weights(pred: np.ndarray, y: np.ndarray) -> np.ndarray:

    pred = np.atleast_2d(np.asarray(pred, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64).ravel()
    if pred.shape[1] != y.size:
        raise ValueError(f"pred width {pred.shape[1]} != {y.size} labels")
    loglik = np.full(pred.shape[0], -np.inf)
    for j in range(pred.shape[0]):
        resid = pred[j] - y
        if not np.all(np.isfinite(resid)):
            continue
        scale = max(float(np.sqrt(np.mean(resid ** 2))), 1e-6)
        loglik[j] = -0.5 * (
            np.sum((resid / scale) ** 2) + y.size * (2.0 * np.log(scale) + _LOG2PI)
        )
    ok = np.isfinite(loglik)
    if not ok.any():
        return np.full(pred.shape[0], 1.0 / pred.shape[0])
    z = np.where(ok, loglik, -np.inf)
    z = z - z[ok].max()
    w = np.where(ok, np.exp(z), 0.0)
    total = w.sum()
    return w / total if total > 0 else np.full(pred.shape[0], 1.0 / pred.shape[0])

def pooled_elite(mu, sd, w, *, tau: float) -> np.ndarray:
    mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
    sd = np.atleast_2d(np.asarray(sd, dtype=np.float64))
    w = np.asarray(w, dtype=np.float64).ravel()
    if mu.shape != sd.shape:
        raise ValueError(f"mu {mu.shape} and sd {sd.shape} disagree")
    if w.shape[0] != mu.shape[0]:
        raise ValueError(f"{w.size} weights for {mu.shape[0]} worlds")
    if np.any(w < 0) or abs(float(w.sum()) - 1.0) > 1e-6:
        raise ValueError("weights must be non-negative and sum to 1")
    z = mu - float(tau)
    safe = sd > 0
    p = np.where(
        safe,
        ndtr(np.divide(z, np.where(safe, sd, 1.0))),
        (z > 0).astype(np.float64),
    )
    return w @ p

def stack_predict(beliefs: list[Belief], X, *, include_noise: bool = False):
    mus, sds = [], []
    for belief in beliefs:
        mu, sd = belief.predict(X, include_noise=include_noise)
        mus.append(mu)
        sds.append(sd)
    return np.asarray(mus), np.asarray(sds)

def adaptive_elite_threshold(pooled_mean, *, k: int, n: int) -> float:
    m = np.asarray(pooled_mean, dtype=np.float64).ravel()
    k, n = int(k), int(n)
    if not 0 < k < n:
        raise ValueError(f"k={k} must lie strictly inside n={n}")
    if m.size == 0:
        raise ValueError("no predictions to calibrate against")
    kth = float(np.partition(m, m.size - k)[m.size - k])
    below = m[m < kth]
    return float(0.5 * (kth + below.max())) if below.size else float(kth - 1e-12)

def map_index(nll) -> int:

    nll = np.asarray(nll, dtype=np.float64).ravel()
    if nll.size == 0:
        raise ValueError("empty pool")
    ok = np.isfinite(nll)
    if not ok.any():
        return 0
    masked = np.where(ok, nll, np.inf)
    return int(np.argmin(masked))

def refit_three(Xf_train, Xa_train, y_train, Xf_query, Xa_query, *, tau: float,
                single: bool = False):

    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    bf = fit_pool(Xf_train, y_train)
    ba = fit_pool(Xa_train, y_train)
    if single:
        bf = [bf[map_index([b.nll for b in bf])]]
        ba = [ba[map_index([b.nll for b in ba])]]
    nq = int(np.asarray(Xf_query).shape[0])
    empty = np.zeros(0, dtype=np.float64)
    if nq == 0:
        return empty, empty.copy(), empty.copy()
    mf, sf = stack_predict(bf, Xf_query, include_noise=False)
    ma, sa = stack_predict(ba, Xa_query, include_noise=False)
    w1 = ml_weights(np.asarray([b.nll for b in bf]))
    w2 = ml_weights(np.asarray([b.nll for b in ba]))
    w = np.concatenate([w1, w2])
    w = w / w.sum()
    pf = pooled_elite(mf, sf, w1, tau=float(tau))
    pa = pooled_elite(ma, sa, w2, tau=float(tau))
    pc = pooled_elite(np.vstack([mf, ma]), np.vstack([sf, sa]), w, tau=float(tau))
    return pf, pa, pc
