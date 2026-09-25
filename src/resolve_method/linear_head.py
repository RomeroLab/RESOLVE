from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import ndtr

LAM = 1.0
SIGMA = 1.0
JITTER = 1e-8

class HeadError(RuntimeError):
    pass

class LinearHead:
    def __init__(self):
        self._fitted = False

    def fit(self, Z, y) -> "LinearHead":
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if Z.ndim != 2 or Z.shape[0] == 0 or Z.shape[0] != y.size:
            raise HeadError(f"design {Z.shape} does not match {y.size} labels")
        if not np.all(np.isfinite(Z)) or not np.all(np.isfinite(y)):
            raise HeadError("training data is not finite")
        self._z_mu = Z.mean(axis=0)
        sd = Z.std(axis=0)
        self._z_sd = np.where(sd > 1e-12, sd, 1.0)
        self._y_mu = float(y.mean())
        y_sd = float(y.std())
        self._y_sd = y_sd if y_sd > 1e-12 else 1.0
        Zs = (Z - self._z_mu) / self._z_sd
        ys = (y - self._y_mu) / self._y_sd
        width = Zs.shape[1]
        precision = LAM * np.eye(width) + (Zs.T @ Zs) / (SIGMA ** 2)
        precision[np.diag_indices_from(precision)] += JITTER
        try:
            self._chol = cho_factor(precision, lower=True, check_finite=True)
        except Exception as exc:
            raise HeadError(f"posterior precision is not factorable: {exc}") from exc
        self._w = cho_solve(self._chol, (Zs.T @ ys) / (SIGMA ** 2), check_finite=True)
        self._fitted = True
        return self

    def _rows(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2:
            raise HeadError(f"expected a 2-D design, got {Z.shape}")
        return (Z - self._z_mu) / self._z_sd

    def predict(self, Z):
        if not self._fitted:
            raise HeadError("predict() called before fit()")
        n = np.asarray(Z).shape[0]
        mu_s = np.empty(n, dtype=np.float64)
        quad = np.empty(n, dtype=np.float64)
        for start in range(0, n, 8192):
            stop = min(start + 8192, n)
            rows = self._rows(np.asarray(Z)[start:stop])
            mu_s[start:stop] = rows @ self._w
            solved = cho_solve(self._chol, rows.T, check_finite=True)
            quad[start:stop] = np.einsum("ij,ji->i", rows, solved)
        var_s = np.clip(SIGMA ** 2 + quad, 0.0, None)
        return mu_s * self._y_sd + self._y_mu, np.sqrt(var_s) * self._y_sd

    def predict_mean(self, Z) -> np.ndarray:
        if not self._fitted:
            raise HeadError("predict_mean() called before fit()")
        return self._rows(Z) @ self._w * self._y_sd + self._y_mu

def fit_head(X, revealed) -> LinearHead:
    keys = np.asarray(sorted(revealed), dtype=np.int64)
    if keys.size == 0:
        raise HeadError("cannot fit a posterior on an empty revealed set")
    y = np.asarray([revealed[int(i)] for i in keys], dtype=np.float64)
    return LinearHead().fit(X[keys], y)

def rank_pool(pool, score, batch_size) -> list[int]:
    pool = np.asarray(pool, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    order = sorted(range(len(pool)), key=lambda j: (-score[j], int(pool[j])))
    return sorted(int(pool[j]) for j in order[: int(batch_size)])

def ucb_scores(mu, sd, *, beta: float) -> np.ndarray:
    return np.asarray(mu, dtype=np.float64) + float(beta) * np.asarray(sd, dtype=np.float64)

def ei_scores(mu, sd, *, f_best: float, jitter: float = 0.0) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    sd_safe = np.maximum(sd, 1e-12)
    gap = mu - float(f_best) - float(jitter)
    z = gap / sd_safe
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei = gap * ndtr(z) + sd_safe * phi
    return np.where(sd > 0, ei, np.maximum(gap, 0.0))

class ShortlistWorld:

    def __init__(self, index, mean, cov, sig2):
        self.index = np.asarray(index, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.sig2 = float(sig2)

    @classmethod
    def from_head(cls, head: LinearHead, X, candidates) -> "ShortlistWorld":
        idx = np.asarray(list(candidates), dtype=np.int64)
        rows = head._rows(np.asarray(X[idx], dtype=np.float64))
        factor, lower = head._chol
        if not lower:
            raise HeadError("rollout requires a lower Cholesky factor")
        solved = solve_triangular(factor, rows.T, lower=True, check_finite=True)
        cov = (solved.T @ solved) * head._y_sd ** 2
        cov = 0.5 * (cov + cov.T)
        mean = rows @ head._w * head._y_sd + head._y_mu
        sig2 = (SIGMA * head._y_sd) ** 2
        return cls(idx, mean, cov, sig2)

    @property
    def n(self) -> int:
        return int(self.mean.size)

    @property
    def var(self) -> np.ndarray:
        return np.maximum(np.diag(self.cov), 0.0)

    def sd(self) -> np.ndarray:
        return np.sqrt(self.var + self.sig2)

    def condition(self, i: int, y: float) -> "ShortlistWorld":
        i = int(i)
        column = self.cov[:, i]
        scale = float(self.cov[i, i] + self.sig2)
        if scale <= 1e-300:
            return ShortlistWorld(self.index, self.mean.copy(), self.cov.copy(), self.sig2)
        mean = self.mean + column * ((float(y) - float(self.mean[i])) / scale)
        cov = self.cov - np.outer(column, column) / scale
        cov = 0.5 * (cov + cov.T)
        return ShortlistWorld(self.index, mean, cov, self.sig2)
