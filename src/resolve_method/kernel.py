from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ESM_DIM = 1280
GRAM_DIVISOR = 2304
AMP_BOUNDS = (1e-3, 100.0)
SIGMA_BOUNDS = (1e-3, 10.0)
JITTER = 1e-10

def linear_cross(A: np.ndarray, B: np.ndarray, divisor: float = GRAM_DIVISOR) -> np.ndarray:
    return (np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64).T) / float(divisor)

def linear_diag(A: np.ndarray, divisor: float = GRAM_DIVISOR) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    return np.einsum("ij,ij->i", A, A) / float(divisor)

def rbf_from_linear(g: np.ndarray, *, gamma: float,
                    dA: np.ndarray | None = None,
                    dB: np.ndarray | None = None) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64)
    if dA is None or dB is None:
        if g.shape[0] != g.shape[1]:
            raise ValueError("dA and dB are required for a non-square Gram")
        dA = dB = np.diag(g)
    d2 = np.asarray(dA)[:, None] + np.asarray(dB)[None, :] - 2.0 * g
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-float(gamma) * d2)

def within_view_gram(A, B, *, lam: float, gamma: float,
                     divisor: float = GRAM_DIVISOR,
                     dA=None, dB=None) -> np.ndarray:
    lam = float(lam)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError(f"lam must be finite and non-negative, got {lam!r}")
    g = linear_cross(A, B, divisor)
    if lam == 0.0:
        return g
    if dA is None:
        dA = linear_diag(A, divisor)
    if dB is None:
        dB = linear_diag(B, divisor)
    return g + lam * rbf_from_linear(g, gamma=gamma, dA=dA, dB=dB)

def within_view_diag(A, *, lam: float, divisor: float = GRAM_DIVISOR) -> np.ndarray:
    lam = float(lam)
    base = linear_diag(A, divisor)
    return base if lam == 0.0 else base + lam

def negative_log_marginal(G: np.ndarray, ys: np.ndarray, a: float, s: float) -> float:
    n = ys.size
    A = (a ** 2) * G + (s ** 2 + JITTER) * np.eye(n)
    try:
        factor = cho_factor(A, lower=True, check_finite=False)
    except Exception:
        return float("inf")
    alpha = cho_solve(factor, ys, check_finite=False)
    logdet = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    value = 0.5 * (float(ys @ alpha) + logdet + n * np.log(2 * np.pi))
    return value if np.isfinite(value) else float("inf")

def _grid_nll(w, u2, n, amps, sigs) -> np.ndarray:
    denom = (amps[:, None, None] ** 2) * w[None, None, :] + (sigs[None, :, None] ** 2 + JITTER)
    bad = denom <= 0.0
    denom = np.where(bad, np.inf, denom)
    logdet = np.log(denom).sum(axis=2)
    quad = (u2[None, None, :] / denom).sum(axis=2)
    value = 0.5 * (quad + logdet + n * np.log(2 * np.pi))
    return np.where(np.isfinite(value), value, np.inf)

def fit_scales(G: np.ndarray, ys: np.ndarray, *, grid: int = 9):

    ys = np.asarray(ys, dtype=np.float64).ravel()
    n = ys.size
    G = np.asarray(G, dtype=np.float64)
    w, V = np.linalg.eigh(0.5 * (G + G.T))
    w = np.clip(w, 0.0, None)
    u2 = (V.T @ ys) ** 2

    def search(amps, sigs, best):
        M = _grid_nll(w, u2, n, amps, sigs)
        j = int(np.argmin(M))
        ai, si = divmod(j, sigs.size)
        return (float(M[ai, si]), float(amps[ai]), float(sigs[si])) if M[ai, si] < best[0] else best

    best = search(np.geomspace(0.05, 20.0, grid), np.geomspace(0.05, 3.0, grid), (np.inf, 1.0, 1.0))
    a0, s0 = best[1], best[2]
    for _ in range(3):
        best = search(
            np.geomspace(a0 / 2.0, a0 * 2.0, 7),
            np.geomspace(s0 / 2.0, s0 * 2.0, 7),
            best,
        )
        a0, s0 = best[1], best[2]
    return (float(np.clip(best[1], *AMP_BOUNDS)), float(np.clip(best[2], *SIGMA_BOUNDS)))

class Belief:

    def __init__(self, *, z_mu, z_sd, y_mu, y_sd, Zs, alpha, chol, d_tr,
                 lam_scale, sigma, lam, gamma, nll, divisor):
        self.z_mu = z_mu
        self.z_sd = z_sd
        self.y_mu = float(y_mu)
        self.y_sd = float(y_sd)
        self.Zs = Zs
        self.alpha = alpha
        self.chol = chol
        self.d_tr = d_tr
        self.lam_scale = float(lam_scale)
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.nll = float(nll)
        self.divisor = float(divisor)

    @classmethod
    def fit(cls, Z, y, *, lam: float, gamma: float,
            divisor: float = GRAM_DIVISOR) -> "Belief":
        Z = np.asarray(Z, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if Z.ndim != 2 or Z.shape[0] != y.size or y.size == 0:
            raise ValueError(f"design {Z.shape} does not match {y.size} labels")
        lam = float(lam)
        gamma = 0.0 if lam == 0.0 else float(gamma)
        z_mu = Z.mean(axis=0)
        sd = Z.std(axis=0)
        z_sd = np.where(sd > 1e-12, sd, 1.0)
        Zs = (Z - z_mu) / z_sd
        y_mu = float(y.mean())
        y_sd_raw = float(y.std())
        y_sd = y_sd_raw if y_sd_raw > 1e-12 else 1.0
        ys = (y - y_mu) / y_sd
        d_tr = linear_diag(Zs, divisor)
        if y.size < 4:
            a, s = 1.0, 1.0
            lam, gamma = 0.0, 0.0
        else:
            G_search = within_view_gram(Zs, Zs, lam=lam, gamma=gamma, divisor=divisor, dA=d_tr, dB=d_tr)
            a, s = fit_scales(G_search, ys)
            a = float(np.clip(a, *AMP_BOUNDS))
            s = float(np.clip(s, *SIGMA_BOUNDS))
        G = within_view_gram(Zs, Zs, lam=lam, gamma=gamma, divisor=divisor, dA=d_tr, dB=d_tr)
        nll = negative_log_marginal(G, ys, a, s)
        K = (a ** 2) * G
        system = K + (s ** 2 + JITTER) * np.eye(K.shape[0])
        chol = cho_factor(system, lower=True, check_finite=True)
        alpha = cho_solve(chol, ys, check_finite=True)
        return cls(
            z_mu=z_mu, z_sd=z_sd, y_mu=y_mu, y_sd=y_sd, Zs=Zs, alpha=alpha,
            chol=chol, d_tr=d_tr, lam_scale=a, sigma=s, lam=lam, gamma=gamma,
            nll=nll, divisor=divisor,
        )

    def _standardize(self, Z):
        return (np.asarray(Z, dtype=np.float64) - self.z_mu) / self.z_sd

    def predict(self, Z, *, include_noise: bool = True):
        Zt = self._standardize(Z)
        dq = linear_diag(Zt, self.divisor)
        a2 = self.lam_scale ** 2
        Ks = a2 * within_view_gram(
            Zt, self.Zs, lam=self.lam, gamma=self.gamma, divisor=self.divisor,
            dA=dq, dB=self.d_tr,
        )
        mu_s = Ks @ self.alpha
        kss = a2 * within_view_diag(Zt, lam=self.lam, divisor=self.divisor)
        solved = cho_solve(self.chol, Ks.T, check_finite=True)
        var_s = np.clip(kss - np.einsum("ij,ji->i", Ks, solved), 0.0, None)
        if include_noise:
            var_s = var_s + self.sigma ** 2
        return mu_s * self.y_sd + self.y_mu, np.sqrt(var_s) * self.y_sd
