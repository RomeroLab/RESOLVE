from __future__ import annotations

import numpy as np

AGREED_DRAWS = 2

def top_take(p: np.ndarray, take: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()
    legal = np.flatnonzero(np.isfinite(p))
    if legal.size < int(take):
        raise ValueError(f"need {take} finite scores, have {legal.size}")
    order = legal[np.lexsort((legal, -p[legal]))]
    return np.asarray(order[: int(take)], dtype=np.int64)

def mixture_labels(mu, sd, w, idx, rng, world=None) -> np.ndarray:

    mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
    sd = np.atleast_2d(np.asarray(sd, dtype=np.float64))
    w = np.asarray(w, dtype=np.float64).ravel()
    if float(w.sum()) <= 0:
        w = np.full(w.size, 1.0 / w.size)
    else:
        w = w / w.sum()
    drawn = int(rng.choice(w.size, p=w))
    if world is None:
        world = drawn
    else:
        world = int(world)
    sig = np.maximum(sd[world, idx], 1e-12)
    return rng.normal(mu[world, idx], sig)

def future_mass(p, take: int, rounds: int, blocked) -> float:
    score = np.array(np.asarray(p, dtype=np.float64).ravel(), copy=True)
    blocked = np.asarray(blocked, dtype=np.int64).ravel()
    if blocked.size:
        score[blocked] = -np.inf
    total = 0.0
    k = int(take)
    for _ in range(int(rounds)):
        legal = np.flatnonzero(np.isfinite(score))
        if legal.size == 0 or k < 1:
            break
        use = min(k, int(legal.size))
        order = legal[np.lexsort((legal, -score[legal]))][:use]
        total += float(score[order].sum())
        score[order] = -np.inf
    return total

def stable_advantage(a: float, b: float) -> bool:
    return a > 0.0 and b > 0.0 and min(a, b) > abs(a - b)

def stable_batch_value(p_frozen, p_tune, mu, sd, w, *, take: int,
                       remaining_rounds: int, refit_head, refit_repr,
                       rng, tune_ok=None, fallback: str = "repr",
                       world=None) -> dict:
    p = np.asarray(p_frozen, dtype=np.float64).ravel()
    pt = np.asarray(p_tune, dtype=np.float64).ravel()
    n = int(p.size)
    take = int(take)
    rounds = int(remaining_rounds)
    if pt.size != n or take < 1 or rounds < 0:
        raise ValueError(f"p_tune {pt.size} p {n} take={take} remaining_rounds={rounds}")
    if fallback not in ("repr", "head", "none"):
        raise ValueError(f"fallback must be repr, head, or none, not {fallback!r}")
    if tune_ok is None:
        tune_ok = np.ones(n, dtype=bool)
    else:
        tune_ok = np.asarray(tune_ok, dtype=bool).ravel()
    rng = np.random.default_rng(0) if rng is None else rng
    b0 = top_take(p, take)
    score = np.array(pt, copy=True)
    score[~tune_ok] = -np.inf
    score[~np.isfinite(p)] = -np.inf
    b1 = top_take(score, take) if int(np.isfinite(score).sum()) >= take else None
    same = b1 is not None and set(int(i) for i in b1) == set(int(i) for i in b0)

    def future(refit, batch, y):
        pc = np.asarray(refit(np.asarray(batch, dtype=np.int64), np.asarray(y, dtype=np.float64)),
                        dtype=np.float64).ravel()
        if pc.size != n or np.isnan(pc).any() or not np.isfinite(pc).any():
            raise ValueError("refit must return one shared score per candidate")
        return future_mass(pc, take, rounds, batch)

    def draw(batch):
        return mixture_labels(mu, sd, w, batch, rng, world=world)

    def pair(refit):
        advs = []
        for _ in range(AGREED_DRAWS):
            y0 = draw(b0)
            y1 = draw(b1)
            a0 = float(p[b0].sum()) + future(refit, b0, y0)
            a1 = float(p[b1].sum()) + future(refit, b1, y1)
            advs.append(a1 - a0)
        return advs

    mode = "frozen"
    head_adv = (0.0, 0.0)
    second_adv = (0.0, 0.0)
    if b1 is not None and rounds > 0 and not same:
        head_adv = tuple(pair(refit_head))
        if stable_advantage(*head_adv):
            mode = "head"
        elif fallback == "repr":
            second_adv = tuple(pair(refit_repr))
            if stable_advantage(*second_adv):
                mode = "repr"
        elif fallback == "head":

            second_adv = tuple(pair(refit_head))
            if stable_advantage(*second_adv):
                mode = "repeat"
    chosen = "tune" if mode != "frozen" else "keep"
    batch = b1 if chosen == "tune" else b0
    return {
        "keep": float(p[b0].sum()),
        "tune": float(p[b1].sum()) if b1 is not None else -np.inf,
        "chosen": chosen,
        "mode": mode,
        "batch": np.asarray(batch, dtype=np.int64),
        "swapped": set(int(i) for i in batch) != set(int(i) for i in b0),
        "head_adv": head_adv,
        "repr_adv": second_adv if mode == "repr" or fallback == "repr" else (0.0, 0.0),
        "second_adv": second_adv,
        "fallback": fallback,
        "value": float(p[batch].sum()),
    }

def _pair_batches(p_frozen, p_tune, *, take: int, tune_ok=None):

    p = np.asarray(p_frozen, dtype=np.float64).ravel()
    pt = np.asarray(p_tune, dtype=np.float64).ravel()
    take = int(take)
    if tune_ok is None:
        tune_ok = np.ones(p.size, dtype=bool)
    else:
        tune_ok = np.asarray(tune_ok, dtype=bool).ravel()
    b0 = top_take(p, take)
    score = np.array(pt, copy=True)
    score[~tune_ok] = -np.inf
    score[~np.isfinite(p)] = -np.inf
    b1 = top_take(score, take) if int(np.isfinite(score).sum()) >= take else None
    return p, pt, b0, b1

def fixed_choice(p_frozen, p_tune, *, take: int, which: str, tune_ok=None) -> dict:

    if which not in ("keep", "tune", "own"):
        raise ValueError(f"which must be keep, tune, or own, not {which!r}")
    p, pt, b0, b1 = _pair_batches(p_frozen, p_tune, take=take, tune_ok=tune_ok)
    own_keep = float(p[b0].sum())
    own_tune = float(pt[b1].sum()) if b1 is not None else -np.inf
    chosen = "keep"
    if which == "tune" and b1 is not None:
        chosen = "tune"
    elif which == "own" and b1 is not None and own_tune > own_keep:
        chosen = "tune"
    batch = b1 if chosen == "tune" else b0
    return {
        "keep": own_keep,
        "tune": float(p[b1].sum()) if b1 is not None else -np.inf,
        "own_keep": own_keep,
        "own_tune": own_tune,
        "chosen": chosen,
        "mode": "own" if which == "own" else ("tune" if chosen == "tune" else "frozen"),
        "batch": np.asarray(batch, dtype=np.int64),
        "swapped": set(int(i) for i in batch) != set(int(i) for i in b0),
        "head_adv": (0.0, 0.0),
        "repr_adv": (0.0, 0.0),
        "second_adv": (0.0, 0.0),
        "fallback": "none",
        "value": float(p[batch].sum()),
    }
