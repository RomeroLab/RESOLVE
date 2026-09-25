from __future__ import annotations

import hashlib

import numpy as np

from resolve_method.kernel import ESM_DIM, Belief
from resolve_method.pool import (
    adaptive_elite_threshold,
    fit_pool,
    gaussian_weights,
    map_index,
    ml_weights,
    pooled_elite,
    refit_three,
    stack_predict,
)
from resolve_method.policy import fixed_choice, stable_batch_value
from resolve_method.scores import BUDGET, D0_N, normal_scores, shared_d0

SCHEDULES = {
    "2x24": (2, 24),
    "3x16": (3, 16),
    "4x12": (4, 12),
    "6x8": (6, 8),
    "8x6": (8, 6),
    "12x4": (12, 4),
}
WINDOW = 4096

VARIANTS = (
    "full",
    "frozen_pool",
    "tune_pool",
    "own_score",
    "select_once",
    "frozen_single",
    "full_single",
    "head_only",
    "head_repeat",
    "map_fantasy",
    "common_pool_greedy",
)
_ALIASES = {"ours": "full", "resolve": "full"}
_GATE = {
    "full": "repr",
    "head_only": "none",
    "head_repeat": "head",
    "map_fantasy": "repr",
    "full_single": "repr",
    "select_once": "repr",
}

def canonical_variant(name: str) -> str:
    key = _ALIASES.get(name, name)
    if key not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; choose {', '.join(VARIANTS)}")
    return key

def parse_schedule(name: str) -> tuple[int, int]:
    key = name.replace("×", "x").replace("X", "x").strip()
    if key not in SCHEDULES:
        raise ValueError(f"schedule {name!r} is not one of {', '.join(SCHEDULES)}")
    rounds, take = SCHEDULES[key]
    if D0_N + rounds * take != BUDGET:
        raise AssertionError("schedule does not spend the 64-label budget")
    return rounds, take

def _seed_int(text: str, mod: int) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % mod

def _feature_rows(ids, feat, mapping, frozen):
    rows = []
    for i in ids:
        i = int(i)
        rows.append(feat[mapping[i]] if i in mapping else frozen[i])
    return np.asarray(rows, dtype=np.float64)

def _prepare(source, seed, sequences, y, schedule):
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    if n < BUDGET:
        raise ValueError(f"{source} has {n} sequences; a campaign labels {BUDGET}")
    if len(sequences) != n:
        raise ValueError("sequence count does not match labels")
    rounds, take = parse_schedule(schedule)
    return y, n, rounds, take

def _finish(source, schedule, seed, n, elite, d0, revealed, rows, variant):
    labeled = sorted(revealed)
    if len(labeled) != BUDGET:
        raise RuntimeError(f"labeled {len(labeled)} sequences, expected {BUDGET}")
    hits = sum(1 for i in labeled if i in elite)
    return {
        "csv": source,
        "schedule": schedule.replace("×", "x"),
        "seed": int(seed),
        "variant": variant,
        "n": n,
        "k": len(elite),
        "hits_at_64": int(hits),
        "recall": float(hits) / float(len(elite)),
        "d0": d0,
        "rows": rows,
    }

def _adapt_window(encoder, sequences, frozen, idx, tgt, cands, *, source, seed, round_index,
                  seed_text):

    window_head = Belief.fit(frozen[idx], tgt, lam=0.0, gamma=0.0)
    mean = window_head.predict(frozen, include_noise=False)[0]
    unlabeled = [int(i) for i in cands]
    short = sorted(unlabeled, key=lambda i: (-float(mean[i]), i))[:min(WINDOW, len(unlabeled))]
    cand = np.asarray(sorted(set(short) | set(int(i) for i in idx)), dtype=np.int64)
    adapted = encoder.adapt_and_embed(
        [sequences[int(i)] for i in idx], tgt,
        [sequences[int(i)] for i in cand],
        epochs=10, seed=_seed_int(seed_text, 2 ** 31),
    )
    if adapted.shape != (cand.size, ESM_DIM):
        raise ValueError(f"adapted embedding {adapted.shape} does not match the window")
    loc = {int(g): j for j, g in enumerate(cand)}
    return adapted, cand, loc

def _collapse(beliefs, single: bool):
    if not single:
        return beliefs
    return [beliefs[map_index([b.nll for b in beliefs])]]

def run_campaign(*, source: str, seed: int, sequences: list[str], y: np.ndarray,
                 elite: set[int], encoder, schedule: str, variant: str = "full",
                 start: dict | None = None) -> dict:

    variant = canonical_variant(variant)
    if variant == "common_pool_greedy":
        return run_common_continuation(
            source=source, seed=seed, sequences=sequences, y=y, elite=elite,
            encoder=encoder, schedule=schedule, start=start,
        )
    y, n, rounds, take = _prepare(source, seed, sequences, schedule)
    frozen = np.asarray(encoder.embed_frozen(sequences), dtype=np.float64)
    if frozen.shape != (n, ESM_DIM):
        raise ValueError(f"frozen embedding {frozen.shape} is not {(n, ESM_DIM)}")
    if start is not None:
        raise ValueError("start is only used by common_pool_greedy")
    d0 = shared_d0(source, seed, n)
    revealed = {int(i): float(y[int(i)]) for i in d0}
    single = variant in ("full_single", "frozen_single")
    lock = None
    rows = []
    for t in range(rounds):
        if len(revealed) >= BUDGET:
            break
        take_now = min(take, BUDGET - len(revealed))
        idx = np.asarray(sorted(revealed), dtype=np.int64)
        tgt = normal_scores(np.array([revealed[int(i)] for i in idx]))
        cands = np.asarray([i for i in range(n) if i not in revealed], dtype=np.int64)
        if variant == "select_once" and lock == "keep":
            phase = "frozen"
        elif variant == "select_once" and lock == "tune":
            phase = "tune"
        elif variant in ("frozen_pool", "frozen_single"):
            phase = "frozen"
        elif variant == "tune_pool":
            phase = "tune"
        else:
            phase = "both"
        adapted = None
        loc = {}
        cand = idx
        if phase != "frozen":
            adapted, cand, loc = _adapt_window(
                encoder, sequences, frozen, idx, tgt, cands,
                source=source, seed=seed, round_index=t,
                seed_text=("VWB31-lora|" + source + "|" + str(seed) + "|" + str(t) + "|"
                           + ",".join(str(int(i)) for i in idx)),
            )
        bf = _collapse(fit_pool(frozen[idx], tgt), single)
        world_lock = map_index([b.nll for b in bf]) if variant == "map_fantasy" else None
        bt = []
        if phase != "frozen":
            bt = _collapse(fit_pool(adapted[[loc[int(i)] for i in idx]], tgt), single)
        mu_u, _ = stack_predict(bf, frozen, include_noise=False)
        w_frozen = ml_weights(np.asarray([b.nll for b in bf]))
        tau = adaptive_elite_threshold(w_frozen @ mu_u, k=len(elite), n=n)
        mu_c, sd_c = stack_predict(bf, frozen[cands], include_noise=False)
        p_frozen = pooled_elite(mu_c, sd_c, w_frozen, tau=tau)
        p_tune = np.full(cands.size, -np.inf)
        tune_ok = np.zeros(cands.size, dtype=bool)
        if bt:
            live = np.asarray([int(g) for g in cand if int(g) not in revealed], dtype=np.int64)
            pos = {int(g): j for j, g in enumerate(live)}
            shared = np.asarray([pos[int(g)] for g in cands if int(g) in pos], dtype=np.int64)
            cin = np.asarray([j for j, g in enumerate(cands) if int(g) in pos], dtype=np.int64)
            if shared.size:
                train_adapted = adapted[[loc[int(i)] for i in idx]]
                ins = np.asarray([
                    b.predict(train_adapted, include_noise=False)[0] for b in bt
                ])
                w_tune = gaussian_weights(ins, tgt)
                mu_t, sd_t = stack_predict(
                    bt, adapted[[loc[int(g)] for g in live]], include_noise=False)
                p_tune[cin] = pooled_elite(mu_t[:, shared], sd_t[:, shared], w_tune, tau=tau)
                tune_ok[cin] = True

        def scores_of(batch_local, y_fant, feat, mapping, _single=single, _tau=tau,
                      _idx=idx, _tgt=tgt, _cands=cands):
            batch_local = np.asarray(batch_local, dtype=np.int64).ravel()
            chosen_ids = _cands[batch_local]
            train_ids = np.concatenate([_idx, chosen_ids])
            y_train = np.concatenate([_tgt, np.asarray(y_fant, dtype=np.float64).ravel()])
            mask = np.ones(_cands.size, dtype=bool)
            mask[batch_local] = False
            query = _cands[mask]
            out = np.full(_cands.size, -np.inf)
            if query.size == 0:
                return out
            _, _, common = refit_three(
                frozen[train_ids],
                _feature_rows(train_ids, feat, mapping, frozen),
                y_train,
                frozen[query],
                _feature_rows(query, feat, mapping, frozen),
                tau=_tau, single=_single,
            )
            out[mask] = common
            return out

        def refit_head(batch_local, y_fant, _adapted=adapted, _loc=loc):
            return scores_of(batch_local, y_fant, _adapted, _loc)

        def refit_repr(batch_local, y_fant, _idx=idx, _tgt=tgt, _cands=cands, _cand=cand, _t=t):

            batch_local = np.asarray(batch_local, dtype=np.int64).ravel()
            y_fant = np.asarray(y_fant, dtype=np.float64).ravel()
            chosen_ids = _cands[batch_local]
            train_ids = np.concatenate([_idx, chosen_ids])
            targets = np.concatenate([_tgt, y_fant])
            embed_ids = np.asarray(sorted(set(int(i) for i in _cand) | set(int(i) for i in chosen_ids)),
                                   dtype=np.int64)
            repr_seed = _seed_int(
                "VWB31|" + str(source) + "|" + str(seed) + "|" + str(_t) + "|"
                + ",".join(str(int(i)) for i in chosen_ids),
                2 ** 31,
            )
            embedded = encoder.adapt_and_embed(
                [sequences[int(i)] for i in train_ids], targets,
                [sequences[int(i)] for i in embed_ids],
                epochs=1, seed=repr_seed,
            )
            mapping = {int(g): j for j, g in enumerate(embed_ids)}
            return scores_of(batch_local, y_fant, embedded, mapping)

        if phase == "frozen" or variant == "tune_pool" or variant == "own_score" or (
                variant == "select_once" and lock is not None):
            which = "keep" if phase == "frozen" else ("own" if variant == "own_score" else "tune")
            decision = fixed_choice(
                p_frozen, p_tune, take=take_now, which=which, tune_ok=tune_ok)
        else:
            decision_rng = np.random.default_rng(_seed_int(f"VWB31|{source}|{seed}|{t}", 2 ** 63))
            decision = stable_batch_value(
                p_frozen, p_tune, mu_c, sd_c, w_frozen,
                take=take_now, remaining_rounds=rounds - t - 1,
                refit_head=refit_head, refit_repr=refit_repr,
                rng=decision_rng, tune_ok=tune_ok,
                fallback=_GATE[variant], world=world_lock,
            )
        if variant == "select_once" and lock is None:
            lock = decision["chosen"]
        picks = sorted(int(cands[j]) for j in decision["batch"])
        if len(picks) != len(set(picks)) or len(picks) != take_now or set(picks) & set(revealed):
            raise RuntimeError(f"illegal batch at round {t}: {picks}")
        revealed.update({int(i): float(y[int(i)]) for i in picks})
        second = decision.get("second_adv", (0.0, 0.0))
        rows.append({
            "round": t,
            "mode": decision["mode"],
            "chosen": decision["chosen"],
            "picks": picks,
            "keep": decision["keep"],
            "tune": None if not np.isfinite(decision["tune"]) else decision["tune"],
            "head_adv": [float(v) for v in decision["head_adv"]],
            "repr_adv": [float(v) for v in decision["repr_adv"]],
            "second_adv": [float(v) for v in second],
            "locked": lock,
        })
    return _finish(source, schedule, seed, n, elite, d0, revealed, rows, variant)

def run_common_continuation(*, source: str, seed: int, sequences: list[str], y: np.ndarray,
                            elite: set[int], encoder, schedule: str,
                            start: dict | None = None) -> dict:

    y, n, rounds, take = _prepare(source, seed, sequences, schedule)
    frozen_emb = np.asarray(encoder.embed_frozen(sequences), dtype=np.float64)
    if frozen_emb.shape != (n, ESM_DIM):
        raise ValueError(f"frozen embedding {frozen_emb.shape} is not {(n, ESM_DIM)}")
    d0 = shared_d0(source, seed, n)
    if start is None:
        revealed = {int(i): float(y[int(i)]) for i in d0}
    else:
        revealed = {int(i): float(y[int(i)]) for i in start}
    rows = []
    step = 0
    while len(revealed) < BUDGET:
        take_now = min(take, BUDGET - len(revealed))
        idx = np.asarray(sorted(revealed), dtype=np.int64)
        tgt = normal_scores(np.array([revealed[int(i)] for i in idx]))
        cands = np.asarray([i for i in range(n) if i not in revealed], dtype=np.int64)
        adapted, cand, loc = _adapt_window(
            encoder, sequences, frozen_emb, idx, tgt, cands,
            source=source, seed=seed, round_index=step,
            seed_text=f"RESOLVE-cont|{source}|{seed}|{step}",
        )
        bf = fit_pool(frozen_emb[idx], tgt)
        bt = fit_pool(adapted[[loc[int(i)] for i in idx]], tgt)
        mu_u, _ = stack_predict(bf, frozen_emb, include_noise=False)
        w1 = ml_weights(np.asarray([b.nll for b in bf]))
        w2 = ml_weights(np.asarray([b.nll for b in bt]))
        tau = adaptive_elite_threshold(w1 @ mu_u, k=len(elite), n=n)
        query_adapted = _feature_rows(cands, adapted, loc, frozen_emb)
        mf, sf = stack_predict(bf, frozen_emb[cands], include_noise=False)
        ma, sa = stack_predict(bt, query_adapted, include_noise=False)
        w = np.concatenate([w1, w2])
        w = w / w.sum()
        common = pooled_elite(
            np.vstack([mf, ma]), np.vstack([sf, sa]), w, tau=tau)
        local = fixed_choice(common, np.full(cands.size, -np.inf), take=take_now, which="keep")
        picks = sorted(int(cands[j]) for j in local["batch"])
        if len(picks) != take_now or set(picks) & set(revealed):
            raise RuntimeError(f"illegal continuation batch at step {step}: {picks}")
        revealed.update({int(i): float(y[int(i)]) for i in picks})
        rows.append({
            "round": step,
            "mode": "common",
            "chosen": "common",
            "picks": picks,
            "keep": local["keep"],
            "tune": None,
            "head_adv": [0.0, 0.0],
            "repr_adv": [0.0, 0.0],
            "second_adv": [0.0, 0.0],
            "locked": None,
        })
        step += 1
        if step > rounds + 2:
            raise RuntimeError("continuation did not reach 64 labels")
    return _finish(source, schedule, seed, n, elite, d0, revealed, rows, "common_pool_greedy")
