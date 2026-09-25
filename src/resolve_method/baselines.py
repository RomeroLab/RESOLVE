from __future__ import annotations

import numpy as np

from resolve_method.linear_head import (
    ShortlistWorld,
    ei_scores,
    fit_head,
    rank_pool,
    ucb_scores,
)
from resolve_method.scores import BUDGET, D0_N, elite_k, shared_d0, stream_seed

BASELINES = (
    "uniform_random",
    "frozen_esm2_masked_marginal_greedy",
    "frozen_esm2_bayes_ucb_beta_2.0",
    "diversity_kcenter",
    "linucb",
    "batchbald_regression",
    "one_batch_voi_topk",
    "gp_rollout_nonmyopic",
    "evolvepro_style_rf_sqrt",
    "alde_style_dnn_ts",
)
FEATURE_METHODS = frozenset(BASELINES) - {
    "uniform_random",
    "frozen_esm2_masked_marginal_greedy",
}
MENU = ("posterior_mean", "ucb_0.5", "ei", "uncertainty", "kcenter")
AA = "ACDEFGHIKLMNPQRSTVWY"

def _commit(batch, revealed, take: int) -> list[int]:
    seen = set()
    out = []
    for i in batch:
        i = int(i)
        if i in seen or i in revealed:
            continue
        seen.add(i)
        out.append(i)
    out = sorted(out)[: int(take)]
    if len(out) != int(take):
        raise RuntimeError(f"baseline returned {len(out)} of {take}")
    return out

def uniform_random(*, pool, revealed, take, source, seed, round_index, **_kw):
    del revealed
    rng = np.random.default_rng(stream_seed(
        source, "uniform_random", seed, round_index, "uniform_baseline"))
    pool = np.asarray(pool, dtype=np.int64)
    return sorted(int(i) for i in rng.choice(pool, size=min(take, pool.size), replace=False))

def masked_marginal_batch(*, pool, scores, take, **_kw):
    pool = [int(i) for i in pool]
    scores = np.asarray(scores, dtype=np.float64)
    order = sorted(pool, key=lambda i: (-scores[i], i))[: int(take)]
    return sorted(order)

def consensus_reference(sequences: list[str]) -> str:
    lengths = {len(seq) for seq in sequences}
    if len(lengths) != 1:
        raise ValueError(
            "masked marginal needs equal-length sequences, or pass --reference"
        )
    columns = []
    width = lengths.pop()
    for j in range(width):
        counts = {aa: 0 for aa in AA}
        for seq in sequences:
            if seq[j] not in counts:
                raise ValueError(f"residue {seq[j]!r} is outside the 20 amino acids")
            counts[seq[j]] += 1
        columns.append(max(AA, key=lambda aa: (counts[aa], -AA.index(aa))))
    return "".join(columns)

def _language_head(model):

    cached = getattr(model, "_resolve_lm_head", None)
    if cached is not None:
        return cached
    import torch
    from safetensors import safe_open
    from transformers.models.esm.modeling_esm import EsmLMHead
    from transformers.utils import cached_file

    from resolve_method.lora import BACKBONE

    head = EsmLMHead(model.config)
    path = cached_file(BACKBONE, "model.safetensors")
    wanted = {
        "dense.weight": "lm_head.dense.weight",
        "dense.bias": "lm_head.dense.bias",
        "layer_norm.weight": "lm_head.layer_norm.weight",
        "layer_norm.bias": "lm_head.layer_norm.bias",
        "bias": "lm_head.bias",
    }
    state = {}
    with safe_open(path, framework="pt") as handle:
        for dest, src in wanted.items():
            state[dest] = handle.get_tensor(src)
    head.load_state_dict(state, strict=False)
    with torch.no_grad():
        head.decoder.weight.copy_(model.embeddings.word_embeddings.weight.detach().cpu())
    device = next(model.parameters()).device
    head = head.to(device=device, dtype=torch.float32).eval()
    model._resolve_lm_head = head
    return head

def masked_marginal_scores(model, tok, sequences, reference, device, *, chunk: int = 32) -> np.ndarray:

    import torch

    if len(reference) == 0 or any(len(seq) != len(reference) for seq in sequences):
        raise ValueError("every sequence must match the reference length")
    aa_ids = []
    for aa in AA:
        ids = tok.encode(aa, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{aa} is not one ESM token")
        aa_ids.append(ids[0])
    aa_ids = torch.tensor(aa_ids, device=device)
    mask_id = tok.mask_token_id
    groups: dict[tuple[int, ...], list[int]] = {}
    for i, seq in enumerate(sequences):
        positions = tuple(p for p, (a, b) in enumerate(zip(seq, reference)) if a != b)
        groups.setdefault(positions, []).append(i)
    base = tok(reference, return_tensors="pt")
    template = base["input_ids"].to(device)
    out = np.zeros(len(sequences), dtype=np.float64)
    keys = [key for key in groups if key]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(keys), chunk):
            part = keys[start:start + chunk]
            batch = template.repeat(len(part), 1)
            for row, positions in enumerate(part):
                for pos in positions:
                    batch[row, pos + 1] = mask_id
            hidden = model(input_ids=batch).last_hidden_state
            logits = _language_head(model)(hidden)
            for row, positions in enumerate(part):
                table = []
                for pos in positions:
                    logp = torch.log_softmax(logits[row, pos + 1], dim=-1)
                    table.append(logp[aa_ids].detach().cpu().numpy())
                table = np.asarray(table)
                for index in groups[positions]:
                    total = 0.0
                    for j, pos in enumerate(positions):
                        mutant = AA.index(sequences[index][pos])
                        wild = AA.index(reference[pos])
                        total += float(table[j, mutant] - table[j, wild])
                    out[index] = total
    return out

def diversity_kcenter(*, X, pool, revealed, take, **_kw):
    pool = np.asarray(list(pool), dtype=np.int64)
    revealed_idx = np.asarray(sorted(revealed), dtype=np.int64)
    if revealed_idx.size == 0:
        best = np.full(pool.size, np.inf)
    else:
        shown = X[revealed_idx]
        shown_norm = (shown ** 2).sum(1)
        best = np.empty(pool.size, dtype=np.float64)
        for start in range(0, pool.size, 4096):
            block = X[pool[start:start + 4096]]
            dist = (block ** 2).sum(1)[:, None] + shown_norm[None, :] - 2.0 * (block @ shown.T)
            np.maximum(dist, 0.0, out=dist)
            best[start:start + block.shape[0]] = dist.min(1)
    points = X[pool]
    chosen, alive = [], np.ones(pool.size, dtype=bool)
    for _ in range(min(int(take), pool.size)):
        masked = np.where(alive, best, -np.inf)
        top = masked.max()
        pick = int(np.flatnonzero((masked == top) & alive)[0])
        chosen.append(int(pool[pick]))
        alive[pick] = False
        dist = ((points - points[pick]) ** 2).sum(1)
        best = np.minimum(best, dist)
    return sorted(chosen)

def _ucb(*, X, pool, revealed, take, beta, **_kw):
    head = fit_head(X, revealed)
    mu, sd = head.predict(X[np.asarray(pool, dtype=np.int64)])
    return rank_pool(pool, ucb_scores(mu, sd, beta=beta), take)

def _mean_batch(*, X, pool, revealed, take, **_kw):
    head = fit_head(X, revealed)
    mu = head.predict_mean(X[np.asarray(pool, dtype=np.int64)])
    return rank_pool(pool, mu, take)

def _ei_batch(*, X, pool, revealed, take, **_kw):
    head = fit_head(X, revealed)
    mu, sd = head.predict(X[np.asarray(pool, dtype=np.int64)])
    score = ei_scores(mu, sd, f_best=max(revealed.values()))
    return rank_pool(pool, score, take)

def _uncertainty(*, X, pool, revealed, take, **_kw):
    head = fit_head(X, revealed)
    _mu, sd = head.predict(X[np.asarray(pool, dtype=np.int64)])
    return rank_pool(pool, sd, take)

def _nominate(X, pool, revealed, take):
    head = fit_head(X, revealed)
    pool_i = np.asarray(pool, dtype=np.int64)
    mu, sd = head.predict(X[pool_i])
    f_best = max(revealed.values())
    table = {
        "posterior_mean": mu,
        "ucb_0.5": ucb_scores(mu, sd, beta=0.5),
        "ei": ei_scores(mu, sd, f_best=f_best),
        "uncertainty": sd,
    }
    noms = {name: rank_pool(pool_i, score, take) for name, score in table.items()}
    noms["kcenter"] = diversity_kcenter(X=X, pool=pool, revealed=revealed, take=take)
    return noms

def _context(*, pool, revealed, round_index, rounds, take, remaining, nominees):
    labels = np.asarray(sorted(revealed.values()), dtype=np.float64)
    median = float(np.median(labels))
    q1, q3 = np.quantile(labels, [0.25, 0.75])
    iqr = max(float(q3 - q1), 1e-12)
    total = max(1, int(rounds) * int(take))
    keys = list(nominees)
    overlaps = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            left, right = set(nominees[keys[i]]), set(nominees[keys[j]])
            overlaps.append(len(left & right) / max(1, len(left | right)))
    agree = float(np.mean(overlaps)) if overlaps else 0.0
    return np.array([
        1.0,
        float(round_index) / float(max(1, rounds)),
        float(remaining) / float(total),
        float(labels.size) / float(total),
        min(1.0, np.log10(max(len(pool), 1)) / 6.0),
        float(np.tanh((float(labels.max()) - median) / iqr)),
        float(np.tanh((median - float(labels.min())) / iqr)),
        agree,
        float(np.tanh(float(labels.std()) / iqr)),
        min(1.0, float(take) / float(max(1, len(pool)))),
    ], dtype=np.float64)

def linucb_step(*, X, pool, revealed, take, round_index, rounds, remaining, state, **_kw):
    noms = _nominate(X, pool, revealed, take)
    state = dict(state or {})
    if "A" not in state:
        state["A"] = {name: np.eye(10) for name in MENU}
        state["b"] = {name: np.zeros(10) for name in MENU}
    context = _context(
        pool=pool, revealed=revealed, round_index=round_index, rounds=rounds,
        take=take, remaining=remaining, nominees=noms,
    )
    scores = {}
    for name in MENU:
        inverse = np.linalg.inv(state["A"][name])
        theta = inverse @ state["b"][name]
        scores[name] = float(theta @ context + np.sqrt(max(context @ inverse @ context, 0.0)))
    chosen = max(MENU, key=lambda name: (scores[name], -MENU.index(name)))
    state["context"] = context
    state["last_expert"] = chosen
    state["f_best_before"] = float(max(revealed.values()))
    state["last_batch"] = list(noms[chosen])
    return list(noms[chosen]), state

def linucb_update(state, revealed_after, take):
    executed = list(state.get("last_batch") or [])
    before = float(state.get("f_best_before", 0.0))
    hits = sum(1 for i in executed if float(revealed_after[int(i)]) > before)
    reward = float(hits) / float(max(1, int(take)))
    expert = state["last_expert"]
    context = np.asarray(state["context"], dtype=np.float64)
    state["A"][expert] = state["A"][expert] + np.outer(context, context)
    state["b"][expert] = state["b"][expert] + reward * context
    return state

def _shortlist(pool, score, want):
    pool = np.asarray(pool, dtype=np.int64)
    order = sorted(range(pool.size), key=lambda j: (-score[j], int(pool[j])))[: int(want)]
    return [int(pool[j]) for j in sorted(order, key=lambda j: int(pool[j]))]

def batchbald_regression(*, X, pool, revealed, take, **_kw):
    head = fit_head(X, revealed)
    pool = np.asarray(list(pool), dtype=np.int64)
    _mu, sd = head.predict(X[pool])
    want = max(512, int(take))
    chosen_ids = _shortlist(pool, sd, min(want, pool.size))
    world = ShortlistWorld.from_head(head, X, chosen_ids)
    local = []
    for _ in range(int(take)):
        var = world.var.copy()
        if local:
            var[np.asarray(local, dtype=np.int64)] = -np.inf
        pick = int(np.argmax(var))
        if not np.isfinite(var[pick]):
            break
        local.append(pick)
        world = world.condition(pick, float(world.mean[pick]))
    return sorted(int(world.index[j]) for j in local)

def _topk_value(mean, k) -> float:
    k = int(min(k, mean.size))
    return 0.0 if k <= 0 else float(np.sort(mean)[-k:].sum())

def one_batch_voi_topk(*, X, pool, revealed, take, **_kw):
    head = fit_head(X, revealed)
    pool = np.asarray(list(pool), dtype=np.int64)
    mu, _sd = head.predict(X[pool])
    want = max(256, int(take) + int(take))
    chosen_ids = _shortlist(pool, mu, min(want, pool.size))
    world = ShortlistWorld.from_head(head, X, chosen_ids)
    nodes, weights = np.polynomial.hermite_e.hermegauss(5)
    weights = weights / np.sqrt(2.0 * np.pi)
    local = []
    for _ in range(int(take)):
        base = _topk_value(world.mean, int(take))
        scale = world.sd()
        best_j, best_v = -1, -np.inf
        blocked = set(local)
        for candidate in range(world.n):
            if candidate in blocked:
                continue
            total = 0.0
            for node, weight in zip(nodes, weights):
                fantasy = world.condition(
                    candidate, float(world.mean[candidate] + scale[candidate] * node))
                total += weight * _topk_value(fantasy.mean, int(take))
            value = total - base
            if value > best_v:
                best_j, best_v = candidate, value
        local.append(best_j)
        world = world.condition(best_j, float(world.mean[best_j]))
    return sorted(int(world.index[j]) for j in local)

def _ei_vector(mu, sd, f_best):
    return ei_scores(mu, sd, f_best=f_best)

def _greedy_ei(world, f_best, mask):
    score = np.where(mask, _ei_vector(world.mean, world.sd(), f_best), -np.inf)
    return int(np.argmax(score))

def _simulate(world, *, f_best, blocks, banned, draws):
    mask = np.ones(world.n, dtype=bool)
    if banned:
        mask[np.fromiter(banned, dtype=np.int64, count=len(banned))] = False
    best = float(f_best)
    cursor = 0
    total = 0.0
    for width in blocks:
        width = min(int(width), int(mask.sum()))
        if width <= 0:
            continue
        inner, selected, inner_mask = world, [], mask.copy()
        for _ in range(width):
            pick = _greedy_ei(inner, best, inner_mask)
            selected.append(pick)
            inner_mask[pick] = False
            inner = inner.condition(pick, float(inner.mean[pick]))
        for pick in selected:
            scale = float(np.sqrt(max(world.cov[pick, pick], 0.0) + world.sig2))
            fantasy = float(world.mean[pick] + scale * draws[cursor % draws.size])
            cursor += 1
            total += max(fantasy - best, 0.0)
            if fantasy > best:
                best = fantasy
            world = world.condition(pick, fantasy)
        mask[np.asarray(selected, dtype=np.int64)] = False
    return total

def gp_rollout_nonmyopic(*, X, pool, revealed, take, rounds_remaining, rng, **_kw):
    head = fit_head(X, revealed)
    pool = np.asarray(list(pool), dtype=np.int64)
    mu, sd = head.predict(X[pool])
    f_best = float(max(revealed.values()))
    score = ei_scores(mu, sd, f_best=f_best)
    horizon = max(0, int(rounds_remaining) - 1)
    want = min(
        max(96, int(take) * (1 + horizon) + int(take)),
        pool.size,
    )
    chosen_ids = _shortlist(pool, score, want)
    world = ShortlistWorld.from_head(head, X, chosen_ids)
    nodes, weights = np.polynomial.hermite_e.hermegauss(3)
    weights = weights / np.sqrt(2.0 * np.pi)
    local = []
    for slot in range(int(take)):
        slots_left = int(take) - slot
        n_future = max(0, slots_left - 1) + horizon * int(take)
        draws = rng.standard_normal(max(1, n_future + int(take)))
        immediate = _ei_vector(world.mean, world.sd(), f_best)
        value = np.full(world.n, -np.inf)
        banned = set(local)
        scale = world.sd()
        for candidate in range(world.n):
            if candidate in banned:
                continue
            total = float(immediate[candidate])
            if n_future > 0:
                future = 0.0
                for node, weight in zip(nodes, weights):
                    fantasy = float(world.mean[candidate] + scale[candidate] * node)
                    updated = world.condition(candidate, fantasy)
                    blocks = [max(0, slots_left - 1)] + [int(take)] * horizon
                    future += weight * _simulate(
                        updated, f_best=max(f_best, fantasy), blocks=blocks,
                        banned=banned | {candidate}, draws=draws,
                    )
                total += future
            value[candidate] = total
        best = int(np.argmax(value))
        local.append(best)
        world = world.condition(best, float(world.mean[best]))
    return sorted(int(world.index[j]) for j in local)

def evolvepro_style_rf_sqrt(*, X, pool, revealed, take, source, seed, round_index, **_kw):
    from sklearn.ensemble import RandomForestRegressor

    keys = np.asarray(sorted(revealed), dtype=np.int64)
    labels = np.asarray([revealed[int(i)] for i in keys], dtype=np.float64)
    forest = RandomForestRegressor(
        n_estimators=100, max_features="sqrt",
        random_state=stream_seed(source, "evolvepro_style_rf_sqrt", seed, round_index, "adaptive"),
        n_jobs=1,
    )
    forest.fit(X[keys], labels)
    pool = np.asarray(list(pool), dtype=np.int64)
    pred = np.empty(pool.size, dtype=np.float64)
    for start in range(0, pool.size, 4096):
        pred[start:start + 4096] = forest.predict(X[pool[start:start + 4096]])
    return rank_pool(pool, pred, take)

def alde_style_dnn_ts(*, X, pool, revealed, take, source, seed, round_index, **_kw):
    import torch

    keys = np.asarray(sorted(revealed), dtype=np.int64)
    labels = np.asarray([revealed[int(i)] for i in keys], dtype=np.float64)
    mu, sd = X[keys].mean(0), X[keys].std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    y_mu = float(labels.mean())
    y_sd = float(labels.std()) if labels.std() > 1e-12 else 1.0
    rng = np.random.default_rng(stream_seed(
        source, "alde_style_dnn_ts", seed, round_index, "adaptive"))
    members = []
    for member in range(5):
        boot = rng.choice(len(keys), size=max(2, int(round(0.9 * len(keys)))), replace=True)
        inputs = torch.tensor((X[keys][boot] - mu) / sd, dtype=torch.float32)
        targets = torch.tensor((labels[boot] - y_mu) / y_sd, dtype=torch.float32).unsqueeze(1)
        torch.manual_seed(stream_seed(
            source, "alde_style_dnn_ts", seed, round_index, f"init{member}"))
        net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        best, bad = float("inf"), 0
        for _ in range(100):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(inputs), targets)
            loss.backward()
            opt.step()
            value = float(loss.detach())
            if value < best - 1e-6:
                best, bad = value, 0
            else:
                bad += 1
                if bad >= 10:
                    break
        net.eval()
        members.append(net)
    pool = np.asarray(list(pool), dtype=np.int64)
    preds = []
    with torch.no_grad():
        for net in members:
            out = np.empty(pool.size, dtype=np.float64)
            for start in range(0, pool.size, 4096):
                block = torch.tensor((X[pool[start:start + 4096]] - mu) / sd, dtype=torch.float32)
                out[start:start + block.shape[0]] = net(block).squeeze(1).numpy()
            preds.append(out)
    chosen, alive = [], np.ones(pool.size, dtype=bool)
    for _ in range(min(int(take), pool.size)):
        sample = preds[int(rng.integers(0, 5))]
        masked = np.where(alive, sample, -np.inf)
        pick = int(np.argmax(masked))
        chosen.append(int(pool[pick]))
        alive[pick] = False
    return sorted(chosen)

def run_baseline(*, name, source, seed, sequences, y, elite, schedule, rounds, take,
                 features=None, marginal=None, reference=None) -> dict:
    if name not in BASELINES:
        raise KeyError(name)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = int(y.size)
    if n < BUDGET:
        raise ValueError(f"{source} has {n} sequences; a campaign labels {BUDGET}")
    if name in FEATURE_METHODS and features is None:
        raise ValueError(f"{name} needs the raw ESM-2 mean pool")
    if name == "frozen_esm2_masked_marginal_greedy" and marginal is None:
        raise ValueError("masked marginal scores were not computed")
    d0 = shared_d0(source, seed, n)
    revealed = {int(i): float(y[int(i)]) for i in d0}
    state = {}
    rows = []
    for t in range(rounds):
        pool = [i for i in range(n) if i not in revealed]
        acquired = len(revealed) - D0_N
        remaining = rounds * take - acquired
        rounds_left = -(-int(remaining) // max(1, int(take)))
        rng = np.random.default_rng(stream_seed(source, name, seed, t, "rollout"))
        common = dict(
            X=features, pool=pool, revealed=revealed, take=take, source=source,
            seed=seed, round_index=t, rounds=rounds, remaining=remaining,
            rounds_remaining=rounds_left, rng=rng, scores=marginal, state=state,
        )
        if name == "uniform_random":
            batch = uniform_random(**common)
        elif name == "frozen_esm2_masked_marginal_greedy":
            batch = masked_marginal_batch(**common)
        elif name == "frozen_esm2_bayes_ucb_beta_2.0":
            batch = _ucb(**common, beta=2.0)
        elif name == "diversity_kcenter":
            batch = diversity_kcenter(**common)
        elif name == "linucb":
            batch, state = linucb_step(**common)
        elif name == "batchbald_regression":
            batch = batchbald_regression(**common)
        elif name == "one_batch_voi_topk":
            batch = one_batch_voi_topk(**common)
        elif name == "gp_rollout_nonmyopic":
            batch = gp_rollout_nonmyopic(**common)
        elif name == "evolvepro_style_rf_sqrt":
            batch = evolvepro_style_rf_sqrt(**common)
        else:
            batch = alde_style_dnn_ts(**common)
        picks = _commit(batch, revealed, take)
        revealed.update({int(i): float(y[int(i)]) for i in picks})
        if name == "linucb":
            state = linucb_update(state, revealed, take)
        rows.append({"round": t, "picks": picks})
    labeled = sorted(revealed)
    if len(labeled) != BUDGET:
        raise RuntimeError(f"{name} labeled {len(labeled)}, expected {BUDGET}")
    hits = {f"hits_at_{mark}": sum(1 for i in labeled[:mark] if i in elite)
            for mark in (16, 32, 48, 64)}
    k = len(elite) if elite else elite_k(n)
    return {
        "method": name,
        "csv": source,
        "schedule": schedule,
        "seed": int(seed),
        "n": n,
        "k": int(k),
        "hits_at_64": int(hits["hits_at_64"]),
        "recall": float(hits["hits_at_64"]) / float(k),
        "checkpoints": hits,
        "reference": reference,
        "d0": d0,
        "rows": rows,
    }
