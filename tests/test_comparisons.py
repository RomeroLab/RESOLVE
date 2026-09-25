import numpy as np

from resolve_method.policy import fixed_choice, mixture_labels, stable_batch_value
from resolve_method.pool import map_index, refit_three

def _scores():
    p = np.array([0.9, 0.8, 0.1, 0.0])
    pt = np.array([0.2, 0.1, 0.7, 0.6])
    mu = np.zeros((2, 4))
    sd = np.ones((2, 4))
    w = np.array([0.5, 0.5])
    return p, pt, mu, sd, w

def test_both_gates_fail_and_execute_frozen():
    p, pt, mu, sd, w = _scores()
    calls = []

    def head(batch, y):
        calls.append("head")
        return np.zeros(4)

    def repr_(batch, y):
        calls.append("repr")
        return np.zeros(4)

    decision = stable_batch_value(
        p, pt, mu, sd, w, take=2, remaining_rounds=1,
        refit_head=head, refit_repr=repr_, rng=np.random.default_rng(0),
        fallback="repr",
    )
    assert decision["chosen"] == "keep"
    assert decision["mode"] == "frozen"
    assert calls == ["head"] * 4 + ["repr"] * 4

def test_head_repeat_refits_features_and_does_not_call_representation():
    p, pt, mu, sd, w = _scores()
    calls = []

    def head(batch, y):
        calls.append("head")
        return np.zeros(4)

    def repr_(batch, y):
        calls.append("repr")
        return np.ones(4)

    decision = stable_batch_value(
        p, pt, mu, sd, w, take=2, remaining_rounds=1,
        refit_head=head, refit_repr=repr_, rng=np.random.default_rng(0),
        fallback="head",
    )
    assert calls == ["head"] * 8
    assert decision["chosen"] == "keep"
    assert decision["mode"] == "frozen"
    assert decision["second_adv"][0] <= 0.0 and decision["second_adv"][1] <= 0.0

def test_head_only_stops_after_the_first_pair():
    p, pt, mu, sd, w = _scores()
    calls = []

    def head(batch, y):
        calls.append("head")
        return np.zeros(4)

    decision = stable_batch_value(
        p, pt, mu, sd, w, take=2, remaining_rounds=1,
        refit_head=head, refit_repr=head, rng=np.random.default_rng(0),
        fallback="none",
    )
    assert calls == ["head"] * 4
    assert decision["chosen"] == "keep"
    assert decision["second_adv"] == (0.0, 0.0)

def test_passing_head_does_not_open_the_second_stage():
    p, pt, mu, sd, w = _scores()
    calls = []

    def head(batch, y):
        calls.append(("head", int(batch[0])))
        score = np.full(4, 0.01)
        if set(int(i) for i in batch) == {2, 3}:
            score[0] = 5.0
        return score

    decision = stable_batch_value(
        p, pt, mu, sd, w, take=2, remaining_rounds=1,
        refit_head=head, refit_repr=head, rng=np.random.default_rng(1),
        fallback="repr",
    )
    assert decision["mode"] == "head"
    assert decision["chosen"] == "tune"
    assert len(calls) == 4

def test_map_lock_keeps_the_normal_stream_and_changes_only_the_world():
    mu = np.array([[0.0, 0.0], [10.0, 10.0]])
    sd = np.ones((2, 2)) * 1e-12
    w = np.array([0.5, 0.5])
    idx = np.array([0, 1])
    free = np.random.default_rng(0)
    held = np.random.default_rng(0)
    unlocked = mixture_labels(mu, sd, w, idx, free)
    locked = mixture_labels(mu, sd, w, idx, held, world=1)

    assert np.allclose(locked, 10.0 + (unlocked - unlocked.mean()), atol=1e-6)

def test_own_score_uses_each_views_probability_and_ties_stay_frozen():
    p = np.array([0.9, 0.8, 0.0, 0.0])
    pt = np.array([-np.inf, -np.inf, 0.4, 0.4])
    lower = fixed_choice(p, pt, take=2, which="own")
    assert lower["chosen"] == "keep"
    higher = fixed_choice(p, np.array([-np.inf, -np.inf, 0.95, 0.95]), take=2, which="own")
    assert higher["chosen"] == "tune"
    tie = fixed_choice(p, np.array([-np.inf, -np.inf, 0.9, 0.8]), take=2, which="own")
    assert tie["chosen"] == "keep"
    missing = fixed_choice(p, np.full(4, -np.inf), take=2, which="tune")
    assert missing["chosen"] == "keep"

def test_map_index_tie_keeps_the_earlier_world():
    assert map_index([1.0, 1.0, 0.5, np.inf]) == 2
    assert map_index([0.2, 0.2, 0.2]) == 0
    assert map_index([np.inf, np.inf]) == 0

def test_single_refit_keeps_one_world_per_view():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 4))
    y = rng.normal(size=12)
    query = rng.normal(size=(5, 4))
    _, _, common = refit_three(x, x + 0.1, y, query, query + 0.1, tau=0.0, single=True)
    assert common.shape == (5,)
    assert np.all(np.isfinite(common))
