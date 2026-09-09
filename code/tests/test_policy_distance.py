"""Probe-set policy distances (rl/policy_distance.py)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from social_collusion.config import load_env_config  # noqa: E402
from social_collusion.env.enums import Phase, Role  # noqa: E402
from social_collusion.policies.scripted_crew import TruthfulCrew  # noqa: E402
from social_collusion.rl import policy_distance as pd  # noqa: E402
from social_collusion.rl.torch_policy import TorchPolicy  # noqa: E402


@pytest.fixture(scope="module")
def probe():
    cfg = load_env_config("full_short_game").with_(n_agents=5, n_coalition=2, max_rounds=1)
    return pd.collect_probe_set(cfg, pd.default_probe_policies(cfg), seed=99, episodes_per_policy=3)


def test_probe_set_covers_both_roles_and_the_vote_phase(probe):
    assert len(probe) > 50
    assert probe.rows(role=Role.COALITION).size > 0
    assert probe.rows(role=Role.CREW).size > 0
    assert probe.rows(role=Role.CREW, phase=Phase.VOTE).size > 0
    for h, m in enumerate(probe.masks):
        assert m.shape == (len(probe), probe.masks[h].shape[1]) and m.any(axis=1).all()


def test_probe_set_is_reproducible(probe):
    again = pd.collect_probe_set(
        probe.cfg, pd.default_probe_policies(probe.cfg), seed=99, episodes_per_policy=3
    )
    assert len(again) == len(probe)
    assert np.array_equal(again.obs, probe.obs)


def test_identical_policies_have_zero_distance(probe):
    pol = TorchPolicy(probe.cfg, seed=1)
    P = pd.head_probs(pol, probe)
    rows = probe.rows(role=Role.COALITION)
    assert np.allclose(pd.kl_divergence(P, P, probe, rows), 0.0, atol=1e-9)
    assert np.allclose(pd.js_divergence(P, P, probe, rows), 0.0, atol=1e-9)
    assert pd.vote_pattern_divergence(P, P, probe, Role.COALITION) == pytest.approx(0.0, abs=1e-9)
    assert pd.argmax_disagreement(P, P, probe, rows) == 0.0


def test_distinct_policies_have_positive_bounded_distance(probe):
    a, b = TorchPolicy(probe.cfg, seed=1), TorchPolicy(probe.cfg, seed=2)
    with torch.no_grad():  # push b away from a so the untrained nets are not near-identical
        for p in b.model.actor.head.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    P, Q = pd.head_probs(a, probe), pd.head_probs(b, probe)
    rows = probe.rows(role=Role.CREW)
    kl = pd.kl_divergence(P, Q, probe, rows)
    js = pd.js_divergence(P, Q, probe, rows)
    assert kl.mean() > 0.0 and js.mean() > 0.0
    assert (js >= -1e-9).all() and js.max() <= len(probe.masks) + 1e-9
    assert 0.0 <= pd.argmax_disagreement(P, Q, probe, rows) <= 1.0
    v = pd.vote_pattern_divergence(P, Q, probe, Role.CREW)
    assert 0.0 <= v <= 1.0 + 1e-9


def test_entropy_matches_the_masked_uniform_of_a_fresh_actor(probe):
    """A freshly initialised actor (head gain 0.01) is near-uniform over legal actions."""
    pol = TorchPolicy(probe.cfg, seed=3)
    P = pd.head_probs(pol, probe)
    rows = probe.rows(role=Role.CREW, phase=Phase.VOTE)
    ent = pd.entropy(P, probe, rows)
    legal = probe.masks[12][rows].sum(axis=1)  # vote head
    assert np.allclose(ent, np.log(legal), atol=0.15)


def test_vote_categories_are_canonical(probe):
    rows = probe.rows(role=Role.COALITION, phase=Phase.VOTE)
    cats = probe.vote_category[rows]
    # every coalition voter can see exactly one 'other coalition' category (its partner) if alive
    assert ((cats == 1).sum(axis=1) <= 1).all()
    assert ((cats == 3).sum(axis=1) == 1).all()  # skip is always available in this config
    pol = TorchPolicy(probe.cfg, seed=4)
    P = pd.head_probs(pol, probe)
    dist = pd.vote_category_distribution(P, probe, rows)
    assert np.allclose(dist.sum(axis=1), 1.0)


def test_scripted_policies_compare_too(probe):
    a = TruthfulCrew(rule="soft_credibility")
    b = TruthfulCrew(rule="median")
    out = pd.compare_policies(a, b, probe, Role.CREW)
    assert out["kl_to_previous"] >= 0.0
    assert 0.0 <= out["argmax_disagreement"] <= 1.0
    same = pd.compare_policies(a, a, probe, Role.CREW)
    assert same["argmax_disagreement"] == 0.0 and same["vote_pattern_js"] == pytest.approx(0.0)
