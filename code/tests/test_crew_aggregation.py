"""The testimony-aggregation lattice (the fold of the cheap-talk receiver into the crew).

The load-bearing test is `test_mean_rule_is_the_sum_of_per_speaker_signals`: it pins the
decomposition invariant that makes `rule="mean"` the pre-refactor crew, so the fold stays
non-destructive as the lattice grows.
"""

from __future__ import annotations

import numpy as np
import pytest

from social_collusion.env import VecEnv
from social_collusion.policies import crew_aggregation as ca
from social_collusion.policies import make_pair

# NB: imported as a module rather than by name. `testimony_signals` matches pytest's default
# `test*` collection glob, so importing it directly makes pytest try to collect it as a test.
RULES = ca.RULES
own_evidence = ca.own_evidence
speaker_credibility = ca.speaker_credibility
suspicion_scores = ca.suspicion_scores
aggregate_testimony = ca.aggregate_testimony


@pytest.fixture(scope="module")
def states(cfg):
    return VecEnv(cfg, 32, seed=9090).run_episodes(make_pair("truthful", "alibi"), 120)


def _living_voters(st):
    return [v for v in range(st.config.n_agents) if st.alive[v]]


def test_mean_rule_is_the_sum_of_per_speaker_signals(cfg, states):
    """The decomposition invariant: own evidence + summed testimony == the mean rule.

    This is what guarantees the refactor reproduces the original hand-tuned crew (verified
    separately against the pre-refactor implementation at max |diff| = 4.4e-16, differing only
    on exact ties).
    """
    for st in states[:40]:
        for v in _living_voters(st):
            sig, spoke = ca.testimony_signals(st, v)
            recomposed = own_evidence(st, v) + sig[np.flatnonzero(spoke)].sum(axis=0)
            actual = suspicion_scores(st, v, "mean")
            fin = np.isfinite(actual)
            assert np.allclose(recomposed[fin], actual[fin], atol=1e-12)


def test_only_speakers_contribute_signal(cfg, states):
    for st in states[:40]:
        for v in _living_voters(st):
            sig, spoke = ca.testimony_signals(st, v)
            assert not sig[~spoke].any(), "a non-speaker contributed testimony signal"


def test_every_rule_masks_self_and_inactive(cfg, states):
    for rule in RULES:
        for st in states[:20]:
            for v in _living_voters(st):
                s = suspicion_scores(st, v, rule)
                assert s[v] == -np.inf
                for j in range(cfg.n_agents):
                    if not st.alive[j]:
                        assert s[j] == -np.inf
                assert np.isfinite(s[np.isfinite(s)]).all()


def test_rules_are_not_all_identical(cfg, states):
    """Sanity: the lattice must actually do something, or the whole fold is a no-op."""
    differs = False
    for st in states:
        for v in _living_voters(st):
            a = suspicion_scores(st, v, "mean")
            b = suspicion_scores(st, v, "median")
            fin = np.isfinite(a) & np.isfinite(b)
            if fin.any() and not np.allclose(a[fin], b[fin]):
                differs = True
                break
        if differs:
            break
    assert differs, "median and mean produced identical suspicion everywhere"


def test_caught_liar_loses_credibility(cfg, states):
    """A speaker the voter personally caught must not outrank an uncaught one."""
    from social_collusion.env.knowledge import ObserverKnowledge

    checked = 0
    for st in states:
        for v in _living_voters(st):
            k = ObserverKnowledge.build(st, v)
            cred = speaker_credibility(st, v, k)
            caught = {int(c.speaker) for c in st.claims if k.caught_false(c)}
            uncaught = {int(c.speaker) for c in st.claims} - caught
            for a in caught:
                for b in uncaught:
                    assert cred[a] < cred[b], (
                        f"caught speaker {a} (cred {cred[a]}) outranked uncaught {b} "
                        f"(cred {cred[b]})"
                    )
                    checked += 1
    assert checked > 0, "no caught-vs-uncaught pair appeared; the test was vacuous"


def test_unknown_rule_raises(cfg, states):
    st = states[0]
    v = _living_voters(st)[0]
    sig, spoke = ca.testimony_signals(st, v)
    with pytest.raises(ValueError):
        aggregate_testimony(sig, spoke, "not_a_rule")
