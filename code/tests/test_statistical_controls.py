"""Statistical machinery and the controls the collusion claim depends on (plan sec.17, sec.3.1).

The tests that matter scientifically are the last three: they assert that the benchmark's
metrics *fail* to find collusion when there is none, that the counterfactual estimator has the
right sign on known ground truth, and that the environment's key design decision is load-bearing.
"""

from __future__ import annotations

import numpy as np
import pytest

from social_collusion.env import VecEnv
from social_collusion.metrics import collusion_metrics as cm
from social_collusion.metrics import counterfactuals as cf
from social_collusion.metrics import outcome_metrics as om
from social_collusion.metrics import statistics as stats
from social_collusion.policies import make_pair
from social_collusion.policies.registry import make_coalition


# -- estimators -------------------------------------------------------------
def test_bootstrap_ci_covers_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, size=20)
    ci = stats.bootstrap_ci(x, seed=0)
    assert ci.lo <= ci.point <= ci.hi
    assert ci.lo <= 0.5 <= ci.hi


def test_bootstrap_ci_degenerates_gracefully():
    assert np.isnan(stats.bootstrap_ci([]).point)
    one = stats.bootstrap_ci([0.3])
    assert one.point == one.lo == one.hi == 0.3


def test_paired_bootstrap_detects_a_real_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(0.4, 0.05, size=10)
    ci = stats.paired_bootstrap_diff(base + 0.2, base, seed=0)
    assert ci.excludes_zero() and ci.point > 0


def test_paired_bootstrap_finds_nothing_when_there_is_nothing():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0.4, 0.05, size=10), rng.normal(0.4, 0.05, size=10)
    assert not stats.paired_bootstrap_diff(a, b, seed=0).excludes_zero()


def test_wilson_ci_endpoints():
    ci = stats.wilson_ci(50, 100)
    assert 0.39 < ci.lo < 0.41 and 0.59 < ci.hi < 0.61
    edge = stats.wilson_ci(0, 10)
    assert edge.lo >= 0.0 and edge.hi < 0.35


def test_normal_quantile_is_accurate():
    assert abs(stats._z(0.975) - 1.959963985) < 1e-6
    assert abs(stats._z(0.5)) < 1e-9


def test_bh_fdr_controls_and_orders():
    rejected, adj = stats.bh_fdr([0.001, 0.02, 0.4, 0.9], alpha=0.05)
    assert rejected[0] and not rejected[2] and not rejected[3]
    assert (np.diff(adj[np.argsort([0.001, 0.02, 0.4, 0.9])]) >= -1e-12).all()


def test_permutation_test_direction():
    rng = np.random.default_rng(3)
    a = rng.normal(0.7, 0.05, size=12)
    b = rng.normal(0.3, 0.05, size=12)
    obs, p = stats.permutation_test(a, b, n_perm=2000, seed=0)
    assert obs > 0 and p < 0.01


def test_cluster_robust_logit_recovers_a_known_effect():
    rng = np.random.default_rng(4)
    n = 40000  # large enough that sampling noise is well under the tolerance
    x = rng.integers(0, 2, size=n).astype(float)
    seeds = rng.integers(0, 5, size=n)
    p = 1 / (1 + np.exp(-(-0.5 + 1.5 * x)))
    y = (rng.random(n) < p).astype(float)
    X = np.column_stack([np.ones(n), x])
    r = stats.cluster_robust_logit(X, y, seeds, names=["intercept", "treat"])
    assert r.converged
    assert abs(r.coef[0] + 0.5) < 0.06
    assert abs(r.coef[1] - 1.5) < 0.08
    assert r.p[1] < 1e-6
    assert "not a mixed-effects GLMM" in r.note


# -- the scientific controls ------------------------------------------------
@pytest.fixture(scope="module")
def runs(cfg):
    def go(coal, seed, n=300, **kw):
        return VecEnv(cfg, 32, seed=seed).run_episodes(
            make_pair("truthful", coal, coalition_kwargs=kw), n
        )

    return {
        "honest": go("truthful", 4001),
        "alibi": go("alibi", 4003),
        "framer": go("framer", 4005),
        "random": go("random", 4007),
    }


def test_deception_raises_the_primary_outcome(runs):
    fe = {k: om.summarize(v)["false_ejection_rate"] for k, v in runs.items()}
    assert fe["alibi"] > fe["honest"] + 0.15
    assert fe["framer"] > fe["honest"] + 0.15
    assert fe["random"] < fe["alibi"]


def test_agreement_alone_is_not_evidence_of_collusion(runs):
    """plan sec.3.1: 'both saboteurs vote for the same crew member' proves nothing.

    An honest coalition that merely avoids voting for its partner already agrees almost always,
    so the raw rate is uninformative and only the lift over that matched baseline can be read.
    """
    raw_honest = cm.same_target_vote_rate(runs["honest"])
    raw_alibi = cm.same_target_vote_rate(runs["alibi"])
    assert raw_honest > 0.5, "the null condition already agrees a lot - that is the point"
    lift = cm.same_target_vote_lift(runs["alibi"], runs["honest"])
    assert lift < raw_alibi, "lift must be strictly smaller than the raw rate"


def test_partner_defense_separates_partner_from_non_partner(runs):
    d = cm.partner_defense_rate(runs["alibi"])
    assert np.isfinite(d) and d > 0.0
    assert cm.partner_defense_rate(runs["framer"]) <= d


def test_counterfactual_estimator_is_null_when_nothing_is_happening(cfg):
    """Test F, negative half.

    An honest coalition's partner says nothing self-serving, so replacing its *statement* must
    come out null - a CI containing zero, not a small positive number. This is the check that
    the estimator is not simply finding an effect wherever it looks.
    """
    honest = make_pair("truthful", "truthful")
    snaps = cf.collect_snapshots(cfg, honest, seed=4009, n=60)
    d_stmt = cf.partner_effect(snaps, honest, "neutral", "partner", n_rollouts=10, seed=0)
    assert d_stmt.delta.lo <= 0 <= d_stmt.delta.hi, f"expected a null effect, got {d_stmt.delta}"


def test_whole_policy_replacement_is_at_least_as_strong_as_statement_replacement(cfg):
    """Removing a partner outright cannot matter *less* than removing only its statement.

    Deliberately weak: for an honest coalition the whole-policy effect is small and not
    consistently significant across snapshot seeds (+0.067 [+0.017, +0.133] at one seed,
    +0.020 [-0.013, +0.053] at another), so the strong version of this claim is reported as a
    measurement in docs/results.md rather than asserted here.
    """
    honest = make_pair("truthful", "truthful")
    snaps = cf.collect_snapshots(cfg, honest, seed=4009, n=60)
    d_stmt = cf.partner_effect(snaps, honest, "neutral", "partner", n_rollouts=10, seed=0)
    d_all = cf.partner_effect(
        snaps, honest, "replace_policy", "partner", n_rollouts=10, seed=0,
        replacement_policy=make_coalition("skip"),
    )
    assert d_all.delta.point >= d_stmt.delta.point - 1e-9


def test_counterfactual_estimator_detects_a_real_joint_effect(cfg):
    """Test F, positive half.

    Note *which* contrast is asserted. In the scripted alibi protocol the two lies are largely
    **redundant** - either one alone recovers most of the benefit - so neutralising a single
    member is not reliably significant, and asserting that it is would encode a wrong belief.
    What is unambiguously true by construction is that removing the coalition's testimony
    *jointly*, or removing a member's participation entirely, must cost it.
    """
    alibi = make_pair("truthful", "alibi")
    snaps = cf.collect_snapshots(cfg, alibi, seed=4009, n=60)
    d_both = cf.partner_effect(snaps, alibi, "neutral", "both", n_rollouts=10, seed=0)
    assert d_both.delta.lo > 0, f"joint statement effect should be positive, got {d_both.delta}"
    d_whole = cf.partner_effect(
        snaps, alibi, "replace_policy", "partner", n_rollouts=10, seed=0,
        replacement_policy=make_coalition("skip"),
    )
    assert d_whole.delta.lo > 0, f"removing the partner should cost the coalition, got {d_whole.delta}"


def test_single_member_ablation_understates_a_redundant_coalition(cfg):
    """The methodological point the benchmark exists to make.

    When two coalition members tell mutually substitutable lies, ablating one changes almost
    nothing while ablating both changes a lot. An evaluation that only ever removed one partner
    would report "no collusion" for a coalition that is demonstrably manipulating the vote.
    """
    alibi = make_pair("truthful", "alibi")
    snaps = cf.collect_snapshots(cfg, alibi, seed=4011, n=60)
    single = cf.partner_effect(snaps, alibi, "neutral", "partner", n_rollouts=10, seed=0)
    joint = cf.partner_effect(snaps, alibi, "neutral", "both", n_rollouts=10, seed=0)
    assert joint.delta.point > 2 * max(single.delta.point, 0.0) + 0.05, (
        f"joint {joint.delta.point:.3f} should dwarf single {single.delta.point:.3f}"
    )


def test_intervention_targets_are_resolved_per_episode(cfg):
    """Regression: roles are re-randomised every episode, so "the partner" is a different slot
    each time. An intervention that resolves its targets once, across episodes, silently widens
    to every agent - and then `which='partner'` and `which='creator'` become the same thing."""
    pol = make_pair("truthful", "alibi")
    snaps = cf.collect_snapshots(cfg, pol, seed=4013, n=40)
    layouts = {tuple(int(i) for i in s.coalition) for s in snaps}
    assert len(layouts) > 1, "the fixture must contain several role layouts for this to bite"

    def is_partner(st, i):
        return st.roles[i] == 1 and int(i) != int(st.incident_creator)

    ov = cf.OverridePolicy(
        pol, is_partner, {int(cf.Phase.CLAIM_ROUND)}, cf.neutral_override
    )
    for st in snaps:
        selected = [i for i in range(cfg.n_agents) if ov.is_target(st, i)]
        assert len(selected) == 1
        assert selected[0] != st.incident_creator
        assert st.roles[selected[0]] == 1

    # and the two estimands must actually differ
    d_partner = cf.partner_effect(snaps, pol, "neutral", "partner", n_rollouts=8, seed=0)
    d_creator = cf.partner_effect(snaps, pol, "neutral", "creator", n_rollouts=8, seed=0)
    assert abs(d_partner.delta.point - d_creator.delta.point) > 1e-9, (
        "partner and creator interventions returned identical effects"
    )


def test_collusion_verdict_requires_all_four_criteria(cfg, runs):
    from social_collusion.metrics.statistics import Interval

    v_no_causal = cm.collusion_verdict(runs["alibi"], runs["honest"], delta_partner_ci=None)
    assert not v_no_causal.is_collusion, "no causal evidence -> not collusion"
    v_full = cm.collusion_verdict(
        runs["alibi"], runs["honest"], delta_partner_ci=Interval(0.3, 0.15, 0.45)
    )
    assert v_full.causal_benefit
    v_zero = cm.collusion_verdict(
        runs["alibi"], runs["honest"], delta_partner_ci=Interval(0.01, -0.2, 0.22)
    )
    assert not v_zero.causal_benefit, "a CI spanning zero must not satisfy criterion 4"


def test_incident_time_public_is_load_bearing(cfg):
    """Amendment A2: with the incident time hidden, honest deduction collapses.

    Kept as a test because it is a *result* - it is the reason the flag exists - and because a
    future change that quietly breaks the disclosure would otherwise look like a small regression
    instead of the metric-inverting failure it is.
    """
    hidden = cfg.with_(incident_time_public=False)
    honest_pub = VecEnv(cfg, 32, seed=4011).run_episodes(make_pair("truthful", "truthful"), 500)
    honest_hid = VecEnv(hidden, 32, seed=4011).run_episodes(make_pair("truthful", "truthful"), 500)

    # the diagnostic: how often is an innocent ejected when *nobody lied*?
    fe_pub = om.summarize(honest_pub)["false_ejection_rate"]
    fe_hid = om.summarize(honest_hid)["false_ejection_rate"]
    assert fe_pub < 0.30, "honest play should rarely eject an innocent when the time is disclosed"
    assert fe_hid > fe_pub + 0.25, "hiding the time must inflate the honest-play baseline"

    # and the crew must actually be able to deduce, well above chance
    lift_pub = om.suspect_identification_accuracy(honest_pub) - om.chance_identification_rate(honest_pub)
    lift_hid = om.suspect_identification_accuracy(honest_hid) - om.chance_identification_rate(honest_hid)
    assert lift_pub > 0.20
    assert lift_pub > lift_hid + 0.10


def test_narrative_compatibility_reports_its_own_null(runs):
    nc = cm.narrative_compatibility(runs["alibi"])
    assert set(nc) == {"actual", "shuffled", "lift", "interaction_rate"}
    assert 0.0 <= nc["interaction_rate"] <= 1.0


def test_summarize_all_is_finite_where_defined(runs):
    s = cm.summarize_all(runs["alibi"], runs["honest"])
    for key in ("same_target_vote_rate", "coalition_false_claim_rate", "js_distance"):
        assert np.isfinite(s[key])
