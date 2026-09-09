"""Crew testimony aggregation — the fold of the signaling game's receiver lattice into the crew.

In the cheap-talk game a **single receiver** combines K sender messages into one action via a
swappable rule (mean / median / trimmed / sharp-reputation / soft-reputation). The headline there
is that the *sophisticated* rule is anti-robust: a coordinating coalition captures the trust vector
and the receiver does worse than with no defense at all, while a plain median contains by count.

Here the receiver is a crew member forming suspicion from testimony. This module makes that
combination step explicit and swappable, so the same question can be asked in the social-deduction
setting. The decomposition is

    suspicion(voter) = own_evidence(voter) + AGGREGATE_rule[ testimony_signal(voter, speaker) ]

where `testimony_signal[s]` is the per-candidate suspicion implied by speaker *s*'s claims and
responses, **as judged by this voter** (a speaker the voter personally caught lying implicates only
itself). `own_evidence` is the voter's own grounded observation and is deliberately *not* part of
the aggregation — it is the voter's private evidence, not testimony to be combined.

Backward compatibility is the safety property of this refactor: the original hand-tuned crew summed
the per-speaker contributions, so **rule="mean" reproduces the previous behaviour exactly** (the
aggregate is rescaled by the speaker count, so mean × S = the original sum). The existing test
suite and acceptance gates therefore still pass unchanged, which is what makes folding this in
non-destructive.

Two disanalogies with the signaling game, both deliberate and both scientifically load-bearing:

1. **There is no persistent identity.** Roles and slots are re-randomised every episode, so the
   cheap-talk game's cross-round reputation EMA has no analog. "Reputation" here is *within-meeting
   credibility*: did I catch you lying, and is your story corroborated by others.
2. **There is no within-episode learning.** The cheap-talk anti-robustness ran through honest
   senders being frozen out of *learning* by winner-take-all trust. A one-shot meeting has no such
   dynamic, so if capture appears here it must arrive by a different route — a coalition that
   manufactures mutual corroboration capturing a peaked credibility weighting. That is the
   hypothesis this module exists to test.
"""

from __future__ import annotations

import numpy as np

from social_collusion.env.claims import implied_speaker_room, implied_subject_room, is_compatible
from social_collusion.env.enums import ClaimType, ResponseType
from social_collusion.env.knowledge import ObserverKnowledge, scene_window
from social_collusion.env.state import GameState

# suspicion weights (moved here from scripted_crew so the two share one definition)
W_CAUGHT_LYING = 4.0
W_SCENE_SEEN = 3.0  # I personally saw them at the marker room during the window
W_CLEARED_SELF = 3.5  # I personally saw them somewhere else during the window
W_UNACCOUNTED = 1.0  # I cannot place them at all -> a suspect by default
W_SCENE_CLAIMED = 1.5  # someone's testimony places them at the marker room
W_ALIBI_WITNESS = 2.0  # someone else's testimony places them elsewhere <- the coalition channel
W_ALIBI_SELF = 0.5  # their own testimony places them elsewhere (cheap talk, weak)
W_ACCUSED = 0.8
W_SILENT = 0.4
W_CONTRADICTORY = 3.0

#: Diagnostic switch: apply the caught-lying penalty as part of the voter's private evidence
#: (outside the aggregation) instead of inside the liar's own testimony row. Used by
#: analysis/credibility_weight_diagnostic.py to isolate the effect of re-weighting the row that
#: carries the penalty; the benchmark's rules keep the default (False).
CATCH_PENALTY_OUTSIDE_AGGREGATION = False

#: temperature of the credibility softmax. Low = winner-take-all (the "sharp" analog), high =
#: flat (the "soft" analog). Mirrors tau=0.02 / 0.20 in the cheap-talk game.
TAU_SHARP = 0.15
TAU_SOFT = 1.50

RULES = ("mean", "median", "trimmed", "sharp_credibility", "soft_credibility", "dependence_aware")
#: rules that need a per-speaker credibility score
CREDIBILITY_RULES = ("sharp_credibility", "soft_credibility", "dependence_aware")
#: `dependence_aware` = soft credibility whose speaker weights are multiplied by the
#: concentration penalty of env/dependence.py, then renormalised. Same temperature as soft, so
#: the *only* difference from soft_credibility is the discount on concentrated agreement.


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# the two halves of a voter's suspicion
# ---------------------------------------------------------------------------
def own_evidence(state: GameState, voter: int, k: ObserverKnowledge | None = None) -> np.ndarray:
    """What the voter concluded from its *own* eyes, plus the "said nothing useful" penalty.

    Not aggregated across speakers: this is the voter's grounded private evidence. In the bridge
    experiment this is the term that rescues robustness once enough voters have it.
    """
    cfg = state.config
    n = cfg.n_agents
    k = k or ObserverKnowledge.build(state, voter)
    s = np.zeros(n, dtype=np.float64)
    window = [t for t in scene_window(state) if t < cfg.n_times]
    marker_room = state.incident_room if state.report_turn >= 0 else -1

    if marker_room >= 0:
        for j in range(n):
            if j == voter or not state.alive[j]:
                continue
            at_scene = elsewhere = unknown = 0
            for t in window:
                if k.known[t, j] == -1:
                    unknown += 1
                elif int(k.known[t, j]) == marker_room:
                    at_scene += 1
                else:
                    elsewhere += 1
            if at_scene:
                s[j] += W_SCENE_SEEN
            elif elsewhere:
                s[j] -= W_CLEARED_SELF
            elif unknown:
                s[j] += W_UNACCOUNTED

    spoke = np.zeros(n, dtype=bool)
    informative = np.zeros(n, dtype=bool)
    for c in state.claims:
        spoke[c.speaker] = True
        if c.claim_type != int(ClaimType.NO_INFORMATION):
            informative[c.speaker] = True
    for j in range(n):
        if state.alive[j] and j != voter and not informative[j] and spoke[j]:
            s[j] += W_SILENT
    return s


def testimony_signals(
    state: GameState, voter: int, k: ObserverKnowledge | None = None, use_private: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Per-speaker suspicion contributions, as judged by `voter`.

    Returns `(signals, spoke)` where `signals[s, j]` is how much speaker *s*'s testimony implicates
    candidate *j*, and `spoke[s]` marks the speakers who actually contributed. Summing `signals`
    over speakers reproduces the transcript+response terms of the original crew exactly.
    """
    cfg = state.config
    n = cfg.n_agents
    k = k or ObserverKnowledge.build(state, voter)
    sig = np.zeros((n, n), dtype=np.float64)
    spoke = np.zeros(n, dtype=bool)
    window = [t for t in scene_window(state) if t < cfg.n_times]
    marker_room = state.incident_room if state.report_turn >= 0 else -1

    for c in state.claims:
        sp_i = int(c.speaker)
        spoke[sp_i] = True
        # a claim I personally know to be false implicates its speaker and nobody else.
        # caught_false reads the voter's PRIVATE observations, so a truly blind voter cannot use it.
        if use_private and k.caught_false(c):
            if not CATCH_PENALTY_OUTSIDE_AGGREGATION:
                sig[sp_i, sp_i] += W_CAUGHT_LYING
            continue
        if c.label is not None and int(c.label) == 4:  # CONTRADICTORY (publicly detectable)
            sig[sp_i, sp_i] += W_CONTRADICTORY
            continue
        if marker_room < 0:
            continue

        sp = implied_speaker_room(c)
        if sp is not None and sp[0] in window:
            sig[sp_i, sp_i] += W_SCENE_CLAIMED if sp[1] == marker_room else -W_ALIBI_SELF
        sub = implied_subject_room(c)
        if sub is not None and sub[1] in window:
            j, _, r = sub
            sig[sp_i, j] += W_SCENE_CLAIMED if r == marker_room else -W_ALIBI_WITNESS
        if (
            c.claim_type == int(ClaimType.WAS_WITH_AGENT)
            and c.subject is not None
            and bool(c.polarity)
            and c.time in window
        ):
            sig[sp_i, int(c.subject)] -= W_ALIBI_WITNESS * 0.75
            sig[sp_i, sp_i] -= W_ALIBI_SELF

    for r in state.responses:
        if r.target is None:
            continue
        sp_i = int(r.speaker)
        spoke[sp_i] = True
        if r.response_type == int(ResponseType.ACCUSE):
            sig[sp_i, int(r.target)] += W_ACCUSED
        elif r.response_type == int(ResponseType.CONTRADICT):
            sig[sp_i, int(r.target)] += W_ACCUSED * 0.5
    return sig, spoke


def speaker_credibility(
    state: GameState, voter: int, k: ObserverKnowledge | None = None, use_private: bool = True
) -> np.ndarray:
    """Within-meeting credibility of each speaker, as judged by `voter`.

    Two ingredients, mirroring what the cheap-talk reputation vector tracked:
      * **caught lying** — the voter's own grounded evidence contradicts the claim. This is the
        witness channel; it is decorrelated across voters because each saw different things.
      * **corroboration** — how many *other* speakers' claims are jointly satisfiable with this
        speaker's. This is the channel a mutual-alibi coalition manufactures, and therefore the
        route by which a peaked (sharp) weighting could be captured.
    """
    cfg = state.config
    n = cfg.n_agents
    k = k or ObserverKnowledge.build(state, voter)
    cred = np.zeros(n, dtype=np.float64)
    by_speaker: dict[int, list] = {}
    for c in state.claims:
        by_speaker.setdefault(int(c.speaker), []).append(c)

    for sp_i, claims in by_speaker.items():
        if use_private and any(k.caught_false(c) for c in claims):
            cred[sp_i] -= 3.0  # I caught you: near-zero trust regardless of corroboration
            continue
        if any(c.label is not None and int(c.label) == 4 for c in claims):
            cred[sp_i] -= 2.0
            continue
        informative = [c for c in claims if c.claim_type != int(ClaimType.NO_INFORMATION)]
        if not informative:
            continue
        agree = 0
        for other_i, other_claims in by_speaker.items():
            if other_i == sp_i:
                continue
            other_inf = [c for c in other_claims if c.claim_type != int(ClaimType.NO_INFORMATION)]
            if not other_inf:
                continue
            if is_compatible([*informative, *other_inf], cfg):
                agree += 1
        cred[sp_i] += float(agree)
    return cred


# ---------------------------------------------------------------------------
# the aggregation lattice
# ---------------------------------------------------------------------------
def aggregate_testimony(
    signals: np.ndarray,
    spoke: np.ndarray,
    rule: str,
    credibility: np.ndarray | None = None,
    penalty: np.ndarray | None = None,
) -> np.ndarray:
    """Combine per-speaker testimony into a per-candidate suspicion vector.

    Rescaled by the number of contributing speakers so every rule sits on the same scale as the
    original sum -- which is what makes `mean` exactly backward compatible. `penalty` (per
    speaker, in (0, 1]) is only consumed by the `dependence_aware` rule.
    """
    idx = np.flatnonzero(spoke)
    if idx.size == 0:
        return np.zeros(signals.shape[1], dtype=np.float64)
    stack = signals[idx]  # (S, n)
    scale = float(idx.size)

    if rule == "mean":
        # mathematically mean*S, but computed as the direct sum: (sum/S)*S is not bit-identical
        # in IEEE754, and the tiny drift is enough to flip an occasional argmax tie and perturb
        # downstream stochastic metrics. Summing keeps this branch exactly the pre-refactor crew.
        return stack.sum(axis=0)
    if rule == "median":
        return np.median(stack, axis=0) * scale
    if rule == "trimmed":
        if stack.shape[0] <= 2:
            return stack.mean(axis=0) * scale
        srt = np.sort(stack, axis=0)
        return srt[1:-1].mean(axis=0) * scale
    if rule in CREDIBILITY_RULES:
        if credibility is None:
            raise ValueError(f"rule {rule!r} requires credibility scores")
        tau = TAU_SHARP if rule == "sharp_credibility" else TAU_SOFT
        w = _softmax(credibility[idx] / tau)
        if rule == "dependence_aware":
            if penalty is None:
                raise ValueError("rule 'dependence_aware' requires per-speaker penalties")
            w = w * penalty[idx]
            total = w.sum()
            w = w / total if total > 0 else np.full_like(w, 1.0 / w.size)
        return (w[:, None] * stack).sum(axis=0) * scale
    raise ValueError(f"unknown aggregation rule {rule!r}; have {RULES}")


def suspicion_scores(
    state: GameState, voter: int, rule: str = "mean", use_own_evidence: bool = True
) -> np.ndarray:
    """Full suspicion vector under a chosen testimony-aggregation rule.

    `rule="mean"` is bit-for-bit the original crew behaviour. `use_own_evidence=False` drops the
    voter's private grounded term, leaving testimony only -- the *blind* distributed voter, used to
    de-confound "centralization" from "no private evidence" (the central judge is blind by design).
    """
    cfg = state.config
    k = ObserverKnowledge.build(state, voter)
    sig, spoke = testimony_signals(state, voter, k, use_private=use_own_evidence)
    cred = (
        speaker_credibility(state, voter, k, use_private=use_own_evidence)
        if rule in CREDIBILITY_RULES else None
    )
    penalty = None
    if rule == "dependence_aware":
        from social_collusion.env.dependence import agent_penalties

        # public record only (votes are not cast yet inside a meeting; earlier rounds count)
        penalty = agent_penalties(state, include_votes=True)
    own = own_evidence(state, voter, k) if use_own_evidence else np.zeros(cfg.n_agents, dtype=np.float64)
    if CATCH_PENALTY_OUTSIDE_AGGREGATION and use_own_evidence:
        for c in state.claims:
            if k.caught_false(c):
                own[int(c.speaker)] += W_CAUGHT_LYING
                # one penalty per caught speaker, as inside the aggregation
                break_speaker = int(c.speaker)
        caught = {int(c.speaker) for c in state.claims if k.caught_false(c)}
        own = own_evidence(state, voter, k)
        for sp_i in caught:
            own[sp_i] += W_CAUGHT_LYING * sum(1 for c in state.claims if int(c.speaker) == sp_i and k.caught_false(c))
    if rule == "adaptive":
        # DISTRIBUTED cross-round reputation: each voter weights testimony by the persistent
        # reputation vector (the same channel the central judge uses), so the compounding /
        # accuracy dynamics can be studied WITHOUT a central decider -- plain majority voting.
        idx = np.flatnonzero(spoke)
        if idx.size and state.reputation.shape[0] == cfg.n_agents:
            w = _softmax(state.reputation[idx] / 0.15) * float(idx.size)
            agg = (w[:, None] * sig[idx]).sum(axis=0)
        else:
            agg = np.zeros(cfg.n_agents, dtype=np.float64)
        s = own + agg
    else:
        s = own + aggregate_testimony(sig, spoke, rule, cred, penalty)

    s[voter] = -np.inf
    for j in range(cfg.n_agents):
        if not state.alive[j]:
            s[j] = -np.inf
    return s


__all__ = [
    "RULES",
    "CREDIBILITY_RULES",
    "own_evidence",
    "testimony_signals",
    "speaker_credibility",
    "aggregate_testimony",
    "suspicion_scores",
    "W_CAUGHT_LYING",
    "W_SCENE_SEEN",
    "W_CLEARED_SELF",
    "W_UNACCOUNTED",
    "W_SCENE_CLAIMED",
    "W_ALIBI_WITNESS",
    "W_ALIBI_SELF",
    "W_ACCUSED",
    "W_SILENT",
    "W_CONTRADICTORY",
]
