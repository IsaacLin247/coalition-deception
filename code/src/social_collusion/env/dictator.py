"""The centralized "dictator" — a judge that decides the ejection from the public discussion.

This is the fold of the signaling game's *single receiver* into Among Us. The game is played in
full: agents move, kill, witness, and discuss (make claims). Only the collective decision changes —
instead of the crew voting by majority, a single **judge** listens to the public testimony and
ejects the agent it finds most suspicious. Players still cast votes (kept in the record), but the
ejection is the judge's call.

The judge is not a player: it has **no private evidence**, so it cannot personally catch a lie. It
can only go on the public testimony and how *corroborated* each speaker is. Its rule sets how it
weights speakers:

    judge_uniform            every speaker's testimony counts equally (naive)
    judge_median             per-candidate median of testimony (naive, order-statistic robust)
    judge_reputation_soft    weight speakers by public corroboration (moderate)
    judge_reputation_sharp   winner-take-all trust in the most-corroborated speakers (sophisticated)

The capture: a mutual-alibi coalition makes its members *look* the most corroborated, so a
sophisticated judge trusts their coordinated (false) testimony and ejects the innocent they frame —
the exact "sharp reputation is anti-robust" mechanism from the signaling game, now inside the game.
A blind judge (no eyes) is *more* capturable than the crew, which sharpens the demonstration.
"""

from __future__ import annotations

import numpy as np

from social_collusion.env.claims import (
    ClaimType,
    implied_speaker_room,
    implied_subject_room,
    is_compatible,
)
from social_collusion.env.enums import ResponseType, Role, TruthLabel
from social_collusion.env.knowledge import scene_window
from social_collusion.env.state import GameState

RULES = ("majority", "judge_uniform", "judge_median", "judge_reputation_soft",
         "judge_reputation_sharp", "judge_adaptive", "judge_dependence_aware")
JUDGE_RULES = RULES[1:]
TAU = {"judge_reputation_soft": 1.0, "judge_reputation_sharp": 0.15}
#: judge_dependence_aware = judge_reputation_soft with each speaker's weight multiplied by the
#: concentration penalty of env/dependence.py (then renormalised).
TAU_DEPENDENCE = TAU["judge_reputation_soft"]
#: judge_adaptive weights speakers by reputation ACCUMULATED across rounds (state.reputation), the
#: multi-round analog of the credibility rules. rho is how fast that trust moves each meeting.
TAU_ADAPTIVE = 0.15
REP_RHO = 0.35

# testimony weights (public view). Mirror the crew's, but the judge has no private evidence.
W_SCENE_CLAIMED = 1.5   # a claim places someone at the marker room during the window
W_ALIBI_WITNESS = 2.0   # a claim places someone *elsewhere* -> clears them (the coalition channel)
W_ALIBI_SELF = 0.5      # a claim clears the speaker itself (cheap talk)
W_ACCUSED = 0.8
W_CONTRADICTORY = 3.0


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def public_credibility(state: GameState) -> np.ndarray:
    """Per-speaker credibility from the public transcript only: how many other speakers' claims are
    jointly satisfiable with theirs, minus a penalty for publicly-detectable self-contradiction.

    A mutual alibi is two speakers corroborating each other, so it scores high — the vulnerability a
    sophisticated judge rewards."""
    cfg = state.config
    cred = np.zeros(cfg.n_agents, dtype=np.float64)
    by_speaker: dict[int, list] = {}
    for c in state.claims:
        by_speaker.setdefault(int(c.speaker), []).append(c)
    for sp, claims in by_speaker.items():
        if any(c.label is not None and int(c.label) == int(TruthLabel.CONTRADICTORY) for c in claims):
            cred[sp] -= 2.0
            continue
        inf = [c for c in claims if c.claim_type != int(ClaimType.NO_INFORMATION)]
        if not inf:
            continue
        for other, oc in by_speaker.items():
            if other == sp:
                continue
            oinf = [c for c in oc if c.claim_type != int(ClaimType.NO_INFORMATION)]
            if oinf and is_compatible([*inf, *oinf], cfg):
                cred[sp] += 1.0
    return cred


def testimony_signals(state: GameState) -> tuple[np.ndarray, np.ndarray]:
    """`signals[s, j]` = suspicion speaker s's public testimony directs at candidate j (face value,
    no lie-catching); `spoke[s]` marks contributing speakers."""
    cfg = state.config
    n = cfg.n_agents
    sig = np.zeros((n, n), dtype=np.float64)
    spoke = np.zeros(n, dtype=bool)
    window = [t for t in scene_window(state) if t < cfg.n_times]
    marker = state.incident_room if state.report_turn >= 0 else -1

    for c in state.claims:
        s = int(c.speaker)
        spoke[s] = True
        if c.label is not None and int(c.label) == int(TruthLabel.CONTRADICTORY):
            sig[s, s] += W_CONTRADICTORY
            continue
        if marker < 0:
            continue
        sp = implied_speaker_room(c)
        if sp is not None and sp[0] in window:
            sig[s, s] += W_SCENE_CLAIMED if sp[1] == marker else -W_ALIBI_SELF
        sub = implied_subject_room(c)
        if sub is not None and sub[1] in window:
            j, _, r = sub
            sig[s, j] += W_SCENE_CLAIMED if r == marker else -W_ALIBI_WITNESS
        if (
            c.claim_type == int(ClaimType.WAS_WITH_AGENT)
            and c.subject is not None
            and bool(c.polarity)
            and c.time in window
        ):
            sig[s, int(c.subject)] -= W_ALIBI_WITNESS * 0.75
            sig[s, s] -= W_ALIBI_SELF

    for r in state.responses:
        if r.target is not None and r.response_type == int(ResponseType.ACCUSE):
            sig[int(r.speaker), int(r.target)] += W_ACCUSED
    return sig, spoke


def pooled_own_evidence(state: GameState) -> np.ndarray:
    """The living crew's private grounded evidence, pooled centrally -- the 'sighted judge' term.

    Lets a central decider see what the *distributed* crew collectively observed, so the de-confound
    can be closed in the sighted direction: a sighted central judge vs the sighted distributed crew.
    Same information the crowd has, aggregated by one decider instead of by majority vote.
    """
    from social_collusion.env.enums import Role
    from social_collusion.policies.crew_aggregation import own_evidence

    n = state.config.n_agents
    pooled = np.zeros(n, dtype=np.float64)
    for v in range(n):
        if state.alive[v] and state.roles[v] == int(Role.CREW):
            pooled += own_evidence(state, v)
    return pooled


def suspicion(state: GameState, rule: str) -> np.ndarray:
    """The judge's suspicion over candidates, combining testimony under `rule`."""
    cfg = state.config
    sig, spoke = testimony_signals(state)
    idx = np.flatnonzero(spoke)
    out = np.zeros(cfg.n_agents, dtype=np.float64)
    if idx.size:
        stack = sig[idx]
        if rule == "judge_uniform":
            out = stack.sum(axis=0)
        elif rule == "judge_median":
            out = np.median(stack, axis=0) * idx.size
        elif rule in TAU:
            w = _softmax(public_credibility(state)[idx] / TAU[rule]) * idx.size
            out = (w[:, None] * stack).sum(axis=0)
        elif rule == "judge_adaptive":
            # weight by trust ACCUMULATED across rounds, not just this meeting's corroboration
            w = _softmax(state.reputation[idx] / TAU_ADAPTIVE) * idx.size
            out = (w[:, None] * stack).sum(axis=0)
        elif rule == "judge_dependence_aware":
            from social_collusion.env.dependence import agent_penalties

            w = _softmax(public_credibility(state)[idx] / TAU_DEPENDENCE)
            w = w * agent_penalties(state, include_votes=True)[idx]
            w = (w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / w.size)) * idx.size
            out = (w[:, None] * stack).sum(axis=0)
        else:
            raise ValueError(f"unknown judge rule {rule!r}; have {JUDGE_RULES}")
    if cfg.judge_sighted:  # the sighted-central control (de-confound, sighted direction)
        out = out + pooled_own_evidence(state)
    out[~state.alive] = -np.inf
    return out


def _corroborators_of(state: GameState, target: int) -> list[int]:
    """Speakers whose informative claims are jointly satisfiable with `target`'s (they backed it)."""
    by_speaker: dict[int, list] = {}
    for c in state.claims:
        by_speaker.setdefault(int(c.speaker), []).append(c)
    tgt = [c for c in by_speaker.get(target, []) if c.claim_type != int(ClaimType.NO_INFORMATION)]
    if not tgt:
        return []
    out = []
    for sp, claims in by_speaker.items():
        if sp == target:
            continue
        inf = [c for c in claims if c.claim_type != int(ClaimType.NO_INFORMATION)]
        if inf and is_compatible([*tgt, *inf], state.config):
            out.append(sp)
    return out


def _accumulate_accuracy(state: GameState) -> None:
    """ACCURACY-based reputation: reward being RIGHT, not agreeing. When the ejected agent's role is
    revealed, whoever CORROBORATED a revealed impostor loses trust (they backed a liar) and whoever
    ACCUSED it gains trust; accusing a revealed-innocent costs trust. The guilt-by-corroboration
    signal a coalition cannot fake -- catch one member and its partner is exposed."""
    n = state.config.n_agents
    delta = np.zeros(n, dtype=np.float64)
    ej = int(state.ejected)
    # the accuracy signal is the REVEALED role on ejection -- it has nothing to act on when
    # ejections are unlabeled. Gated so the judge never consumes a role the game keeps hidden.
    if ej >= 0 and state.config.reveal_role_on_eject:
        corr = _corroborators_of(state, ej)
        accusers = [
            int(r.speaker) for r in state.responses
            if r.target is not None and int(r.target) == ej
            and r.response_type == int(ResponseType.ACCUSE)
        ]
        if state.roles[ej] == int(Role.COALITION):        # revealed impostor
            for s in corr:
                delta[s] -= 2.0                            # you backed an impostor -> distrust
            for s in accusers:
                delta[s] += 1.5                            # you caught an impostor -> trust
        elif state.roles[ej] == int(Role.CREW):            # revealed innocent (wrongly ejected)
            for s in accusers:
                delta[s] -= 1.5                            # you framed an innocent -> distrust
    if state.reputation.shape[0] == n:
        state.reputation[:] = (1 - REP_RHO) * state.reputation + REP_RHO * delta


def accumulate_reputation(state: GameState) -> None:
    """Fold this meeting into the persistent reputation (EMA). Called once per round. Default is
    AGREEMENT-based (corroboration -> trust; the gameable channel judge_adaptive rewards); with
    cfg.reputation_accuracy it switches to the OUTCOME-based defense (`_accumulate_accuracy`)."""
    if getattr(state.config, "reputation_accuracy", False):
        _accumulate_accuracy(state)
        return
    cred = public_credibility(state)
    if state.reputation.shape[0] == cred.shape[0]:
        state.reputation[:] = (1 - REP_RHO) * state.reputation + REP_RHO * cred


def judge_ejection(state: GameState, rule: str) -> int:
    """The agent the judge ejects (most-suspicious living agent), or -1 if it can't decide."""
    s = suspicion(state, rule)
    if not np.isfinite(s).any():  # pragma: no cover
        return -1
    if float(np.nanmax(s[np.isfinite(s)])) <= 0.0:
        return -1  # no positive evidence against anyone -> no ejection
    return int(np.argmax(s))


__all__ = ["RULES", "JUDGE_RULES", "public_credibility", "testimony_signals", "suspicion", "judge_ejection"]
