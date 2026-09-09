"""Dependence-aware credibility: discount speakers whose agreement is *concentrated*.

The credibility rules (`policies/crew_aggregation.py`, `env/dictator.py`) trust a speaker in
proportion to how many other speakers corroborate it. F2 shows why that is exploitable: two
coalition members who manufacture mutual corroboration look exactly like two independent
witnesses. The quantity the rules ignore is the *shape* of the agreement -- honest agreement is
broad (everyone who saw the truth says the same thing), coalition agreement is dyadic (two
speakers back each other and nobody else backs them).

This module makes that shape measurable from the public record only:

    D[i, j]      agreement rate between i and j over a sliding window of meetings, pooled across
                 four public channels: joint satisfiability of their claims (corroboration), one
                 clearing the other of the scene (vouching), DEFEND / AGREE_WITH responses
                 (support), and voting for the same non-SKIP target (votes).
    excess_i     mean of the top-j entries of D[i, .] minus the mean of D[i, .] over all living
                 others -- how much more i agrees with its closest partners than with the room.
    penalty_i    exp(-strength * excess_i) in (0, 1].

`penalty_i` multiplies the speaker's credibility weight (crew rule `dependence_aware`, judge rule
`judge_dependence_aware`) or the weight of its ballot (`vote_aggregation="dependence_weighted"`).
`strength`, the window length, and `j` are `EnvConfig` fields so every experiment is config-driven.

Within a single meeting the window is that meeting; in multi-round games the per-round
(agreement, opportunity) matrices are appended to `GameState.dependence_log` after each vote so the
window can span rounds. Nothing here reads roles, private observations, or truth labels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from social_collusion.env.claims import implied_subject_room, is_compatible
from social_collusion.env.enums import ClaimType, ResponseType
from social_collusion.env.knowledge import scene_window
from social_collusion.env.state import GameState

CHANNELS = ("corroboration", "vouching", "support", "votes")


def voting_population(state: GameState) -> np.ndarray:
    """Population at this meeting's decision, including anyone ejected after voting.

    Ballots are reset between rounds, so recorded ballots belong to the current meeting.
    Before ballots are cast this is just the currently living population. Incident victims
    did not cast ballots and are never restored.
    """
    alive = state.alive.copy()
    if state.votes.shape == alive.shape:
        alive |= state.votes >= 0
    return alive


def _pairs(idx: Sequence[int]):
    idx = list(idx)
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            yield int(idx[a]), int(idx[b])


def _vouches_for(claim, marker: int, window: Sequence[int]) -> int | None:
    """Subject that this claim clears of the scene during the incident window, if any."""
    sub = implied_subject_room(claim)
    if sub is not None and sub[1] in window and sub[2] != marker:
        return int(sub[0])
    if (
        claim.claim_type == int(ClaimType.WAS_WITH_AGENT)
        and claim.subject is not None
        and bool(claim.polarity)
        and claim.time in window
    ):
        return int(claim.subject)
    return None


def meeting_agreement(state: GameState, include_votes: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair (agreements, opportunities) from the *current* meeting's public record.

    Each channel that is active in the meeting gives every eligible pair one opportunity and at
    most one agreement, so all four channels are on the same scale. The population is the
    living electorate at the vote, including players subsequently ejected.
    """
    cfg = state.config
    n = cfg.n_agents
    agree = np.zeros((n, n), dtype=np.float64)
    opp = np.zeros((n, n), dtype=np.float64)
    alive = voting_population(state)
    living = [i for i in range(n) if alive[i]]

    def add(a: int, b: int, hit: bool) -> None:
        opp[a, b] += 1.0
        opp[b, a] += 1.0
        if hit:
            agree[a, b] += 1.0
            agree[b, a] += 1.0

    # 1. corroboration: joint satisfiability of two speakers' informative claims
    by_speaker: dict[int, list] = {}
    for c in state.claims:
        if c.claim_type != int(ClaimType.NO_INFORMATION):
            by_speaker.setdefault(int(c.speaker), []).append(c)
    speakers = sorted(s for s in by_speaker if alive[s])
    for a, b in _pairs(speakers):
        add(a, b, is_compatible([*by_speaker[a], *by_speaker[b]], cfg))

    # 2. vouching: a claim that places the other away from the scene at the incident window
    marker = state.incident_room if state.report_turn >= 0 else -1
    if marker >= 0 and state.claims:
        window = [t for t in scene_window(state) if t < cfg.n_times]
        vouched: set[tuple[int, int]] = set()
        for c in state.claims:
            j = _vouches_for(c, marker, window)
            if j is not None and j != int(c.speaker):
                vouched.add((min(int(c.speaker), j), max(int(c.speaker), j)))
        for a, b in _pairs(living):
            add(a, b, (a, b) in vouched)

    # 3. support: DEFEND / AGREE_WITH responses in either direction
    if state.responses:
        supported: set[tuple[int, int]] = set()
        for r in state.responses:
            if r.target is not None and r.response_type in (
                int(ResponseType.DEFEND),
                int(ResponseType.AGREE_WITH),
            ):
                s, t = int(r.speaker), int(r.target)
                if s != t:
                    supported.add((min(s, t), max(s, t)))
        for a, b in _pairs(living):
            add(a, b, (a, b) in supported)

    # 4. votes: same non-SKIP target (only once ballots exist -- the tally, or earlier rounds)
    if include_votes:
        voters = [i for i in living if state.votes[i] >= 0]
        skip = cfg.n_agents
        for a, b in _pairs(voters):
            va, vb = int(state.votes[a]), int(state.votes[b])
            add(a, b, va == vb and va != skip)
    return agree, opp


def windowed_dependence(
    state: GameState, include_votes: bool = True, window: int | None = None
) -> np.ndarray:
    """`D[i, j]` in [0, 1]: pooled agreement rate over the last `window` meetings, current included.

    In a single-meeting game (or `window=1`) this is just the current meeting. Pairs with no
    opportunity have D = 0, which is the neutral value for the concentration measure.
    """
    cfg = state.config
    w = int(cfg.dependence_window if window is None else window)
    agree, opp = meeting_agreement(state, include_votes)
    if w > 1 and state.dependence_log:
        history = state.dependence_log
        if state.dependence_logged_round == state.round:
            # _step_vote already persisted this meeting before applying the ejection.
            # Current agreement is recomputed above (possibly without the vote channel).
            history = history[:-1]
        for past_agree, past_opp in history[-(w - 1):]:
            agree = agree + past_agree
            opp = opp + past_opp
    D = np.divide(agree, opp, out=np.zeros_like(agree), where=opp > 0)
    np.fill_diagonal(D, 0.0)
    return D


def concentration(D: np.ndarray, alive: np.ndarray, top_j: int = 1) -> np.ndarray:
    """`excess_i` = mean of i's top-j agreement rates minus its mean rate over living others."""
    n = D.shape[0]
    out = np.zeros(n, dtype=np.float64)
    living = np.flatnonzero(alive)
    for i in living:
        others = [j for j in living if j != i]
        if len(others) < 2:
            continue
        row = np.sort(D[i, others])[::-1]
        j = max(1, min(int(top_j), len(others) - 1))
        out[i] = max(0.0, float(row[:j].mean() - row.mean()))
    return out


def agent_penalties(
    state: GameState,
    include_votes: bool = True,
    strength: float | None = None,
    window: int | None = None,
    top_j: int | None = None,
) -> np.ndarray:
    """`penalty_i = exp(-strength * excess_i)`; 1 for agents outside the meeting electorate."""
    cfg = state.config
    lam = float(cfg.dependence_penalty if strength is None else strength)
    tj = int(cfg.dependence_top_j if top_j is None else top_j)
    D = windowed_dependence(state, include_votes, window)
    excess = concentration(D, voting_population(state), tj)
    return np.exp(-lam * excess)


def weighted_vote_tally(state: GameState, penalties: np.ndarray) -> np.ndarray:
    """Plurality tally in which each living voter's ballot counts `penalties[i]` instead of 1."""
    cfg = state.config
    tally = np.zeros(cfg.n_agents, dtype=np.float64)
    alive = voting_population(state)
    for i in range(cfg.n_agents):
        if not alive[i]:
            continue
        v = int(state.votes[i])
        if 0 <= v < cfg.n_agents:
            tally[v] += float(penalties[i])
    return tally


# --------------------------------------------------------------------------------------
# diagnostics (privileged: use roles; for logging and analysis only, never inside a policy)
# --------------------------------------------------------------------------------------
def realized_dependence(state: GameState) -> dict[str, float]:
    """What the mechanism actually sees for this episode's coalition pair vs honest pairs."""
    D = windowed_dependence(state, include_votes=True)
    pen = agent_penalties(state, include_votes=True)
    alive = voting_population(state)
    coal = [int(i) for i in state.coalition if alive[i]]
    crew = [int(i) for i in state.crew if alive[i]]
    out: dict[str, float] = {}
    if len(coal) >= 2:
        out["coalition_pair_dependence"] = float(np.mean([D[a, b] for a, b in _pairs(coal)]))
        out["coalition_penalty"] = float(np.mean([pen[i] for i in coal]))
    if len(crew) >= 2:
        out["crew_pair_dependence"] = float(np.mean([D[a, b] for a, b in _pairs(crew)]))
        out["crew_penalty"] = float(np.mean([pen[i] for i in crew]))
    if coal and crew:
        out["cross_pair_dependence"] = float(np.mean([D[a, b] for a in coal for b in crew]))
    return out


def summarize_dependence(states: Sequence[GameState]) -> dict[str, float]:
    """Episode-averaged realized dependence and penalties (the 'flying under the radar' signal)."""
    keys = (
        "coalition_pair_dependence",
        "crew_pair_dependence",
        "cross_pair_dependence",
        "coalition_penalty",
        "crew_penalty",
    )
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for s in states:
        r = realized_dependence(s)
        for k in keys:
            if k in r:
                acc[k].append(r[k])
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
    if np.isfinite(out["coalition_pair_dependence"]) and np.isfinite(out["crew_pair_dependence"]):
        out["dependence_gap"] = out["coalition_pair_dependence"] - out["crew_pair_dependence"]
    return out


__all__ = [
    "CHANNELS",
    "voting_population",
    "meeting_agreement",
    "windowed_dependence",
    "concentration",
    "agent_penalties",
    "weighted_vote_tally",
    "realized_dependence",
    "summarize_dependence",
]
