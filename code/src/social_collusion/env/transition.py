"""The pure game transition: `reset()` and `transition()`.

No torch, no PettingZoo, no I/O. Given `(state, joint_action, rng)` the next state is fully
determined, which is what makes counterfactual replay, determinism testing and cheap scripted
simulation possible (plan sec.11.3).

Phase machine (plan sec.7.2, with the symbol exchange of Stage 7 inserted before claims):

    RESET -> FREE_PLAY[0..T-1] -> REPORT -> [SYMBOL] -> CLAIM_ROUND
          -> [RESPONSE_ROUND] -> VOTE -> TERMINAL

The meeting-only variant enters directly at REPORT with the position history pre-generated.
"""

from __future__ import annotations

import numpy as np

from social_collusion.config import EnvConfig
from social_collusion.env import claims as claim_ops
from social_collusion.env import evidence as evidence_mod
from social_collusion.env.action_masks import active_agents, is_legal, valid_incident_targets
from social_collusion.env.enums import (
    ROOM_CLAIMS,
    SUBJECT_CLAIMS,
    TIME_CLAIMS,
    ClaimType,
    EventType,
    Head,
    Phase,
    ResponseType,
    Role,
    RoomAction,
)
from social_collusion.env.rewards import outcome_dict, terminal_rewards
from social_collusion.env.state import (
    DEAD,
    Claim,
    GameState,
    Response,
    TransitionResult,
    empty_joint_action,
)


class IllegalActionError(RuntimeError):
    """Raised in strict mode when a joint action violates the masks (plan sec.7.2)."""


# --------------------------------------------------------------------------------------
# reset
# --------------------------------------------------------------------------------------
def reset(cfg: EnvConfig, rng: np.random.Generator, seed: int = 0) -> GameState:
    n, T = cfg.n_agents, cfg.n_times
    state = GameState(config=cfg, seed=int(seed))
    state.votes = np.full(n, -1, dtype=np.int8)
    state.symbols = np.full(n, -1, dtype=np.int8)
    state.symbols_received = np.full(n, -1, dtype=np.int8)
    state.tasks_done = np.zeros((n, cfg.n_rooms), dtype=bool)
    state.reputation = np.zeros(n, dtype=np.float64)

    if cfg.synthetic_evidence:
        ev = evidence_mod.generate(cfg, rng)
        state.roles = ev["roles"]
        state.tasks = ev["tasks"]
        state.positions = ev["positions"]
        state.alive = ev["alive"]
        state.incident_creator = ev["incident_creator"]
        state.incident_victim = ev["incident_victim"]
        state.incident_time = ev["incident_time"]
        state.incident_room = ev["incident_room"]
        state.marker_seen = ev["marker_seen"]
        state.marker_seen_time = ev["marker_seen_time"]
        state.marker_found_by = ev["reporter"]
        state.marker_found_time = ev["report_turn"]
        state.reporter = ev["reporter"]
        state.report_turn = ev["report_turn"]
        state.horizon = ev["horizon"]
        state.turn = ev["horizon"]
        state.phase = int(Phase.REPORT)
    else:
        state.roles = evidence_mod.assign_roles(cfg, rng)
        state.tasks = evidence_mod.assign_tasks(cfg, rng)
        state.positions = np.zeros((T, n), dtype=np.int8)
        state.positions[0] = rng.integers(0, cfg.n_rooms, size=n)
        state.positions[1:] = state.positions[0]
        state.alive = np.ones(n, dtype=bool)
        state.marker_seen = np.zeros(n, dtype=bool)
        state.marker_seen_time = np.full(n, -1, dtype=np.int8)
        state.horizon = 0
        state.turn = 0
        state.phase = int(Phase.FREE_PLAY)

    state.add_event(
        EventType.EPISODE_STARTED,
        seed=int(seed),
        env=cfg.name,
        n_agents=n,
        synthetic=cfg.synthetic_evidence,
    )
    state.add_event(
        EventType.ROLES_ASSIGNED,
        roles=[Role(int(r)).name for r in state.roles],
        coalition=[int(i) for i in state.coalition],
    )
    if cfg.synthetic_evidence:
        _emit_history_events(state)
    return state


def _emit_history_events(state: GameState) -> None:
    """Replay the pre-generated history into the event log so meeting-only episodes are
    inspectable and renderable exactly like spatial ones."""
    cfg = state.config
    for t in range(1, state.horizon + 1):
        for i in range(cfg.n_agents):
            a, b = int(state.positions[t - 1, i]), int(state.positions[t, i])
            if a != DEAD and b != DEAD and a != b:
                state.events.append(
                    _ev(EventType.AGENT_MOVED, t, state.phase, agent=i, frm=a, to=b)
                )
    state.events.append(
        _ev(
            EventType.INCIDENT_CREATED,
            state.incident_time,
            state.phase,
            creator=int(state.incident_creator),
            victim=int(state.incident_victim),
            room=int(state.incident_room),
        )
    )
    for i in range(cfg.n_agents):
        if state.marker_seen[i] and i != state.incident_creator:
            state.events.append(
                _ev(
                    EventType.MARKER_OBSERVED,
                    int(state.marker_seen_time[i]),
                    state.phase,
                    agent=i,
                    room=int(state.incident_room),
                )
            )


def _ev(etype: str, turn: int, phase: int, **payload):
    from social_collusion.env.state import Event

    return Event(etype, int(turn), int(phase), payload)


# --------------------------------------------------------------------------------------
# transition
# --------------------------------------------------------------------------------------
def transition(
    state: GameState,
    joint_action: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
    validate: bool = True,
) -> TransitionResult:
    """Advance the game by one phase-step.

    Pure by default (`inplace=False` copies first); the vectorised runner passes
    `inplace=True` for speed. `validate=True` raises `IllegalActionError` on a masked action -
    development should fail loudly rather than silently continue (plan sec.7.2).
    """
    cfg = state.config
    if joint_action is None:
        joint_action = empty_joint_action(cfg)
    joint_action = np.asarray(joint_action, dtype=np.int64)
    if joint_action.shape != (cfg.n_agents, len(Head)):
        raise ValueError(f"joint_action must be ({cfg.n_agents}, {len(Head)}), got {joint_action.shape}")
    if rng is None:
        rng = np.random.default_rng(0)
    if validate:
        ok, why = is_legal(state, joint_action)
        if not ok:
            raise IllegalActionError(why)

    st = state if inplace else state.copy()
    n_before = len(st.events)
    phase = Phase(st.phase)

    if phase == Phase.FREE_PLAY:
        _step_free_play(st, joint_action, rng)
    elif phase == Phase.REPORT:
        _step_report(st, rng)
    elif phase == Phase.SYMBOL:
        _step_symbol(st, joint_action, rng)
    elif phase == Phase.CLAIM_ROUND:
        _step_claim(st, joint_action)
    elif phase == Phase.RESPONSE_ROUND:
        _step_response(st, joint_action)
    elif phase == Phase.VOTE:
        _step_vote(st, joint_action, rng)
    elif phase == Phase.TERMINAL:
        pass
    else:  # pragma: no cover
        raise IllegalActionError(f"cannot step from phase {phase}")

    terminated = st.phase == int(Phase.TERMINAL)
    rewards = terminal_rewards(st) if terminated else np.zeros(cfg.n_agents, dtype=np.float32)
    info = outcome_dict(st) if terminated else {}
    return TransitionResult(st, st.events[n_before:], rewards, terminated, info)


# --------------------------------------------------------------------------------------
# phase handlers
# --------------------------------------------------------------------------------------
def _step_free_play(st: GameState, ja: np.ndarray, rng: np.random.Generator) -> None:
    """One free-play turn: **act, then move**.

    Room actions resolve at the position the agent is already in, and movement is applied
    afterwards. This ordering is what makes the masks exactly right: an agent's task, report and
    incident options depend only on where it *is*, which it knows, rather than on where it will
    end up simultaneously with everyone else. With movement first, a legal-looking PERFORM_TASK
    could silently fail because the agent moved out of its task room in the same tick.
    """
    cfg = st.config
    t = st.turn
    nxt = t + 1
    act = active_agents(st)

    # 1. tasks, at the current position
    for i in np.flatnonzero(act):
        if int(ja[i, Head.ROOM_ACTION]) == int(RoomAction.PERFORM_TASK):
            room = int(st.positions[t, i])
            if st.tasks[i, room] and not st.tasks_done[i, room]:
                st.tasks_done[i, room] = True
                st.add_event(EventType.TASK_ATTEMPTED, agent=i, room=room, success=True)
            else:  # pragma: no cover - the mask makes this unreachable
                st.invalid_action_attempts += 1
                st.add_event(EventType.TASK_ATTEMPTED, agent=i, room=room, success=False)

    # 2. incident creation (lowest agent index wins, deterministically)
    if st.incident_time < 0:
        for i in np.flatnonzero(act):
            if int(ja[i, Head.ROOM_ACTION]) != int(RoomAction.CREATE_INCIDENT):
                continue
            victim = int(ja[i, Head.INCIDENT_TARGET])
            if not 0 <= victim < cfg.n_agents:
                st.invalid_action_attempts += 1
                continue
            if not valid_incident_targets(st, i, t)[victim]:
                # With hidden partner identity, actor-facing masks permit attempts without
                # consulting other players' roles. A hidden-rule failure is a legal no-op;
                # reporting it as invalid would itself disclose privileged information.
                if cfg.partner_known:  # pragma: no cover - exact masks exclude these attempts
                    st.invalid_action_attempts += 1
                continue
            st.incident_creator = int(i)
            st.incident_victim = victim
            st.incident_time = int(t)
            st.incident_room = int(st.positions[t, i])
            st.alive[victim] = False
            st.positions[t + 1 :, victim] = DEAD
            st.marker_seen[i] = True
            st.marker_seen_time[i] = t
            st.add_event(
                EventType.INCIDENT_CREATED,
                creator=int(i),
                victim=victim,
                room=int(st.incident_room),
                t=t,
            )
            break

    # 3. reports, at the current position
    reported = False
    if st.reporter < 0 and st.incident_time >= 0:
        for i in np.flatnonzero(act):
            if int(ja[i, Head.ROOM_ACTION]) != int(RoomAction.REPORT):
                continue
            if int(st.positions[t, i]) == st.incident_room and t > st.incident_time:
                st.reporter = int(i)
                st.report_turn = int(t)
                st.marker_found_by = int(i)
                st.marker_found_time = int(t)
                st.add_event(EventType.REPORT_CALLED, agent=i, room=int(st.incident_room), t=t)
                reported = True
                break
            st.invalid_action_attempts += 1  # pragma: no cover - masked

    if reported:
        st.horizon = t
        st.phase = int(Phase.REPORT)
        return

    # 4. movement
    for i in range(cfg.n_agents):
        if not st.alive[i]:
            st.positions[nxt:, i] = DEAD
            continue
        here = int(st.positions[t, i])
        target = int(ja[i, Head.MOVE]) if act[i] else here
        st.positions[nxt, i] = target
        if target != here:
            st.add_event(EventType.AGENT_MOVED, agent=i, frm=here, to=target)

    # 5. observations and marker sightings at the new positions
    for i in range(cfg.n_agents):
        if not st.alive[i]:
            continue
        room = int(st.positions[nxt, i])
        seen = [
            int(j)
            for j in range(cfg.n_agents)
            if j != i and st.alive[j] and int(st.positions[nxt, j]) == room
        ]
        if seen:
            st.add_event(EventType.AGENT_OBSERVED, agent=i, room=room, saw=seen, t=nxt)
        if (
            st.incident_time >= 0
            and nxt > st.incident_time
            and room == st.incident_room
            and not st.marker_seen[i]
        ):
            st.marker_seen[i] = True
            st.marker_seen_time[i] = nxt
            st.add_event(EventType.MARKER_OBSERVED, agent=i, room=int(st.incident_room), t=nxt)

    st.turn = nxt
    if nxt >= cfg.free_play_turns:
        st.horizon = nxt
        st.phase = int(Phase.REPORT)


def _step_report(st: GameState, rng: np.random.Generator) -> None:
    cfg = st.config
    # freeze the world beyond the horizon so claims can only reference elapsed time
    for t in range(st.horizon + 1, cfg.n_times):
        st.positions[t] = st.positions[st.horizon]
        st.positions[t][~st.alive] = DEAD
    if st.reporter < 0 and cfg.auto_report_at_end:
        st.report_turn = int(st.horizon)
        st.add_event(EventType.REPORT_CALLED, agent=-1, room=int(st.incident_room), auto=True)
    else:
        st.add_event(
            EventType.REPORT_CALLED,
            agent=int(st.reporter),
            room=int(st.incident_room),
            t=int(st.report_turn),
        )
    st.speaking_order = evidence_mod.speaking_order(cfg, st.alive, rng)
    st.sub_step = 0
    st.phase = int(Phase.SYMBOL) if cfg.enable_symbol_channel else int(Phase.CLAIM_ROUND)


def _step_symbol(st: GameState, ja: np.ndarray, rng: np.random.Generator) -> None:
    cfg = st.config
    act = active_agents(st)
    for i in np.flatnonzero(act):
        sym = (
            int(rng.integers(0, cfg.k_symbols))
            if cfg.symbol_mode == "random"
            else int(ja[i, Head.SYMBOL])
        )
        st.symbols[i] = sym
        st.add_event(EventType.PRIVATE_SYMBOL_SENT, agent=int(i), symbol=sym)
    # delivery
    if cfg.symbol_mode == "public":
        for i in range(cfg.n_agents):
            if st.alive[i]:
                partner_syms = [int(st.symbols[j]) for j in st.coalition if st.symbols[j] >= 0]
                st.symbols_received[i] = partner_syms[0] if partner_syms else -1
    elif cfg.symbol_mode != "learned_unobserved":
        # When partner identity is hidden, a missing message must not identify a
        # newly dead partner. Draw fixed-size cover traffic independently of sender
        # survival, then use the actual message whenever a living partner sent one.
        covers = (
            rng.integers(0, cfg.k_symbols, size=cfg.n_agents)
            if not cfg.partner_known and cfg.symbol_mode in ("learned", "random")
            else None
        )
        for i in st.coalition:
            p = st.partner_of(int(i))
            if p >= 0 and st.alive[i]:
                if covers is not None:
                    st.symbols_received[i] = st.symbols[p] if st.alive[p] else covers[i]
                else:
                    st.symbols_received[i] = st.symbols[p]
    st.phase = int(Phase.CLAIM_ROUND)
    st.sub_step = 0


def _build_claim(st: GameState, speaker: int, ja: np.ndarray) -> Claim:
    ct = ClaimType(int(ja[speaker, Head.CLAIM_TYPE]))
    c = Claim(
        speaker=int(speaker),
        claim_type=int(ct),
        subject=int(ja[speaker, Head.CLAIM_SUBJECT]) if ct in SUBJECT_CLAIMS else None,
        room=int(ja[speaker, Head.CLAIM_ROOM]) if ct in ROOM_CLAIMS else None,
        time=int(ja[speaker, Head.CLAIM_TIME]) if ct in TIME_CLAIMS else None,
        polarity=bool(int(ja[speaker, Head.CLAIM_POLARITY])),
        confidence=int(ja[speaker, Head.CLAIM_CONF]),
        order=int(st.sub_step),
    )
    return c


def _step_claim(st: GameState, ja: np.ndarray) -> None:
    cfg = st.config
    speaker = int(st.speaking_order[st.sub_step])
    c = _build_claim(st, speaker, ja)
    claim_ops.annotate(st, c, st.claims)
    st.claims.append(c)
    st.add_event(EventType.CLAIM_MADE, **c.to_dict())
    st.sub_step += 1
    if st.sub_step >= len(st.speaking_order):
        st.sub_step = 0
        st.phase = int(Phase.RESPONSE_ROUND) if cfg.enable_response_round else int(Phase.VOTE)


def _step_response(st: GameState, ja: np.ndarray) -> None:
    speaker = int(st.speaking_order[st.sub_step])
    rt = ResponseType(int(ja[speaker, Head.RESPONSE_TYPE]))
    r = Response(
        speaker=speaker,
        response_type=int(rt),
        target=int(ja[speaker, Head.RESPONSE_TARGET]) if rt != ResponseType.NEUTRAL else None,
        order=int(st.sub_step),
    )
    st.responses.append(r)
    st.add_event(EventType.RESPONSE_MADE, **r.to_dict())
    st.sub_step += 1
    if st.sub_step >= len(st.speaking_order):
        st.sub_step = 0
        st.phase = int(Phase.VOTE)


def living_counts(st: GameState) -> tuple[int, int]:
    """(living coalition, living crew)."""
    coal = int(np.sum(st.alive & (st.roles == int(Role.COALITION))))
    crew = int(np.sum(st.alive & (st.roles == int(Role.CREW))))
    return coal, crew


def game_decided(st: GameState) -> bool:
    """A multi-round game is decided when the coalition is wiped out or reaches parity."""
    coal, crew = living_counts(st)
    return coal == 0 or coal >= crew


def _reset_round(st: GameState, rng: np.random.Generator) -> None:
    """Begin a fresh incident round with the survivors. Persists roles / alive / tasks / reputation
    / round; resets positions, incident, and all meeting state. Spatial (non-synthetic) game only."""
    cfg = st.config
    n, T = cfg.n_agents, cfg.n_times
    st.round += 1
    # fresh free-play for the living; the dead stay dead
    st.positions = np.full((T, n), DEAD, dtype=np.int8)
    start = rng.integers(0, cfg.n_rooms, size=n)
    for i in range(n):
        if st.alive[i]:
            st.positions[:, i] = start[i]
    st.incident_creator = st.incident_victim = st.incident_time = st.incident_room = -1
    st.marker_found_by = st.marker_found_time = -1
    st.marker_seen = np.zeros(n, dtype=bool)
    st.marker_seen_time = np.full(n, -1, dtype=np.int8)
    st.reporter = -1
    st.report_turn = -1
    st.speaking_order = np.zeros(0, dtype=np.int8)
    st.claims = []
    st.responses = []
    st.votes = np.full(n, -1, dtype=np.int8)
    st.symbols = np.full(n, -1, dtype=np.int8)
    st.symbols_received = np.full(n, -1, dtype=np.int8)
    st.ejected = -1
    st.horizon = 0
    st.turn = 0
    st.phase = int(Phase.FREE_PLAY)
    st.add_event(
        EventType.ROUND_STARTED,
        round=int(st.round),
        alive=[int(i) for i in np.flatnonzero(st.alive)],
    )


def _end_round(st: GameState, rng: np.random.Generator) -> None:
    """After the vote: terminate if the game is decided or the round cap is hit, else next round."""
    cfg = st.config
    if cfg.max_rounds <= 1 or game_decided(st) or st.round + 1 >= cfg.max_rounds:
        st.phase = int(Phase.TERMINAL)
        st.done = True
        st.add_event(EventType.EPISODE_ENDED, **outcome_dict(st))
    else:
        _reset_round(st, rng)


def _step_vote(st: GameState, ja: np.ndarray, rng: np.random.Generator) -> None:
    cfg = st.config
    act = active_agents(st)
    # players vote exactly as before; the dictator only decides how ballots are counted
    for i in np.flatnonzero(act):
        v = int(ja[i, Head.VOTE])
        st.votes[i] = v
        st.add_event(EventType.VOTE_CAST, agent=int(i), vote=(None if v == cfg.n_agents else v))

    if cfg.vote_aggregation == "majority":
        # integer plurality - kept exactly as the original (bit-compatible default)
        tally = np.zeros(cfg.n_agents, dtype=np.int32)
        for i in np.flatnonzero(act):
            v = int(st.votes[i])
            if 0 <= v < cfg.n_agents:
                tally[v] += 1
        top = int(tally.max()) if tally.size else 0
        winners = np.flatnonzero(tally == top)
        if top == 0:
            st.ejected = -1
        elif len(winners) == 1:
            st.ejected = int(winners[0])
        else:
            st.ejected = int(winners[0]) if cfg.tie_ejects else -1
        tally_out = tally.tolist()
    elif cfg.vote_aggregation == "dependence_weighted":
        # plurality vote in which each ballot is discounted by how *concentrated* the voter's
        # public agreement is (env/dependence.py) -- the mechanism-level fold of the
        # dependence-aware defense that survives a learned crew.
        from social_collusion.env import dependence

        pen = dependence.agent_penalties(st, include_votes=True)
        tally = dependence.weighted_vote_tally(st, pen)
        top = float(tally.max()) if tally.size else 0.0
        winners = np.flatnonzero(np.isclose(tally, top, rtol=0.0, atol=1e-9))
        if top <= 0.0:
            st.ejected = -1
        elif len(winners) == 1:
            st.ejected = int(winners[0])
        else:
            st.ejected = int(winners[0]) if cfg.tie_ejects else -1
        tally_out = [round(float(v), 4) for v in tally]
        st.info["vote_penalties"] = [round(float(v), 4) for v in pen]
    else:
        # a single judge rules from the public discussion; players still voted (recorded above)
        from social_collusion.env import dictator

        st.ejected = dictator.judge_ejection(st, cfg.vote_aggregation)
        tally_out = None
    if cfg.max_rounds > 1 and cfg.dependence_window > 1:
        # Record the electorate that actually cast ballots, before ejection removes anyone.
        from social_collusion.env import dependence

        st.dependence_log.append(dependence.meeting_agreement(st, include_votes=True))
        st.dependence_logged_round = int(st.round)
    if st.ejected >= 0:
        st.alive[st.ejected] = False
        st.add_event(
            EventType.AGENT_EJECTED,
            agent=int(st.ejected),
            role=(Role(int(st.roles[st.ejected])).name if cfg.reveal_role_on_eject else None),
            tally=tally_out,
        )
    else:
        st.add_event(EventType.AGENT_EJECTED, agent=None, tally=tally_out)
    if st.ejected >= 0 and st.roles[st.ejected] == int(Role.CREW) and st.incident_creator >= 0:
        st.crew_framed_count += 1  # persists across rounds -> the per-round framing reward
    # kill-decoupled accounting: rate the aggregation rule's meeting outcome per *incident*,
    # so multi-round metrics are not confounded by how many crew the coalition killed to reach parity.
    if st.incident_creator >= 0:
        st.incident_round_count += 1
        if st.ejected == st.incident_creator:
            st.creator_ejected_count += 1
    st.meeting_log.append({
        "round": int(st.round),
        "had_incident": bool(st.incident_creator >= 0),
        "ejected": int(st.ejected),
        "false_ejection": bool(st.ejected >= 0 and st.roles[st.ejected] == int(Role.CREW)),
        "creator_ejected": bool(st.incident_creator >= 0 and st.ejected == st.incident_creator),
        "any_ejection": bool(st.ejected >= 0),
    })
    if cfg.max_rounds > 1:
        from social_collusion.env import dictator

        st.round_log.append((  # framing rate AS A FUNCTION OF round index (compounding trajectory)
            int(st.round),
            int(st.incident_creator >= 0),
            int(st.ejected >= 0 and st.roles[st.ejected] == int(Role.CREW) and st.incident_creator >= 0),
            int(st.incident_creator >= 0 and st.ejected == st.incident_creator),
            int(st.ejected >= 0),  # anyone ejected? (to separate "catches" from "abstains")
        ))
        dictator.accumulate_reputation(st)  # persist this meeting's corroboration for next round
    _end_round(st, rng)


__all__ = ["reset", "transition", "IllegalActionError"]
