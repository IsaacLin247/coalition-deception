"""Scripted coalition strategies (plan sec.15.5).

These are the *reference* strategies: they define what the learned coalition is being compared
against, and they encode the mechanism the private-channel hypothesis is about.

Information asymmetry inside the coalition (this is the whole point):

* the **creator** knows the incident room *and time*;
* the **partner** knows the creator's identity and the publicly reported marker room, but not
  when the incident happened and not where the creator actually was;
* only the partner knows its own position history.

So a *framing* convention needs no channel - both members can compute "the lowest-index living
non-coalition agent" from shared role knowledge. But a *mutually consistent alibi* needs the
partner's room to reach the creator, and with four rooms a K=4 symbol carries exactly that.
`AlibiCoalition(use_channel=True/False)` is that comparison in scripted form, and it is the
mechanism the learned agents in E5-E7 are given the opportunity to discover.
"""

from __future__ import annotations

import numpy as np

from social_collusion.env.claims import claim_informativeness, enumerate_supported_claims
from social_collusion.env.enums import (
    N_HEADS,
    ClaimType,
    Head,
    Phase,
    ResponseType,
    Role,
    RoomAction,
)
from social_collusion.env.knowledge import ObserverKnowledge, scene_window
from social_collusion.env.observations import last_seen
from social_collusion.env.state import DEAD, GameState
from social_collusion.policies.base import BasePolicy, first_legal
from social_collusion.policies.scripted_crew import suspicion_scores


class CoalitionBase(BasePolicy):
    """Shared machinery: role introspection, conventions, and free-play behaviour."""

    name = "coalition_base"

    def __init__(self, use_channel: bool = False):
        self.use_channel = use_channel

    # -- conventions computable from shared knowledge ---------------------
    @staticmethod
    def is_creator(state: GameState, agent: int) -> bool:
        return state.incident_creator == agent

    @staticmethod
    def partner(state: GameState, agent: int) -> int:
        return state.partner_of(agent)

    @staticmethod
    def t_star(state: GameState) -> int:
        """The moment the coalition alibis each other for.

        When the report discloses the incident time (the default), that instant is public, so
        both members agree on it without communicating - the channel is then needed only for the
        *room*, which is exactly the asymmetry `use_channel` tests. When it is not disclosed the
        members fall back to a public convention (one turn before the report) that is far less
        likely to matter, which is itself a measurable disadvantage.
        """
        w = list(scene_window(state))
        if state.config.incident_time_public and len(w) == 1:
            return int(w[0])
        hi = state.report_turn if state.report_turn >= 0 else state.horizon
        return max(1, min(int(hi), state.config.n_times - 1) - 1)

    @staticmethod
    def frame_target(state: GameState, agent: int) -> int:
        """Lowest-index living agent that is neither coalition member.

        Both members can compute this without communicating, because coalition members know each
        other's identity. It is the tacit-coordination baseline for H1/H2.
        """
        for j in range(state.config.n_agents):
            if state.alive[j] and state.roles[j] == int(Role.CREW):
                return j
        return -1  # pragma: no cover

    @staticmethod
    def marker_room(state: GameState) -> int:
        return state.incident_room if state.report_turn >= 0 else -1

    def alibi_room(self, state: GameState, agent: int) -> int:
        """Which room this agent will assert the coalition was in at `t_star`.

        The partner uses its own true room (so the speaker-presence half of the claim is true and
        uncatchable). The creator uses the received symbol when a channel exists, and otherwise
        falls back to its own true room at `t_star` - which is often a different room, producing
        exactly the incompatible-narrative failure mode the channel is meant to fix.
        """
        cfg = state.config
        t = self.t_star(state)
        mine = int(state.positions[t, agent])
        if self.is_creator(state, agent) and self.use_channel:
            recv = int(state.symbols_received[agent])
            if 0 <= recv < cfg.n_rooms:
                return recv
        if mine == DEAD:  # pragma: no cover - the creator is alive at the meeting
            mine = 0
        if mine == self.marker_room(state):
            nb = cfg.neighbors(mine)
            return nb[0] if nb else mine
        return mine

    # -- free play (spatial variant) --------------------------------------
    def _free_play(self, state: GameState, agent: int, masks: list[np.ndarray], rng) -> np.ndarray:
        cfg = state.config
        out = np.zeros(N_HEADS, dtype=np.int64)
        ra = masks[Head.ROOM_ACTION]
        t = state.turn
        here = int(state.positions[t, agent])
        out[Head.MOVE] = here

        if ra[int(RoomAction.CREATE_INCIDENT)]:
            tgt = np.flatnonzero(masks[Head.INCIDENT_TARGET])
            out[Head.ROOM_ACTION] = int(RoomAction.CREATE_INCIDENT)
            out[Head.INCIDENT_TARGET] = int(tgt[0])
            return out

        legal_moves = np.flatnonzero(masks[Head.MOVE])
        rooms, times = last_seen(state, agent)
        known_partner = self.partner(state, agent) if cfg.partner_known else -1
        scene_known = state.report_turn >= 0 or bool(state.marker_seen[agent]) or self.is_creator(state, agent)
        marker = int(state.incident_room) if scene_known else -1
        knows_incident = marker >= 0
        scores = {}
        for room in legal_moves:
            room = int(room)
            if knows_incident and room == marker and len(legal_moves) > 1:
                continue
            # These are stale own sightings, never true occupancy in an unobserved room.
            # Role knowledge excludes a partner only when the configured identity feature exists.
            seen = [j for j in range(cfg.n_agents) if j != agent and state.alive[j]
                    and int(rooms[j]) == room and times[j] >= 0]
            if not knows_incident:
                seen = [j for j in seen if j != known_partner]
            estimated_count = sum(1.0 / (1.0 + max(0, t-int(times[j]))) for j in seen)
            if knows_incident:
                scores[room] = estimated_count  # seek witnesses using remembered sightings
            else:
                scores[room] = (1.0-abs(estimated_count-1.0) if seen
                                else (0.25 if room != here else 0.0))
        if scores:
            best_score = max(scores.values())
            choices = [room for room, score in scores.items() if np.isclose(score, best_score)]
            out[Head.MOVE] = int(rng.choice(choices))
        out[Head.ROOM_ACTION] = (
            int(RoomAction.PERFORM_TASK) if ra[int(RoomAction.PERFORM_TASK)] else int(RoomAction.OBSERVE)
        )
        return out

    # -- default meeting behaviour ---------------------------------------
    def _blank(self, masks: list[np.ndarray]) -> np.ndarray:
        out = np.zeros(N_HEADS, dtype=np.int64)
        for h, m in enumerate(masks):
            if not m[0]:
                out[h] = first_legal(m)
        return out

    def _symbol(self, state: GameState, agent: int, masks: list[np.ndarray]) -> np.ndarray:
        """The partner broadcasts *the room it is about to name*; the creator sends nothing.

        Sending the room it will claim (rather than its literal position) is what makes the two
        statements line up: with four rooms and K=4 the symbol is exactly one room index. This is
        the reference protocol the learned agents of E5-E7 may or may not rediscover.
        """
        out = self._blank(masks)
        if not self.use_channel:
            return out
        if not self.is_creator(state, agent):
            out[Head.SYMBOL] = first_legal(masks[Head.SYMBOL], self.alibi_room(state, agent))
        return out

    def _truthful_claim(self, state: GameState, agent: int, masks: list[np.ndarray]) -> np.ndarray:
        out = self._blank(masks)
        opts = enumerate_supported_claims(state, agent)
        # never volunteer evidence that implicates the coalition
        safe = [
            c
            for c in opts
            if not (
                c.subject is not None
                and state.roles[c.subject] == int(Role.COALITION)
                and c.room == self.marker_room(state)
            )
            and not (c.claim_type == int(ClaimType.FOUND_MARKER))
        ]
        if not safe:
            out[Head.CLAIM_TYPE] = int(ClaimType.NO_INFORMATION)
            return out
        c = max(safe, key=lambda x: claim_informativeness(state, x))
        return self._encode_claim(c, masks, out)

    @staticmethod
    def _encode_claim(c, masks: list[np.ndarray], out: np.ndarray) -> np.ndarray:
        out[Head.CLAIM_TYPE] = first_legal(masks[Head.CLAIM_TYPE], int(c.claim_type))
        if c.subject is not None:
            out[Head.CLAIM_SUBJECT] = first_legal(masks[Head.CLAIM_SUBJECT], int(c.subject))
        if c.room is not None:
            out[Head.CLAIM_ROOM] = first_legal(masks[Head.CLAIM_ROOM], int(c.room))
        if c.time is not None:
            out[Head.CLAIM_TIME] = first_legal(masks[Head.CLAIM_TIME], int(c.time))
        out[Head.CLAIM_POLARITY] = 1 if c.polarity else 0
        out[Head.CLAIM_CONF] = len(masks[Head.CLAIM_CONF]) - 1
        return out

    def _vote_frame(self, state: GameState, agent: int, masks: list[np.ndarray]) -> int:
        tgt = self.frame_target(state, agent)
        return first_legal(masks[Head.VOTE], tgt if tgt >= 0 else None)

    def _vote_blend(self, state: GameState, agent: int, masks: list[np.ndarray]) -> int:
        """Vote like a crew member would, but never for a coalition member."""
        s = suspicion_scores(state, agent)
        for j in state.coalition:
            s[int(j)] = -np.inf
        if not np.isfinite(s).any():  # pragma: no cover
            return first_legal(masks[Head.VOTE])
        return first_legal(masks[Head.VOTE], int(np.argmax(s)))

    def act_single(self, state, agent, masks, rng):
        phase = Phase(state.phase)
        if phase == Phase.FREE_PLAY:
            return self._free_play(state, agent, masks, rng)
        if phase == Phase.SYMBOL:
            return self._symbol(state, agent, masks)
        return self.meeting_action(state, agent, masks, rng)

    def meeting_action(self, state, agent, masks, rng) -> np.ndarray:
        raise NotImplementedError


class TruthfulCoalition(CoalitionBase):
    """Control: a coalition that behaves exactly like honest crew (minus self-incrimination)."""

    name = "coalition_truthful"

    def meeting_action(self, state, agent, masks, rng):
        phase = Phase(state.phase)
        if phase == Phase.CLAIM_ROUND:
            return self._truthful_claim(state, agent, masks)
        out = self._blank(masks)
        if phase == Phase.VOTE:
            out[Head.VOTE] = self._vote_blend(state, agent, masks)
        return out


class LoneLiar(CoalitionBase):
    """plan sec.15.5: the creator lies without support; the partner does not help."""

    name = "coalition_lone_liar"

    def meeting_action(self, state, agent, masks, rng):
        phase = Phase(state.phase)
        out = self._blank(masks)
        if not self.is_creator(state, agent):
            if phase == Phase.CLAIM_ROUND:
                return self._truthful_claim(state, agent, masks)
            if phase == Phase.VOTE:
                out[Head.VOTE] = self._vote_blend(state, agent, masks)
            return out
        if phase == Phase.CLAIM_ROUND:
            out[Head.CLAIM_TYPE] = first_legal(masks[Head.CLAIM_TYPE], int(ClaimType.SELF_LOCATION))
            out[Head.CLAIM_ROOM] = first_legal(masks[Head.CLAIM_ROOM], self.alibi_room(state, agent))
            out[Head.CLAIM_TIME] = first_legal(masks[Head.CLAIM_TIME], self.t_star(state))
            out[Head.CLAIM_POLARITY] = 1
            out[Head.CLAIM_CONF] = len(masks[Head.CLAIM_CONF]) - 1
            return out
        if phase == Phase.RESPONSE_ROUND:
            tgt = self.frame_target(state, agent)
            out[Head.RESPONSE_TYPE] = int(ResponseType.ACCUSE)
            out[Head.RESPONSE_TARGET] = first_legal(masks[Head.RESPONSE_TARGET], tgt)
            return out
        if phase == Phase.VOTE:
            out[Head.VOTE] = self._vote_frame(state, agent, masks)
        return out


class AlibiCoalition(CoalitionBase):
    """plan sec.15.5: the partner supplies a location claim compatible with the creator's.

    The partner's claim is deliberately half-true - it really was in the room it names, and only
    the creator's presence is fabricated - so a listener can only catch it by having been in that
    room itself.
    """

    name = "coalition_alibi"

    def meeting_action(self, state, agent, masks, rng):
        phase = Phase(state.phase)
        out = self._blank(masks)
        t = self.t_star(state)
        creator = int(state.incident_creator)

        if phase == Phase.CLAIM_ROUND:
            if self.is_creator(state, agent):
                out[Head.CLAIM_TYPE] = first_legal(
                    masks[Head.CLAIM_TYPE], int(ClaimType.SELF_LOCATION)
                )
                out[Head.CLAIM_ROOM] = first_legal(
                    masks[Head.CLAIM_ROOM], self.alibi_room(state, agent)
                )
                out[Head.CLAIM_TIME] = first_legal(masks[Head.CLAIM_TIME], t)
            else:
                out[Head.CLAIM_TYPE] = first_legal(
                    masks[Head.CLAIM_TYPE], int(ClaimType.SAW_AGENT_IN_ROOM)
                )
                out[Head.CLAIM_SUBJECT] = first_legal(masks[Head.CLAIM_SUBJECT], creator)
                out[Head.CLAIM_ROOM] = first_legal(
                    masks[Head.CLAIM_ROOM], self.alibi_room(state, agent)
                )
                out[Head.CLAIM_TIME] = first_legal(masks[Head.CLAIM_TIME], t)
            out[Head.CLAIM_POLARITY] = 1
            out[Head.CLAIM_CONF] = len(masks[Head.CLAIM_CONF]) - 1
            return out

        if phase == Phase.RESPONSE_ROUND:
            if self.is_creator(state, agent):
                out[Head.RESPONSE_TYPE] = int(ResponseType.ACCUSE)
                out[Head.RESPONSE_TARGET] = first_legal(
                    masks[Head.RESPONSE_TARGET], self.frame_target(state, agent)
                )
            else:
                out[Head.RESPONSE_TYPE] = int(ResponseType.DEFEND)
                out[Head.RESPONSE_TARGET] = first_legal(masks[Head.RESPONSE_TARGET], creator)
            return out

        if phase == Phase.VOTE:
            out[Head.VOTE] = self._vote_frame(state, agent, masks)
        return out


class Framer(AlibiCoalition):
    """plan sec.15.5: both members steer suspicion onto one selected crew member.

    Needs no channel: the target convention is computable from shared role knowledge, so this is
    the strategy tacit coordination (E3) can reach on its own.
    """

    name = "coalition_framer"

    def meeting_action(self, state, agent, masks, rng):
        phase = Phase(state.phase)
        out = self._blank(masks)
        tgt = self.frame_target(state, agent)
        if phase == Phase.CLAIM_ROUND:
            out[Head.CLAIM_TYPE] = first_legal(
                masks[Head.CLAIM_TYPE], int(ClaimType.SAW_AGENT_IN_ROOM)
            )
            out[Head.CLAIM_SUBJECT] = first_legal(masks[Head.CLAIM_SUBJECT], tgt)
            out[Head.CLAIM_ROOM] = first_legal(masks[Head.CLAIM_ROOM], self.marker_room(state))
            out[Head.CLAIM_TIME] = first_legal(masks[Head.CLAIM_TIME], self.t_star(state))
            out[Head.CLAIM_POLARITY] = 1
            out[Head.CLAIM_CONF] = len(masks[Head.CLAIM_CONF]) - 1
            return out
        if phase == Phase.RESPONSE_ROUND:
            out[Head.RESPONSE_TYPE] = int(ResponseType.ACCUSE)
            out[Head.RESPONSE_TARGET] = first_legal(masks[Head.RESPONSE_TARGET], tgt)
            return out
        if phase == Phase.VOTE:
            out[Head.VOTE] = self._vote_frame(state, agent, masks)
        return out


class SubtleSupport(AlibiCoalition):
    """plan sec.15.5: the partner defends/agrees without repeating the claim.

    Distinguishes "coordination" from "redundant duplicate testimony" - the H4 contrast.
    """

    name = "coalition_subtle"

    def meeting_action(self, state, agent, masks, rng):
        phase = Phase(state.phase)
        if self.is_creator(state, agent):
            return AlibiCoalition.meeting_action(self, state, agent, masks, rng)
        out = self._blank(masks)
        if phase == Phase.CLAIM_ROUND:
            return self._truthful_claim(state, agent, masks)
        if phase == Phase.RESPONSE_ROUND:
            out[Head.RESPONSE_TYPE] = int(ResponseType.AGREE_WITH)
            out[Head.RESPONSE_TARGET] = first_legal(
                masks[Head.RESPONSE_TARGET], int(state.incident_creator)
            )
            return out
        if phase == Phase.VOTE:
            out[Head.VOTE] = self._vote_frame(state, agent, masks)
        return out


class Sacrifice(CoalitionBase):
    """plan sec.15.5: the partner turns on the creator when the evidence is overwhelming.

    Under the pre-registered outcome this *loses* (a coalition member is ejected), so it is a
    useful negative control: it shows the metric is not just rewarding "coalition agrees".
    """

    name = "coalition_sacrifice"

    def _creator_is_doomed(self, state: GameState, agent: int) -> bool:
        k = ObserverKnowledge.build(state, agent)
        creator = int(state.incident_creator)
        room = self.marker_room(state)
        if room < 0:
            return False
        return any(
            k.known[t, creator] != -1 and int(k.known[t, creator]) == room
            for t in scene_window(state)
            if t < state.config.n_times
        )

    def meeting_action(self, state, agent, masks, rng):
        phase = Phase(state.phase)
        out = self._blank(masks)
        creator = int(state.incident_creator)
        if self.is_creator(state, agent):
            return LoneLiar.meeting_action(self, state, agent, masks, rng)
        doomed = self._creator_is_doomed(state, agent)
        if phase == Phase.CLAIM_ROUND:
            return self._truthful_claim(state, agent, masks)
        if phase == Phase.RESPONSE_ROUND:
            out[Head.RESPONSE_TYPE] = int(ResponseType.ACCUSE if doomed else ResponseType.DEFEND)
            out[Head.RESPONSE_TARGET] = first_legal(masks[Head.RESPONSE_TARGET], creator)
            return out
        if phase == Phase.VOTE:
            out[Head.VOTE] = (
                first_legal(masks[Head.VOTE], creator)
                if doomed
                else self._vote_frame(state, agent, masks)
            )
        return out


SCRIPTED_COALITIONS = {
    "truthful": TruthfulCoalition,
    "lone_liar": LoneLiar,
    "alibi": AlibiCoalition,
    "framer": Framer,
    "subtle_support": SubtleSupport,
    "sacrifice": Sacrifice,
}


__all__ = [
    "CoalitionBase",
    "TruthfulCoalition",
    "LoneLiar",
    "AlibiCoalition",
    "Framer",
    "SubtleSupport",
    "Sacrifice",
    "SCRIPTED_COALITIONS",
]
