"""Enumerations for the game engine.

Convention that the whole codebase relies on: **index 0 of every categorical head is the
safe no-op** (WAIT / NO_INFORMATION / NEUTRAL / STAY-in-place is handled separately). Action
masks therefore always have at least one legal entry, and an agent that is not the active
speaker in a sequential round is masked down to index 0 on every head.
"""

from __future__ import annotations

from enum import IntEnum


class Phase(IntEnum):
    RESET = 0
    FREE_PLAY = 1
    REPORT = 2
    SYMBOL = 3
    CLAIM_ROUND = 4
    RESPONSE_ROUND = 5
    VOTE = 6
    TERMINAL = 7


class Role(IntEnum):
    CREW = 0
    COALITION = 1


class RoomAction(IntEnum):
    WAIT = 0
    PERFORM_TASK = 1
    OBSERVE = 2
    REPORT = 3
    CREATE_INCIDENT = 4


class ClaimType(IntEnum):
    NO_INFORMATION = 0
    SELF_LOCATION = 1
    SAW_AGENT_IN_ROOM = 2
    SAW_AGENT_ENTER = 3
    SAW_AGENT_LEAVE = 4
    WAS_WITH_AGENT = 5
    FOUND_MARKER = 6


#: claim types whose truth conditions involve another agent
SUBJECT_CLAIMS = frozenset(
    {
        ClaimType.SAW_AGENT_IN_ROOM,
        ClaimType.SAW_AGENT_ENTER,
        ClaimType.SAW_AGENT_LEAVE,
        ClaimType.WAS_WITH_AGENT,
    }
)
#: claim types that name a room
ROOM_CLAIMS = frozenset(
    {
        ClaimType.SELF_LOCATION,
        ClaimType.SAW_AGENT_IN_ROOM,
        ClaimType.SAW_AGENT_ENTER,
        ClaimType.SAW_AGENT_LEAVE,
        ClaimType.FOUND_MARKER,
    }
)
#: claim types that name a timestep
TIME_CLAIMS = frozenset(
    {
        ClaimType.SELF_LOCATION,
        ClaimType.SAW_AGENT_IN_ROOM,
        ClaimType.SAW_AGENT_ENTER,
        ClaimType.SAW_AGENT_LEAVE,
        ClaimType.WAS_WITH_AGENT,
        ClaimType.FOUND_MARKER,
    }
)
#: enter/leave claims compare t-1 with t, so t = 0 is meaningless for them
TRANSITION_CLAIMS = frozenset({ClaimType.SAW_AGENT_ENTER, ClaimType.SAW_AGENT_LEAVE})


class ResponseType(IntEnum):
    NEUTRAL = 0
    ACCUSE = 1
    DEFEND = 2
    AGREE_WITH = 3
    CONTRADICT = 4
    QUESTION = 5


TARGETED_RESPONSES = frozenset(
    {
        ResponseType.ACCUSE,
        ResponseType.DEFEND,
        ResponseType.AGREE_WITH,
        ResponseType.CONTRADICT,
        ResponseType.QUESTION,
    }
)


class TruthLabel(IntEnum):
    """Automatic label assigned to every claim against ground truth.

    TRUE           content matches ground truth *and* the speaker had an entitling observation
    FALSE          content contradicts ground truth
    UNSUPPORTED    content happens to be true but the speaker could not have observed it,
                   or the claim is about something unobservable to the speaker
    AMBIGUOUS      true under more than one reading (eliminated by construction in the MVP;
                   `claims.assert_no_ambiguity` fails loudly if it ever fires)
    CONTRADICTORY  conflicts with a *prior public claim by the same speaker*
    """

    TRUE = 0
    FALSE = 1
    UNSUPPORTED = 2
    AMBIGUOUS = 3
    CONTRADICTORY = 4


class Head(IntEnum):
    """Factored policy heads (plan sec.9). Every joint action is an (n_agents, N_HEADS) int array."""

    MOVE = 0
    ROOM_ACTION = 1
    INCIDENT_TARGET = 2
    SYMBOL = 3
    CLAIM_TYPE = 4
    CLAIM_SUBJECT = 5
    CLAIM_ROOM = 6
    CLAIM_TIME = 7
    CLAIM_POLARITY = 8
    CLAIM_CONF = 9
    RESPONSE_TYPE = 10
    RESPONSE_TARGET = 11
    VOTE = 12


N_HEADS = len(Head)

#: which heads carry gradient in which phase (all others are masked to a single legal index)
PHASE_HEADS: dict[Phase, tuple[Head, ...]] = {
    Phase.FREE_PLAY: (Head.MOVE, Head.ROOM_ACTION, Head.INCIDENT_TARGET),
    Phase.SYMBOL: (Head.SYMBOL,),
    Phase.CLAIM_ROUND: (
        Head.CLAIM_TYPE,
        Head.CLAIM_SUBJECT,
        Head.CLAIM_ROOM,
        Head.CLAIM_TIME,
        Head.CLAIM_POLARITY,
        Head.CLAIM_CONF,
    ),
    Phase.RESPONSE_ROUND: (Head.RESPONSE_TYPE, Head.RESPONSE_TARGET),
    Phase.VOTE: (Head.VOTE,),
    Phase.REPORT: (),
    Phase.RESET: (),
    Phase.TERMINAL: (),
}


class EventType:
    """String event tags (plan sec.12.1). Kept as plain strings so replays are human-readable."""

    EPISODE_STARTED = "EpisodeStarted"
    ROLES_ASSIGNED = "RolesAssigned"
    AGENT_MOVED = "AgentMoved"
    TASK_ATTEMPTED = "TaskAttempted"
    AGENT_OBSERVED = "AgentObserved"
    INCIDENT_CREATED = "IncidentCreated"
    MARKER_OBSERVED = "MarkerObserved"
    REPORT_CALLED = "ReportCalled"
    PRIVATE_SYMBOL_SENT = "PrivateSymbolSent"
    CLAIM_MADE = "ClaimMade"
    RESPONSE_MADE = "ResponseMade"
    VOTE_CAST = "VoteCast"
    AGENT_EJECTED = "AgentEjected"
    ROUND_STARTED = "RoundStarted"
    EPISODE_ENDED = "EpisodeEnded"


ALL_EVENT_TYPES = tuple(
    v for k, v in vars(EventType).items() if not k.startswith("_") and isinstance(v, str)
)
