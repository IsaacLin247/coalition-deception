"""Visualization stage V0: the text renderer (plan sec.22).

Two views, exactly as the visualization plan specifies:

* **public** - only what a crew observer could know: room occupancy at publicly known times,
  the report, the transcript, responses, votes;
* **privileged** - adds true roles, the incident tuple, claim truth labels, private symbols.

Renderers consume state/replays and never mutate them.
"""

from __future__ import annotations

from social_collusion.env.claims import to_text
from social_collusion.env.enums import Phase, ResponseType, Role, TruthLabel
from social_collusion.env.state import DEAD, GameState

LABEL_MARK = {
    TruthLabel.TRUE: "[TRUE]",
    TruthLabel.FALSE: "[FALSE]",
    TruthLabel.UNSUPPORTED: "[UNSUPPORTED]",
    TruthLabel.AMBIGUOUS: "[AMBIGUOUS]",
    TruthLabel.CONTRADICTORY: "[CONTRADICTORY]",
}


def agent_names(state: GameState) -> list[str]:
    return [f"A{i}" for i in range(state.config.n_agents)]


def render_turn(state: GameState, t: int, names: list[str] | None = None) -> str:
    cfg = state.config
    nm = names or agent_names(state)
    parts = []
    for r, room in enumerate(cfg.rooms):
        who = [nm[i] for i in range(cfg.n_agents) if int(state.positions[t, i]) == r]
        if who:
            parts.append(f"{room}: {' '.join(who)}")
    return f"T{t} | " + " | ".join(parts) if parts else f"T{t} | (empty)"


def render_state(state: GameState, privileged: bool = False) -> str:
    cfg = state.config
    nm = agent_names(state)
    out: list[str] = []
    out.append(f"=== {cfg.name} | seed={state.seed} | phase={Phase(state.phase).name} ===")

    if privileged:
        roles = ", ".join(f"{nm[i]}={Role(int(state.roles[i])).name}" for i in range(cfg.n_agents))
        out.append(f"[PRIVILEGED] roles: {roles}")
        if state.incident_creator >= 0:
            out.append(
                f"[PRIVILEGED] incident: {nm[state.incident_creator]} -> "
                f"{nm[state.incident_victim]} in {cfg.rooms[state.incident_room]} at "
                f"t{state.incident_time}"
            )
        out.append("--- movement ---")
        for t in range(state.horizon + 1):
            out.append("  " + render_turn(state, t, nm))
    else:
        out.append("--- public knowledge ---")
        missing = [nm[i] for i in range(cfg.n_agents) if not state.alive[i]]
        out.append(f"  inactive: {', '.join(missing) if missing else 'none'}")
        if state.report_turn >= 0:
            who = nm[state.reporter] if state.reporter >= 0 else "(auto)"
            room = cfg.rooms[state.incident_room] if state.incident_room >= 0 else "?"
            out.append(f"  marker reported by {who} in {room} at t{state.report_turn}")

    if privileged and state.symbols.size and (state.symbols >= 0).any():
        out.append("--- private channel ---")
        for i in range(cfg.n_agents):
            if state.symbols[i] >= 0:
                p = state.partner_of(i)
                tgt = nm[p] if p >= 0 else "(nobody)"
                out.append(f"  {nm[i]} sent private symbol {int(state.symbols[i])} to {tgt}.")

    if state.claims:
        out.append("--- meeting: claims ---")
        for c in state.claims:
            line = "  " + to_text(c, cfg, nm)
            if privileged and c.label is not None:
                line += " " + LABEL_MARK[TruthLabel(c.label)]
            out.append(line)

    if state.responses:
        out.append("--- meeting: responses ---")
        for r in state.responses:
            rt = ResponseType(r.response_type).name
            tgt = nm[r.target] if r.target is not None else ""
            out.append(f"  {nm[r.speaker]}: {rt}{' ' + tgt if tgt else ''}")

    if (state.votes >= 0).any():
        out.append("--- vote ---")
        for i in range(cfg.n_agents):
            v = int(state.votes[i])
            if v < 0:
                continue
            out.append(f"  {nm[i]} -> {'SKIP' if v == cfg.n_agents else nm[v]}")
        if state.ejected >= 0:
            suffix = ""
            if privileged:
                suffix = f" ({Role(int(state.roles[state.ejected])).name})"
            out.append(f"  EJECTED: {nm[state.ejected]}{suffix}")
        else:
            out.append("  EJECTED: nobody (tie / skip)")

    if state.done:
        from social_collusion.env.rewards import outcome_dict

        o = outcome_dict(state)
        out.append("--- outcome ---")
        out.append(
            f"  coalition_favorable={o['coalition_favorable']} "
            f"crew_ejected={o['crew_ejected']} creator_survived={o['creator_survived']}"
        )
    return "\n".join(out)


def render_events(state: GameState, privileged: bool = True) -> str:
    hidden = {"IncidentCreated", "RolesAssigned", "PrivateSymbolSent", "AgentObserved"}
    lines = []
    for e in state.events:
        if not privileged and e.type in hidden:
            continue
        lines.append(f"  t{e.turn:>2} {e.type}: {e.payload}")
    return "\n".join(lines)


def render_evidence_packets(state: GameState) -> str:
    """The Stage-2 'evidence packet' view: what each agent privately knows."""
    from social_collusion.env.observations import last_seen

    cfg = state.config
    nm = agent_names(state)
    out = ["--- private evidence packets ---"]
    for i in range(cfg.n_agents):
        rooms, times = last_seen(state, i)
        bits = []
        for j in range(cfg.n_agents):
            if j == i or times[j] < 0:
                continue
            bits.append(f"{nm[j]} in {cfg.rooms[int(rooms[j])]} at t{int(times[j])}")
        if state.marker_seen[i]:
            bits.append(
                f"MARKER in {cfg.rooms[state.incident_room]} at t{int(state.marker_seen_time[i])}"
            )
        pos = int(state.positions[min(state.turn, cfg.n_times - 1), i])
        where = "inactive" if pos == DEAD else cfg.rooms[pos]
        out.append(f"  {nm[i]} ({where}): " + ("; ".join(bits) if bits else "no evidence"))
    return "\n".join(out)


__all__ = ["render_state", "render_turn", "render_events", "render_evidence_packets"]
