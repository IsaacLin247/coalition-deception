#!/usr/bin/env python3
"""Render one complete multi-round replay for F4.

The ordinary episode renderer is meeting-oriented: the environment intentionally resets its
position matrix at the start of each new round.  This renderer instead re-simulates the saved
replay and collects the position at every free-play timestep, preserving the whole game.

Outputs:
  - a privileged static figure (roles visible for analysis),
  - an omniscient static figure with direct role labels suppressed (legacy `public` filename),
  - a self-contained HTML walkthrough with a play button and role toggle,
  - the reconstructed trace JSON used by both renderers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from _common import limit_threads

from social_collusion.env.enums import EventType, Phase, Role
from social_collusion.env.transition import reset as engine_reset
from social_collusion.env.transition import transition as engine_transition
from social_collusion.replay import config_of, read
from social_collusion.seeding import episode_rng


DEFAULT_REPLAY = "results/f4_tenseed_crew7_s0/replays/coalition1_vs_learned_crew1_episode0.json.gz"
DEFAULT_OUT = "replay"

ROOM_COLORS = ["#90caf9", "#ffcc80", "#b39ddb", "#80cbc4", "#f48fb1", "#c5e1a5"]
ROLE_COLORS = {"CREW": "#1565c0", "COALITION": "#c62828"}


def _names(n: int) -> list[str]:
    return [f"A{i}" for i in range(n)]


def _role_label(roles: dict[str, str], i: int, visible: bool = True) -> str:
    name = f"A{i}"
    if not visible:
        return name
    return f"{name} ({'I' if roles.get(name) == 'COALITION' else 'C'})"


def _event_round(raw: dict[str, Any], before_round: int) -> int:
    if raw["type"] == EventType.ROUND_STARTED:
        return int(raw.get("payload", {}).get("round", before_round + 1))
    return int(before_round)


def _event_label(raw: dict[str, Any], cfg, roles: dict[str, str]) -> str:
    p = raw.get("payload", {})
    names = _names(cfg.n_agents)
    typ = raw["type"]

    def agent(x: Any) -> str:
        return "nobody" if x is None else names[int(x)]

    if typ == EventType.ROUND_STARTED:
        return f"Round {int(p.get('round', 0)) + 1} begins"
    if typ == EventType.TASK_ATTEMPTED:
        ok = "completed" if p.get("success") else "failed"
        return f"{agent(p.get('agent'))} {ok} a task in {cfg.rooms[int(p['room'])]}"
    if typ == EventType.INCIDENT_CREATED:
        return (
            f"incident: {agent(p.get('creator'))} killed {agent(p.get('victim'))} "
            f"in {cfg.rooms[int(p['room'])]}"
        )
    if typ == EventType.REPORT_CALLED:
        return f"{agent(p.get('agent', p.get('reporter')))} called a report"
    if typ == EventType.CLAIM_MADE:
        return f"{agent(p.get('speaker'))} made a claim"
    if typ == EventType.RESPONSE_MADE:
        return f"{agent(p.get('speaker'))} responded"
    if typ == EventType.VOTE_CAST:
        vote = p.get("vote")
        return f"{agent(p.get('agent'))} voted for {'SKIP' if vote is None else agent(vote)}"
    if typ == EventType.AGENT_EJECTED:
        target = p.get("agent")
        return f"ejected {agent(target)}" if target is not None else "vote skipped / tied"
    if typ == EventType.EPISODE_ENDED:
        return "game ended"
    if typ == EventType.MARKER_OBSERVED:
        return f"{agent(p.get('agent'))} found the marker"
    return typ


def build_trace(path: Path) -> dict[str, Any]:
    """Re-simulate a replay and return a JSON-safe, whole-game trace."""
    record = read(path)
    cfg = config_of(record)
    rng = episode_rng(record.seed, record.episode_index)
    state = engine_reset(cfg, rng, seed=record.seed)
    roles = {str(k): str(v) for k, v in record.roles.items()}

    frames: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    coherent_round: int | None = None
    coherent_positions: list[int] | None = None
    coherent_alive: list[bool] | None = None
    last_free_play_frame: int | None = None

    for step_index, step in enumerate(record.steps):
        before = state
        before_phase = Phase(before.phase).name

        # This is the state at which the action is taken.  It preserves every round because the
        # next round's fresh position array is captured when the loop reaches its first FREE_PLAY
        # step.
        frame_record: dict[str, Any] | None = None
        if before_phase == Phase.FREE_PLAY.name:
            t = min(int(before.turn), int(before.horizon))
            round_number = int(before.round)
            if coherent_positions is None or coherent_round != round_number:
                # A new round legitimately resets the spatial state.  Within a round, however,
                # use the previous frame's event-resolved positions; some replay state arrays
                # retain the pre-generated trajectory rather than the mutable display position.
                coherent_positions = [int(x) for x in before.positions[t]]
                coherent_alive = [bool(x) for x in before.alive]
                coherent_round = round_number
            pre_positions = list(coherent_positions)
            pre_alive = list(coherent_alive or before.alive)
            frame_record = {
                "round": round_number,
                "turn": int(before.turn),
                "positions": pre_positions,
                "alive": pre_alive,
                "positions_after": list(pre_positions),
                "alive_after": list(pre_alive),
                "round_start": False,
                "phases": {
                    "turn_start": [],
                    "tasks": [],
                    "incident": [],
                    "report": [],
                    "movement": [],
                    "observation": [],
                    "meeting": [],
                },
                "events": [],
            }
            frames.append(frame_record)
            last_free_play_frame = len(frames) - 1

        action = np.asarray(step.joint_actions, dtype=np.int64)
        result = engine_transition(before, action, rng, inplace=True, validate=True)
        after = result.state
        if after.state_hash() != step.state_hash:
            raise RuntimeError(
                f"replay hash mismatch at step {step_index}: "
                f"expected {step.state_hash}, got {after.state_hash()}"
            )

        if frame_record is not None:
            # Keep the post-action state for a short visual action-resolution beat.
            post_positions = list(frame_record["positions"])
            for raw in step.events:
                if raw["type"] == EventType.AGENT_MOVED:
                    payload = raw.get("payload", {})
                    agent = payload.get("agent")
                    target = payload.get("to")
                    source = payload.get("frm")
                    if agent is not None and target is not None:
                        expected = int(frame_record["positions"][int(agent)])
                        if source is not None and int(source) != expected:
                            raise RuntimeError(
                                f"movement trace mismatch at step {step_index}, agent {agent}: "
                                f"event says {source}, display state says {expected}"
                            )
                        post_positions[int(agent)] = int(target)
                if raw["type"] == EventType.ROUND_STARTED:
                    frame_record["round_start"] = True
            frame_record["positions_after"] = post_positions
            frame_record["alive_after"] = [bool(x) for x in after.alive]
            coherent_positions = list(post_positions)
            coherent_alive = [bool(x) for x in after.alive]
            for agent, alive in enumerate(coherent_alive):
                if not alive:
                    coherent_positions[agent] = -1

        for raw in step.events:
            row = {
                "step": int(step_index),
                "round": _event_round(raw, int(before.round)),
                "turn": int(raw.get("turn", step.turn)),
                "phase": str(raw.get("phase", step.phase)),
                "type": str(raw["type"]),
                "payload": raw.get("payload", {}),
                "label": _event_label(raw, cfg, roles),
                "source_frame": (
                    len(frames) - 1 if frame_record is not None else last_free_play_frame
                ),
            }
            event_rows.append(row)

        state = after

    if not frames:
        raise RuntimeError("replay contained no FREE_PLAY frames")
    # The initial state is the beginning of round 1 even when the replay format omits an
    # explicit ROUND_STARTED event for the first round.
    frames[0]["round_start"] = True

    # Attach events to the free-play transition that produced them.  Meeting events happen in
    # later phase steps, so they attach to the most recent free-play frame; matching only on
    # (round, turn) can mix the last turn of one round with the first meeting step of the next.
    by_round: dict[int, list[int]] = defaultdict(list)
    for idx, frame in enumerate(frames):
        by_round[int(frame["round"])].append(idx)

    for row in event_rows:
        r = int(row["round"])
        if row["type"] == EventType.ROUND_STARTED:
            # RoundStarted can occur before the first FREE_PLAY step of the new round.
            later = [i for i, f in enumerate(frames) if int(f["round"]) >= r]
            target = later[0] if later else len(frames) - 1
        else:
            target = row.get("source_frame")
            if target is None:
                target = len(frames) - 1
        row.pop("source_frame", None)
        row["frame"] = int(target)
        # The transition state is authoritative for the visual round label.  Vote steps can
        # already carry the incremented internal round number even though they belong to the
        # meeting that follows the preceding free-play frame.
        row["round"] = int(frames[target]["round"])
        frames[target]["events"].append(row)
        if row["type"] == EventType.ROUND_STARTED:
            frames[target]["round_start"] = True
            frames[target]["phases"]["turn_start"].append(row)
        elif row["type"] == EventType.TASK_ATTEMPTED:
            frames[target]["phases"]["tasks"].append(row)
        elif row["type"] == EventType.INCIDENT_CREATED:
            frames[target]["phases"]["incident"].append(row)
        elif row["type"] == EventType.REPORT_CALLED:
            frames[target]["phases"]["report"].append(row)
        elif row["type"] == EventType.AGENT_MOVED:
            frames[target]["phases"]["movement"].append(row)
        elif row["type"] in (EventType.AGENT_OBSERVED, EventType.MARKER_OBSERVED):
            frames[target]["phases"]["observation"].append(row)
        elif row["type"] in (
            EventType.CLAIM_MADE,
            EventType.RESPONSE_MADE,
            EventType.VOTE_CAST,
            EventType.AGENT_EJECTED,
        ):
            frames[target]["phases"]["meeting"].append(row)

    # Meeting summaries and post-vote population counts.
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[int(frames[int(row["frame"])] ["round"])].append(row)
    alive = [bool(x) for x in record.initial_state.get("alive", [])]
    round_summaries: list[dict[str, Any]] = []
    for r in sorted(by_round):
        rows = grouped[r]
        incident = next((x for x in rows if x["type"] == EventType.INCIDENT_CREATED), None)
        report = next((x for x in rows if x["type"] == EventType.REPORT_CALLED), None)
        ejections = [x for x in rows if x["type"] == EventType.AGENT_EJECTED]
        eject = ejections[-1] if ejections else None
        if eject and eject["payload"].get("agent") is not None:
            alive[int(eject["payload"]["agent"])] = False

        tasks = sum(
            1
            for x in rows
            if x["type"] == EventType.TASK_ATTEMPTED and x["payload"].get("success")
        )
        moves = sum(1 for x in rows if x["type"] == EventType.AGENT_MOVED)
        if incident:
            ip = incident["payload"]
            incident_text = (
                f"{_names(cfg.n_agents)[int(ip['creator'])]} → "
                f"{_names(cfg.n_agents)[int(ip['victim'])]} in {cfg.rooms[int(ip['room'])]}"
            )
        else:
            incident_text = "none"
        report_text = "yes" if report else "no"
        eject_target = None if not eject else eject["payload"].get("agent")
        eject_text = "skip / tie" if eject_target is None else _names(cfg.n_agents)[int(eject_target)]
        round_summaries.append(
            {
                "round": int(r) + 1,
                "tasks": int(tasks),
                "moves": int(moves),
                "incident": incident_text,
                "report": report_text,
                "ejected": eject_text,
                "crew_alive": int(sum(alive[i] and roles.get(f"A{i}") == "CREW" for i in range(cfg.n_agents))),
                "coalition_alive": int(sum(alive[i] and roles.get(f"A{i}") == "COALITION" for i in range(cfg.n_agents))),
            }
        )

    task_matrix = np.asarray(record.initial_state.get("tasks", []), dtype=np.int8)
    if task_matrix.ndim == 2 and task_matrix.shape[1] == len(cfg.rooms):
        all_assigned = task_matrix.sum(axis=0).astype(int).tolist()
        crew_rows = [
            i
            for i in range(min(cfg.n_agents, task_matrix.shape[0]))
            if roles.get(f"A{i}") == "CREW"
        ]
        crew_assigned = task_matrix[crew_rows].sum(axis=0).astype(int).tolist()
    else:
        all_assigned = [0] * len(cfg.rooms)
        crew_assigned = [0] * len(cfg.rooms)

    return {
        "config": {
            "name": cfg.name,
            "rooms": list(cfg.rooms),
            "edges": [list(e) for e in cfg.edges],
            "n_agents": int(cfg.n_agents),
        },
        "seed": int(record.seed),
        "roles": roles,
        "tasks": {
            "crew_assigned_by_room": crew_assigned,
            "all_assigned_by_room": all_assigned,
        },
        "outcome": record.outcome,
        "frames": frames,
        "events": event_rows,
        "meetings": round_summaries,
        "replay": str(path),
    }


def _frame_event_x(trace: dict[str, Any], typ: str) -> list[tuple[int, dict[str, Any]]]:
    return [(int(e["frame"]), e) for e in trace["events"] if e["type"] == typ]


def plot_static(trace: dict[str, Any], path: Path, privileged: bool) -> None:
    cfg = trace["config"]
    rooms = cfg["rooms"]
    frames = trace["frames"]
    n = int(cfg["n_agents"])
    arr = np.asarray([f["positions"] for f in frames], dtype=int).T
    dead = len(ROOM_COLORS)
    values = np.where(arr < 0, dead, arr)
    colors = ROOM_COLORS[: max(len(rooms), 1)] + ["#37474f"]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), len(colors))

    fig = plt.figure(figsize=(15, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], width_ratios=[1.7, 1.0], hspace=0.34, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title(
        "F4 omniscient diagnostic replay: learned/adapted coalition versus learned crew\n"
        f"{len(trace['meetings'])} meetings, {n} agents; room trajectory reconstructed from the event-sourced replay",
        fontsize=13,
        weight="bold",
    )
    labels = [_role_label(trace["roles"], i, privileged) for i in range(n)]
    ax.set_yticks(range(n), labels)
    if privileged:
        for tick, i in zip(ax.get_yticklabels(), range(n)):
            tick.set_color(ROLE_COLORS[trace["roles"].get(f"A{i}", "CREW")])
    xlabels = [f"R{int(f['round']) + 1}:t{int(f['turn'])}" for f in frames]
    stride = 1 if len(frames) <= 42 else 2
    ax.set_xticks(range(0, len(frames), stride), xlabels[::stride], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("free-play timeline (each cell is one recorded ground-truth position)")
    ax.set_ylabel("player")
    ax.grid(axis="x", color="white", alpha=0.32, linewidth=0.55)

    round_starts = [i for i in range(1, len(frames)) if frames[i]["round"] != frames[i - 1]["round"]]
    for x in round_starts:
        ax.axvline(x - 0.5, color="#263238", lw=1.8)

    names = _names(n)
    for x, event in _frame_event_x(trace, EventType.INCIDENT_CREATED):
        creator = event["payload"].get("creator")
        if creator is not None:
            ax.scatter(x, int(creator), marker="*", s=100, facecolor="#ff7043", edgecolor="white", zorder=5)
    for x, event in _frame_event_x(trace, EventType.AGENT_EJECTED):
        target = event["payload"].get("agent")
        if target is not None:
            ax.scatter(x, int(target), marker="X", s=70, color="#212121", zorder=5)
    for x, _ in _frame_event_x(trace, EventType.REPORT_CALLED):
        ax.axvline(x, color="#00897b", ls=":", lw=1.1, alpha=0.9)

    legend = [Patch(facecolor=ROOM_COLORS[i], edgecolor="none", label=room) for i, room in enumerate(rooms)]
    legend += [Patch(facecolor="#37474f", edgecolor="none", label="dead")]
    ax.legend(handles=legend, ncol=min(6, len(legend)), loc="upper left", bbox_to_anchor=(0, 1.18), fontsize=8, frameon=False)

    ax2 = fig.add_subplot(gs[1, 0])
    rounds = [0] + [m["round"] for m in trace["meetings"]]
    crew = [n - len([i for i in range(n) if trace["roles"].get(f"A{i}") == "COALITION"])] + [m["crew_alive"] for m in trace["meetings"]]
    coal = [len([i for i in range(n) if trace["roles"].get(f"A{i}") == "COALITION"])] + [m["coalition_alive"] for m in trace["meetings"]]
    ax2.step(rounds, crew, where="post", color=ROLE_COLORS["CREW"], lw=2.4, label="crew alive")
    ax2.step(rounds, coal, where="post", color=ROLE_COLORS["COALITION"], lw=2.4, label="impostors alive")
    ax2.set_xticks(rounds)
    ax2.set_xlabel("meeting completed")
    ax2.set_ylabel("surviving players")
    ax2.set_title("Population across the game", weight="bold")
    ax2.set_ylim(-0.2, n + 0.5)
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    cell_text = []
    for m in trace["meetings"]:
        cell_text.append([m["round"], m["moves"], m["tasks"], m["incident"], m["ejected"]])
    table = ax3.table(
        cellText=cell_text,
        colLabels=["R", "moves", "tasks", "incident", "ejected"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.07, 0.13, 0.13, 0.43, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cfd8dc")
        if row == 0:
            cell.set_facecolor("#eceff1")
            cell.set_text_props(weight="bold")
    ax3.set_title("What happened in each meeting", weight="bold", pad=10)
    out = trace["outcome"]
    outcome_text = "coalition-favourable" if out.get("coalition_favorable") else "crew-favourable"
    role_display = "direct role labels shown" if privileged else "direct role labels suppressed"
    fig.text(0.01, 0.01, f"Omniscient diagnostic; {role_display}. Outcome: {outcome_text}; ★ incident, X ejection, teal line report", fontsize=8, color="#546e7a")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>F4 full-game replay</title>
<style>
:root{--bg:#fafafa;--fg:#17202a;--muted:#607d8b;--line:#d9e1e6;--crew:#1565c0;--coal:#c62828;--accent:#00897b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}h1{font-size:18px;margin:0}.muted{color:var(--muted);font-size:12px}.grow{flex:1}
button{font:inherit;padding:6px 11px;border:1px solid var(--line);border-radius:7px;background:white;color:var(--fg);cursor:pointer}button:hover{border-color:var(--accent)}
.grid{display:grid;grid-template-columns:minmax(420px,1.35fr) minmax(310px,1fr);gap:16px;padding:16px 20px}.panel{border:1px solid var(--line);border-radius:10px;padding:14px;background:white}.panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 10px}
svg{width:100%;height:auto;display:block}.edge{stroke:#cfd8dc;stroke-width:2}.room{fill:#f5f7f8;stroke:#b0bec5;stroke-width:2}.tok{font-size:12px;font-weight:700}.crew{fill:var(--crew)}.coal{fill:var(--coal)}.event{padding:5px 0;border-bottom:1px solid var(--line);font-size:13px}.event:last-child{border:0}.tag{display:inline-block;border:1px solid currentColor;border-radius:999px;padding:0 5px;font-size:10px;margin-right:5px;color:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:5px 4px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}input[type=range]{width:100%;accent-color:var(--accent)}
@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>F4 full-game replay</h1><span class="muted" id="meta"></span><span class="grow"></span><button id="roles">Reveal roles</button><button id="play">Play</button></header>
<div class="grid"><div><div class="panel"><h2>Map at current timestep</h2><svg id="map" viewBox="0 0 500 350"></svg><input id="slider" type="range" min="0" max="0" value="0"><div class="muted" id="time"></div></div><div class="panel" style="margin-top:16px"><h2>Meeting summary</h2><table id="meetings"></table></div></div>
<div><div class="panel"><h2>Events at this timestep</h2><div id="events"></div></div><div class="panel" style="margin-top:16px"><h2>Outcome</h2><div id="outcome"></div></div></div></div>
<script id="data" type="application/json">__DATA__</script><script>
(function(){"use strict";const D=JSON.parse(document.getElementById("data").textContent);let i=0,showRoles=false,timer=null;
const colors=["#90caf9","#ffcc80","#b39ddb","#80cbc4","#f48fb1","#c5e1a5"];const n=D.config.n_agents;
const pos=[[250,170]];for(let r=1;r<D.config.rooms.length;r++){let a=2*Math.PI*(r-1)/Math.max(1,D.config.rooms.length-1)-Math.PI/2;pos.push([250+105*Math.cos(a),170+105*Math.sin(a)]);}
function E(tag,a,txt){let x=document.createElementNS("http://www.w3.org/2000/svg",tag);for(let k in a)x.setAttribute(k,a[k]);if(txt!==undefined)x.textContent=txt;return x}
function role(i){return D.roles["A"+i]||"CREW"}function name(i){return i==null?"nobody":"A"+i}
function draw(){let f=D.frames[i],s=document.getElementById("map");s.textContent="";D.config.edges.forEach(e=>{let a=D.config.rooms.indexOf(e[0]),b=D.config.rooms.indexOf(e[1]);s.appendChild(E("line",{x1:pos[a][0],y1:pos[a][1],x2:pos[b][0],y2:pos[b][1],class:"edge"}))});D.config.rooms.forEach((r,k)=>{s.appendChild(E("circle",{cx:pos[k][0],cy:pos[k][1],r:44,class:"room",fill:colors[k%colors.length]}));s.appendChild(E("text",{x:pos[k][0],y:pos[k][1]+62,"text-anchor":"middle",fill:"#607d8b"},r))});let here={};f.positions.forEach((room,a)=>{if(room<0)return;let q=here[room]||0;here[room]=q+1;let ang=2*Math.PI*q/Math.max(1,f.positions.filter(x=>x===room).length),rad=22;let cl=showRoles?(role(a)==="COALITION"?"tok coal":"tok crew"):"tok";s.appendChild(E("text",{x:pos[room][0]+rad*Math.cos(ang),y:pos[room][1]+rad*Math.sin(ang)+4,"text-anchor":"middle",class:cl},name(a))) });document.getElementById("time").textContent="Round "+(f.round+1)+" · free-play t"+f.turn+" · frame "+(i+1)+"/"+D.frames.length;document.getElementById("slider").value=i;
let ev=f.events||[];document.getElementById("events").innerHTML=ev.length?ev.map(x=>"<div class=event><span class=tag>R"+(x.round+1)+" · "+x.phase+"</span>"+x.label+"</div>").join(""):"<div class=muted>No event at this exact timestep.</div>";
document.getElementById("meta").textContent=D.config.name+" · "+n+" agents · "+D.meetings.length+" meetings";}
function drawTables(){let h="<tr><th>R</th><th>moves</th><th>tasks</th><th>incident</th><th>ejected</th></tr>";D.meetings.forEach(m=>h+="<tr><td>"+m.round+"</td><td>"+m.moves+"</td><td>"+m.tasks+"</td><td>"+m.incident+"</td><td>"+m.ejected+"</td></tr>");document.getElementById("meetings").innerHTML=h;let o=D.outcome;document.getElementById("outcome").innerHTML="<p><b>Coalition-favourable:</b> "+(o.coalition_favorable?"yes":"no")+"</p><p><b>Ejected:</b> "+(o.ejected<0?"nobody":name(o.ejected))+"</p><p class=muted>This omniscient diagnostic shows complete trajectories and incident identities. The toggle changes direct role labels. It replays recorded actions, not a live simulation.</p>"}
document.getElementById("slider").max=D.frames.length-1;document.getElementById("slider").addEventListener("input",e=>{i=+e.target.value;draw()});document.getElementById("roles").addEventListener("click",e=>{showRoles=!showRoles;e.textContent=showRoles?"Hide roles":"Reveal roles";draw()});document.getElementById("play").addEventListener("click",e=>{if(timer){clearInterval(timer);timer=null;e.target.textContent="Play";return}e.target.textContent="Pause";timer=setInterval(()=>{i=i+1;if(i>=D.frames.length){i=0}draw()},700)});drawTables();draw();})();
</script></body></html>"""


# Presentation-oriented replay UI.  The original templates above are retained as compact
# diagnostic fallbacks; this version is the default artifact used in the talk figures folder.
HTML_TEMPLATE_V2 = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Among Us RL · F4 replay</title>
<style>
:root{--bg:#070b18;--panel:#0d1427;--line:#263657;--text:#e8f0ff;--muted:#8b9ab8;--cyan:#32e6ff;--blue:#4d8dff;--red:#ff5b68;--gold:#ffc857;--green:#51e0a4}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% -10%,#16264a 0,#070b18 42%);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif;min-height:100vh}
button{font:inherit;color:inherit;border:1px solid var(--line);background:#111a30;border-radius:8px;padding:8px 13px;cursor:pointer;transition:.16s ease}button:hover{border-color:var(--cyan);box-shadow:0 0 16px #32e6ff22;transform:translateY(-1px)}
.app{max-width:1450px;margin:0 auto;padding:18px 22px 30px}.topbar{display:flex;align-items:center;gap:14px;padding:4px 0 15px;border-bottom:1px solid var(--line)}.brand{color:var(--cyan);font-weight:800;letter-spacing:.16em;font-size:12px}.top-title{font-size:20px;font-weight:800;letter-spacing:.04em}.grow{flex:1}.status-pill{border:1px solid #51e0a455;color:var(--green);padding:4px 9px;border-radius:999px;font-size:11px;letter-spacing:.08em;text-transform:uppercase}.hero{display:flex;justify-content:space-between;gap:24px;padding:22px 0 18px}.eyebrow{color:var(--gold);font-size:11px;letter-spacing:.16em;text-transform:uppercase}.hero h1{font-size:30px;line-height:1.05;margin:5px 0 7px}.hero p{margin:0;color:var(--muted);max-width:720px}.controls{display:flex;align-items:center;gap:8px;align-self:flex-end}.primary{background:var(--cyan);color:#06101d;border-color:var(--cyan);font-weight:800}.primary:hover{background:#9cf7ff;color:#06101d}
.layout{display:grid;grid-template-columns:minmax(600px,1.55fr) minmax(320px,.8fr);gap:16px}.panel{background:linear-gradient(145deg,#101a31,#0b1224);border:1px solid var(--line);border-radius:14px;box-shadow:0 12px 30px #0005}.panel-head{display:flex;justify-content:space-between;align-items:center;padding:14px 16px 10px;border-bottom:1px solid var(--line)}.panel-title{font-size:11px;text-transform:uppercase;letter-spacing:.16em;color:var(--muted);font-weight:800}.panel-badge{color:var(--cyan);font-size:11px}.board{padding:9px 12px 0}.map-wrap{position:relative;overflow:hidden;border-radius:10px;background:radial-gradient(circle at 50% 43%,#19325d 0,#0b1224 66%);border:1px solid #263c66}.map-wrap:before{content:"";position:absolute;inset:0;background-image:linear-gradient(#5a7db311 1px,transparent 1px),linear-gradient(90deg,#5a7db311 1px,transparent 1px);background-size:32px 32px;mask-image:linear-gradient(to bottom,black,transparent 90%);pointer-events:none}.map-wrap svg{position:relative;width:100%;height:auto;min-height:390px;display:block}.room-box{fill:#111d36;stroke:#3c5a8b;stroke-width:2}.room-box.hub{fill:#162b4f;stroke:var(--cyan)}.room-label{fill:#edf5ff;font-size:16px;font-weight:800;letter-spacing:.08em}.room-sub{fill:#8092b4;font-size:9px;letter-spacing:.18em;text-transform:uppercase}.path{stroke:#47638e;stroke-width:4;stroke-dasharray:7 9;opacity:.8}.path-glow{stroke:#32e6ff55;stroke-width:12;filter:url(#glow)}.token-ring{fill:#0b1224;stroke:#5e76a4;stroke-width:2}.token-ring.crew{stroke:var(--blue)}.token-ring.coal{stroke:var(--red)}.token-core{fill:#4d8dff}.token-core.coal{fill:#ff5b68}.token-label{fill:#f7fbff;font-size:11px;font-weight:900;text-anchor:middle;dominant-baseline:central}.event-star{fill:var(--gold);stroke:#fff2b2;stroke-width:2}.event-report{fill:none;stroke:var(--green);stroke-width:4}.tick{flex:1;min-width:7px;height:10px;padding:0;border:0;border-radius:99px;background:#273754;box-shadow:none;transform:none}.tick:hover{background:#8fd8ff;transform:none}.tick.active{height:16px;background:var(--cyan);box-shadow:0 0 14px #32e6ff99}.tick.round-change{background:#ffc857}.tick.active.round-change{background:#fff0ae}
.timeline{padding:13px 16px 3px}.time-row{display:flex;justify-content:space-between;gap:12px;color:var(--muted);font-size:12px}.time-row strong{color:var(--text)}.timeline-row{height:34px;display:flex;gap:4px;align-items:center}.round-strip{display:flex;gap:8px;padding:10px 16px 16px;overflow:auto}.round-card{min-width:118px;padding:9px 11px;border:1px solid var(--line);border-radius:9px;background:#0a1122;cursor:pointer}.round-card:hover,.round-card.active{border-color:var(--cyan);background:#102342}.round-card .rname{font-size:11px;color:var(--cyan);font-weight:800}.round-card .rmeta{font-size:11px;color:var(--muted);margin-top:2px}
.sidebar{display:flex;flex-direction:column;gap:14px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.stat{padding:11px 10px;border:1px solid var(--line);border-radius:11px;background:#0d1529}.stat-label{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}.stat-value{font-size:21px;font-weight:900;margin-top:2px}.stat-value.cyan{color:var(--cyan)}.stat-value.gold{color:var(--gold)}.stat-value.green{color:var(--green)}.side-body{padding:14px 16px}.side-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}.side-title{font-size:11px;text-transform:uppercase;letter-spacing:.15em;color:var(--muted);font-weight:800}.event-list{display:flex;flex-direction:column;gap:7px;max-height:205px;overflow:auto}.event-card{padding:9px 10px;border-left:3px solid var(--cyan);border-radius:6px;background:#101c33}.event-card.incident{border-color:var(--gold)}.event-card.eject{border-color:var(--red)}.event-card.report{border-color:var(--green)}.event-tag{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px}.event-text{font-size:13px}.empty{color:var(--muted);font-size:12px;padding:8px 0}.meeting-list{display:flex;flex-direction:column;gap:7px;max-height:235px;overflow:auto}.meeting{display:grid;grid-template-columns:40px 1fr auto;gap:9px;align-items:center;padding:8px 9px;border:1px solid #1d2b49;border-radius:8px;background:#0b1325;text-align:left}.meeting-num{color:var(--cyan);font-weight:900}.meeting-main{font-size:12px}.meeting-main b{color:#f2f6ff}.meeting-status{font-size:10px;color:var(--muted);text-align:right}.outcome-card{padding:14px 16px;border-radius:12px;border:1px solid #51e0a455;background:linear-gradient(135deg,#102b2e,#0d1729)}.outcome-card.bad{border-color:#ff5b6855;background:linear-gradient(135deg,#321a2a,#0d1729)}.outcome-title{font-size:14px;font-weight:900;color:var(--green)}.outcome-card.bad .outcome-title{color:var(--red)}.outcome-copy{color:var(--muted);font-size:12px;margin-top:4px}.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:11px;padding:0 16px 13px}.legend span:before{content:"";display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;background:var(--blue)}.legend .lcoal:before{background:var(--red)}.legend .levent:before{background:var(--gold)}.footer{color:#62769b;text-align:center;font-size:11px;padding-top:18px}
@media(max-width:1050px){.layout{grid-template-columns:1fr}.sidebar{display:grid;grid-template-columns:1fr 1fr}.stats,.outcome-card{grid-column:1/-1}}@media(max-width:700px){.app{padding:12px}.hero{display:block}.controls{margin-top:15px}.map-wrap svg{min-height:300px}.top-title{font-size:16px}}
</style></head><body>
<div class="app"><header class="topbar"><div class="brand">AMONG US // RL LAB</div><div class="top-title">OMNISCIENT REPLAY</div><span class="status-pill">event-sourced</span><div class="grow"></div><button id="roles">Show role labels</button><button class="primary" id="play">▶ Play replay</button></header>
<section class="hero"><div><div class="eyebrow">F4 · multi-round interaction</div><h1>Signal / Shadow</h1><p id="meta"></p></div><div class="controls"><button id="prev">←</button><button id="next">→</button></div></section>
<main class="layout"><section class="panel"><div class="panel-head"><span class="panel-title">Live map state</span><span class="panel-badge" id="phase"></span></div><div class="board"><div class="map-wrap"><svg id="map" viewBox="0 0 720 470" aria-label="Animated game map"></svg></div></div><div class="timeline"><div class="time-row"><span id="time"><strong>Round 1</strong></span><span id="frame-count"></span></div><div class="timeline-row" id="timeline"></div></div><div class="round-strip" id="rounds"></div><div class="legend"><span>crew / public token</span><span class="lcoal">impostor when roles revealed</span><span class="levent">incident marker</span></div></section>
<aside class="sidebar"><div class="stats"><div class="stat"><div class="stat-label">alive</div><div class="stat-value cyan" id="alive">—</div></div><div class="stat"><div class="stat-label">round</div><div class="stat-value gold" id="round-stat">—</div></div><div class="stat"><div class="stat-label">meeting</div><div class="stat-value green" id="meeting-stat">—</div></div></div><section class="panel"><div class="side-body"><div class="side-head"><span class="side-title">Current events</span><span class="panel-badge" id="event-count"></span></div><div class="event-list" id="events"></div></div></section><section class="panel"><div class="side-body"><div class="side-head"><span class="side-title">Round log</span><span class="panel-badge">click a round</span></div><div class="meeting-list" id="meetings"></div></div></section><section class="outcome-card" id="outcome"></section></aside></main><div class="footer">Recorded actions replayed from a validated event trace · not a live simulation · role information stays hidden until revealed</div></div>
<script id="data" type="application/json">__DATA__</script><script>
(function(){"use strict";const D=JSON.parse(document.getElementById("data").textContent);let i=0,showRoles=false,timer=null;const n=D.config.n_agents,roomNames=D.config.rooms,playerColors=["#39d5ff","#a98cff","#ffcc66","#ff7f91","#51e0a4","#f48fb1","#b8df72","#29b6f6","#ce93d8"];const roomPos=[];const hub=Math.max(0,roomNames.indexOf("Hub"));roomPos[hub]=[360,235];roomNames.map((_,k)=>k).filter(k=>k!==hub).forEach((k,j)=>{let a=2*Math.PI*j/Math.max(1,roomNames.length-1)-Math.PI/2;roomPos[k]=[360+215*Math.cos(a),235+150*Math.sin(a)]});
function E(tag,a,txt){let x=document.createElementNS("http://www.w3.org/2000/svg",tag);for(let k in a)x.setAttribute(k,a[k]);if(txt!==undefined)x.textContent=txt;return x}function role(a){return D.roles["A"+a]||"CREW"}function name(a){return a==null?"nobody":"A"+a}
function defs(s){let d=E("defs",{}),f=E("filter",{id:"glow",x:"-50%",y:"-50%",width:"200%",height:"200%"}),blur=E("feGaussianBlur",{stdDeviation:"4",result:"blur"}),merge=E("feMerge",{});merge.appendChild(E("feMergeNode",{in:"blur"}));merge.appendChild(E("feMergeNode",{in:"SourceGraphic"}));f.appendChild(blur);f.appendChild(merge);d.appendChild(f);s.appendChild(d)}
function drawMap(){let f=D.frames[i],s=document.getElementById("map");s.textContent="";defs(s);D.config.edges.forEach(edge=>{let a=roomNames.indexOf(edge[0]),b=roomNames.indexOf(edge[1]);s.appendChild(E("line",{x1:roomPos[a][0],y1:roomPos[a][1],x2:roomPos[b][0],y2:roomPos[b][1],class:"path-glow",filter:"url(#glow)"}));s.appendChild(E("line",{x1:roomPos[a][0],y1:roomPos[a][1],x2:roomPos[b][0],y2:roomPos[b][1],class:"path"}))});roomNames.forEach((room,k)=>{let x=roomPos[k][0],y=roomPos[k][1],g=E("g",{});g.appendChild(E("rect",{x:x-84,y:y-43,width:168,height:86,rx:18,class:"room-box "+(k===hub?"hub":"")}));g.appendChild(E("text",{x,y:y-1,"text-anchor":"middle",class:"room-label"},room));g.appendChild(E("text",{x,y:y+18,"text-anchor":"middle",class:"room-sub"},k===hub?"central hub":"room node"));s.appendChild(g)});let at={};f.positions.forEach((room,a)=>{if(room<0)return;(at[room]??=[]).push(a)});Object.keys(at).forEach(room=>{let k=+room,arr=at[k],x=roomPos[k][0],y=roomPos[k][1];arr.forEach((a,j)=>{let ang=2*Math.PI*j/Math.max(1,arr.length),rad=58,tx=x+rad*Math.cos(ang),ty=y+rad*Math.sin(ang),coal=role(a)==="COALITION",g=E("g",{transform:"translate("+tx+","+ty+")"});g.appendChild(E("circle",{r:25,class:"token-ring "+(showRoles?(coal?"coal":"crew"):"")}));let core={r:18,class:"token-core "+(showRoles&&coal?"coal":"")};if(!showRoles)core.fill=playerColors[a%playerColors.length];g.appendChild(E("circle",core));g.appendChild(E("text",{class:"token-label"},name(a)));g.appendChild(E("title",{},showRoles?(name(a)+" · "+(coal?"IMPOSTOR":"CREW")):name(a)));s.appendChild(g)})});(f.events||[]).forEach(ev=>{let p=ev.payload||{},room=p.room;if(ev.type==="IncidentCreated"&&room!=null&&roomPos[room])s.appendChild(E("text",{x:roomPos[room][0],y:roomPos[room][1]-62,"text-anchor":"middle",class:"event-star"},"★"));if(ev.type==="ReportCalled")s.appendChild(E("circle",{cx:roomPos[hub][0],cy:roomPos[hub][1],r:70,class:"event-report"}))})}
function eventClass(type){return type==="IncidentCreated"?"incident":type==="AgentEjected"?"eject":type==="ReportCalled"?"report":""}function drawEvents(){let f=D.frames[i],box=document.getElementById("events");box.textContent="";let ev=f.events||[];document.getElementById("event-count").textContent=ev.length?ev.length+" event"+(ev.length===1?"":"s"):"quiet frame";if(!ev.length){box.innerHTML='<div class="empty">No event at this timestep. Watch the tokens move, then advance to the next decision point.</div>';return}ev.forEach(x=>{let card=document.createElement("div");card.className="event-card "+eventClass(x.type);let tag=document.createElement("span");tag.className="event-tag";tag.textContent="Round "+(x.round+1)+" · "+x.phase;let text=document.createElement("div");text.className="event-text";text.textContent=x.label;card.append(tag,text);box.appendChild(card)})}
function drawStatus(){let f=D.frames[i],alive=f.alive.filter(Boolean).length,round=f.round+1;document.getElementById("time").innerHTML="Round <strong>"+round+"</strong> · free-play turn <strong>"+f.turn+"</strong>";document.getElementById("frame-count").textContent="frame "+(i+1)+" / "+D.frames.length;document.getElementById("phase").textContent=(f.events||[]).length?"decision point":"free play";if(showRoles){let c=0,q=0;f.alive.forEach((v,a)=>{if(!v)return;if(role(a)==="COALITION")q++;else c++});document.getElementById("alive").textContent=c+" C / "+q+" I"}else document.getElementById("alive").textContent=alive;document.getElementById("round-stat").textContent=round+" / "+D.meetings.length;document.getElementById("meeting-stat").textContent=f.events.some(e=>e.type==="VOTE_CAST"||e.type==="AgentEjected")?"vote":"—"}
function drawTimeline(){let box=document.getElementById("timeline");box.textContent="";D.frames.forEach((f,k)=>{let b=document.createElement("button");b.className="tick "+(k===i?"active ":"")+(k>0&&D.frames[k-1].round!==f.round?"round-change":"");b.title="Round "+(f.round+1)+", turn "+f.turn;b.onclick=()=>{i=k;draw()};box.appendChild(b)})}
function drawRounds(){let box=document.getElementById("rounds");box.textContent="";let seen={};D.frames.forEach((f,k)=>{if(seen[f.round])return;seen[f.round]=k;let m=D.meetings.find(x=>x.round===f.round+1),card=document.createElement("button");card.className="round-card";card.innerHTML='<div class="rname">ROUND '+(f.round+1)+'</div><div class="rmeta">'+(m?(m.incident==="none"?"no incident":m.incident.split(" in ")[0]):"in progress")+'</div>';card.onclick=()=>{i=k;draw()};box.appendChild(card)})}
function drawMeetings(){let box=document.getElementById("meetings");box.textContent="";D.meetings.forEach(m=>{let card=document.createElement("button");card.className="meeting";let first=D.frames.findIndex(f=>f.round===m.round-1);card.onclick=()=>{i=Math.max(0,first);draw()};card.innerHTML='<span class="meeting-num">R'+m.round+'</span><span class="meeting-main"><b>'+(m.incident==="none"?"No incident":m.incident)+'</b><br><span class="muted">'+m.tasks+' tasks · '+m.moves+' moves</span></span><span class="meeting-status">ejected<br>'+m.ejected+'</span>';box.appendChild(card)})}
function drawOutcome(){let o=D.outcome,box=document.getElementById("outcome");box.className="outcome-card "+(o.coalition_favorable?"":"bad");box.innerHTML='<div class="outcome-title">'+(o.coalition_favorable?"COALITION FAVOURABLE":"CREW FAVOURABLE")+'</div><div class="outcome-copy">'+(o.ejected<0?"No player was ejected.":"Final ejection: "+name(o.ejected))+"<br>Roles are "+(showRoles?"visible.":"hidden.")+" Toggle the role view only for analysis."+'</div>'}
function draw(){drawMap();drawEvents();drawStatus();drawTimeline();drawOutcome();document.getElementById("meta").textContent=D.config.name+" · "+n+" agents · "+D.meetings.length+" recorded rounds";document.querySelectorAll(".round-card").forEach((x,k)=>x.classList.toggle("active",D.frames[i].round===k))}
document.getElementById("roles").onclick=e=>{showRoles=!showRoles;e.target.textContent=showRoles?"Hide roles":"Show role labels";draw()};document.getElementById("play").onclick=e=>{if(timer){clearInterval(timer);timer=null;e.target.textContent="▶ Play replay";return}e.target.textContent="Ⅱ Pause";timer=setInterval(()=>{i=(i+1)%D.frames.length;draw()},680)};document.getElementById("prev").onclick=()=>{i=(i-1+D.frames.length)%D.frames.length;draw()};document.getElementById("next").onclick=()=>{i=(i+1)%D.frames.length;draw()};drawMeetings();drawRounds();draw()})();
</script></body></html>"""


# A deliberately game-first presentation view.  The earlier dashboard was clean, but it still
# read like a data product with a map embedded in it.  This version gives the board most of the
# screen and uses ordinary SVG shapes for a small top-down ship: rooms have floors, doors and
# corridors; agents are tiny crewmates; incidents/reports are visual scene changes.  It remains
# self-contained so it can be opened directly from a file or presented without a server.
HTML_TEMPLATE_V3 = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Among Us RL · replay</title>
<style>
:root{--ink:#f4f7ff;--muted:#93a5c5;--bg:#080d19;--deck:#101a2c;--deck2:#0d1628;--edge:#34496e;--cyan:#42e8ff;--blue:#4d8dff;--red:#ff5e6c;--gold:#ffd166;--green:#51e0a4}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:radial-gradient(ellipse at 50% -18%,#243a68 0,#0d1425 38%,#080d19 75%);color:var(--ink);font:14px/1.4 Inter,ui-sans-serif,system-ui,sans-serif}
button{font:inherit;color:inherit;cursor:pointer}.page{max-width:1500px;margin:auto;padding:20px 28px 28px}.top{display:flex;align-items:center;gap:14px;border-bottom:1px solid #243451;padding:0 0 14px}.brand{font-size:12px;font-weight:900;letter-spacing:.18em;color:var(--cyan)}.title{font-size:19px;font-weight:900;letter-spacing:.04em}.grow{flex:1}.top button{border:1px solid #37517a;border-radius:8px;background:#111d33;padding:8px 13px}.top button:hover{border-color:var(--cyan);box-shadow:0 0 17px #42e8ff33}.top .primary{background:var(--cyan);border-color:var(--cyan);color:#07111e;font-weight:900}
.intro{display:flex;justify-content:space-between;align-items:end;gap:22px;padding:20px 0 13px}.eyebrow{color:var(--gold);font-weight:800;font-size:11px;letter-spacing:.17em;text-transform:uppercase}.intro h1{font-size:34px;line-height:1;margin:5px 0 7px}.intro p{margin:0;color:var(--muted);max-width:720px}.step{color:var(--muted);font-size:12px;text-align:right;white-space:nowrap}.step strong{display:block;color:var(--ink);font-size:22px}
.layout{display:grid;grid-template-columns:minmax(640px,1fr) 295px;gap:15px}.board-panel{min-width:0;border:1px solid #2c4167;border-radius:16px;background:linear-gradient(145deg,#121e35,#0b1323);box-shadow:0 22px 50px #0008;overflow:hidden}.board-head{display:flex;justify-content:space-between;align-items:center;padding:13px 17px;border-bottom:1px solid #2a3d60}.board-head span:first-child{font-size:11px;color:var(--muted);font-weight:900;letter-spacing:.16em;text-transform:uppercase}.board-head span:last-child{color:var(--cyan);font-size:12px;font-weight:800}.ship-wrap{position:relative;padding:10px;background:#0b1426}.ship-wrap:before{content:"";position:absolute;inset:10px;background-image:linear-gradient(#8da7ce0c 1px,transparent 1px),linear-gradient(90deg,#8da7ce0c 1px,transparent 1px);background-size:30px 30px;border-radius:11px;pointer-events:none}.ship{position:relative;border:1px solid #39547c;border-radius:13px;overflow:hidden;background:radial-gradient(circle at 50% 47%,#182d4d 0,#0e1a30 70%);box-shadow:inset 0 0 40px #0007}.ship svg{display:block;width:100%;height:auto;min-height:460px}.map-note{padding:5px 16px 10px;color:#7186aa;font-size:11px}.controls{display:flex;align-items:center;gap:8px;padding:12px 17px 7px;border-top:1px solid #243654}.controls button{width:35px;height:31px;border:1px solid #38547c;border-radius:7px;background:#111f37;font-size:17px}.controls button:hover{border-color:var(--cyan)}.scrub{flex:1;display:flex;gap:3px;align-items:center}.tick{flex:1;min-width:5px;height:8px;padding:0;border:0;border-radius:10px;background:#293c60}.tick.round{background:#c99d41}.tick.active{height:14px;background:var(--cyan);box-shadow:0 0 13px #42e8ffaa}.tick.active.round{background:#ffe18b}.clock{display:flex;justify-content:space-between;padding:1px 17px 12px;color:var(--muted);font-size:12px}.clock strong{color:var(--ink)}
.side{display:flex;flex-direction:column;gap:12px}.side-panel{border:1px solid #2c4167;border-radius:13px;background:linear-gradient(145deg,#121e35,#0b1323);box-shadow:0 10px 24px #0005;overflow:hidden}.side-head{padding:12px 14px 9px;border-bottom:1px solid #293d5d;color:var(--muted);font-size:10px;font-weight:900;letter-spacing:.16em;text-transform:uppercase}.moment{padding:14px;border-left:3px solid var(--cyan);background:#0f1c32}.moment.incident{border-color:var(--gold)}.moment.report{border-color:var(--green)}.moment.ejection{border-color:var(--red)}.moment-title{font-size:15px;font-weight:900}.moment-copy{margin-top:4px;color:var(--muted);font-size:12px}.feed{padding:10px 12px;display:flex;flex-direction:column;gap:7px;max-height:245px;overflow:auto}.feed-row{padding:8px 9px;border-radius:7px;background:#0d172b;border-left:2px solid #425d89;font-size:12px}.feed-row.incident{border-color:var(--gold)}.feed-row.report{border-color:var(--green)}.feed-row.ejection{border-color:var(--red)}.feed-tag{display:block;color:#7e93b8;text-transform:uppercase;letter-spacing:.09em;font-size:9px;margin-bottom:2px}.quiet{color:var(--muted);font-size:12px;padding:5px 2px}.rounds{padding:10px 12px;display:flex;flex-direction:column;gap:6px;max-height:220px;overflow:auto}.round{display:grid;grid-template-columns:42px 1fr auto;gap:8px;align-items:center;border:1px solid #233756;border-radius:8px;background:#0d172a;padding:8px 9px;text-align:left}.round:hover,.round.active{border-color:var(--cyan);background:#122442}.round-num{color:var(--cyan);font-weight:900}.round-copy{font-size:11px;color:#e8efff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.round-out{font-size:10px;color:var(--muted)}.outcome{padding:14px;border:1px solid #51e0a466;border-radius:13px;background:linear-gradient(145deg,#11302f,#0d1729)}.outcome.bad{border-color:#ff5e6c66;background:linear-gradient(145deg,#361c2b,#0d1729)}.outcome b{display:block;color:var(--green);font-size:14px}.outcome.bad b{color:var(--red)}.outcome span{display:block;color:var(--muted);font-size:11px;margin-top:4px}.legend{display:flex;gap:15px;flex-wrap:wrap;padding:10px 17px 15px;color:#7186aa;font-size:11px}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;background:var(--blue)}.legend .i2{background:var(--red)}.legend .i3{background:var(--gold)}.foot{text-align:center;color:#647a9d;font-size:11px;padding-top:15px}
@media(max-width:1050px){.layout{grid-template-columns:1fr}.side{display:grid;grid-template-columns:1fr 1fr}.side .outcome{grid-column:1/-1}}@media(max-width:700px){.page{padding:12px}.intro{display:block}.step{text-align:left;margin-top:10px}.ship svg{min-height:320px}.side{display:flex}.brand{font-size:10px}.title{font-size:16px}}
</style></head><body>
<div class="page"><header class="top"><span class="brand">AMONG US // RL LAB</span><span class="title">OMNISCIENT REPLAY</span><span class="grow"></span><button id="roles">Show role labels</button><button class="primary" id="play">▶ Play</button></header>
<section class="intro"><div><div class="eyebrow">F4 · learned movement and meetings</div><h1>One game, frame by frame</h1><p id="meta"></p></div><div class="step"><strong id="step">ROUND 1</strong><span id="phase">free play</span></div></section>
<main class="layout"><section class="board-panel"><div class="board-head"><span>Ship map</span><span id="moment-label">players moving</span></div><div class="ship-wrap"><div class="ship"><svg id="map" viewBox="0 0 920 610" aria-label="Top-down replay of a multi-round game"></svg></div><div class="map-note">Each frame is one observed free-play turn. Gold bars show assigned crew tasks; green bars show completed tasks cumulatively across the game. Colored tokens are player identities; role colors appear only in the analysis view.</div></div><div class="controls"><button id="prev" title="Previous frame">‹</button><button id="next" title="Next frame">›</button><div id="scrub" class="scrub"></div><button id="play2" title="Play or pause">▶</button></div><div class="clock"><span id="time"></span><span id="count"></span></div><div class="legend"><span><i></i>player</span><span><i class="i2"></i>impostor in analysis view</span><span><i class="i3"></i>incident / meeting</span></div></section>
<aside class="side"><section class="side-panel"><div class="side-head">What is happening now</div><div id="moment" class="moment"></div></section><section class="side-panel"><div class="side-head">Replay feed</div><div id="feed" class="feed"></div></section><section class="side-panel"><div class="side-head">Rounds <span style="float:right;color:var(--cyan);letter-spacing:0">click to jump</span></div><div id="rounds" class="rounds"></div></section><div id="outcome" class="outcome"></div></aside></main><div class="foot">Validated replay of recorded actions · public view hides the roles · this is a replay, not a live simulation</div></div>
<script id="data" type="application/json">__DATA__</script><script>
(function(){"use strict";const D=JSON.parse(document.getElementById("data").textContent),rooms=D.config.rooms,n=D.config.n_agents;let frame=0,showRoles=false,timer=null,phaseTimer=null,showAfter=false,moveT=0,currentPhase="turn_start";
const C=["#42e8ff","#a88cff","#ffcf65","#ff8293","#51e0a4","#ff9f5a","#c7e36b","#65a8ff","#e58cff"];
const roomBox={Hub:[400,315,270,185],Electrical:[55,80,280,175],Navigation:[585,80,280,175],Medbay:[55,390,280,175]};
function E(t,a,txt){const x=document.createElementNS("http://www.w3.org/2000/svg",t);Object.keys(a||{}).forEach(k=>x.setAttribute(k,a[k]));if(txt!==undefined)x.textContent=txt;return x}
function role(a){return D.roles["A"+a]||"CREW"}function who(a){return a==null?"nobody":"A"+a}function box(k){if(roomBox[k])return roomBox[k];const j=rooms.indexOf(k),col=j%3,row=Math.floor(j/3);return [45+col*300,75+row*230,260,170]}
function center(k){const b=box(rooms[k]);return [b[0]+b[2]/2,b[1]+b[3]/2]}
function gradientDefs(s){const d=E("defs",{}),pat=E("pattern",{id:"floor",width:22,height:22,patternUnits:"userSpaceOnUse"}),line1=E("path",{d:"M 0 0 L 22 0 M 0 0 L 0 22",stroke:"#91aed00c",fill:"none"});pat.appendChild(line1);d.appendChild(pat);const fl=E("filter",{id:"softGlow",x:"-60%",y:"-60%",width:"220%",height:"220%"}),blur=E("feGaussianBlur",{stdDeviation:5,result:"b"}),merge=E("feMerge",{});merge.appendChild(E("feMergeNode",{in:"b"}));merge.appendChild(E("feMergeNode",{in:"SourceGraphic"}));fl.appendChild(blur);fl.appendChild(merge);d.appendChild(fl);s.appendChild(d)}
function roomFill(k){return k==="Hub"?"#17335a":"#14233d"}
function drawCorridors(s){const h=rooms.indexOf("Hub");if(h<0)return;const hc=center(h);rooms.forEach((r,k)=>{if(k===h)return;const c=center(k),mx=hc[0],my=hc[1],x=c[0],y=c[1],d=Math.abs(x-mx)>Math.abs(y-my)?("M"+x+" "+y+" H"+mx):("M"+x+" "+y+" V"+my);s.appendChild(E("path",{d:d,stroke:"#07101f","stroke-width":62,"stroke-linecap":"round",fill:"none"}));s.appendChild(E("path",{d:d,stroke:"#263f66","stroke-width":49,"stroke-linecap":"round",fill:"none"}));s.appendChild(E("path",{d:d,stroke:"#4d678d","stroke-width":2,"stroke-dasharray":"5 9",fill:"none",opacity:.8}))})}
function drawRoom(s,k,active,tasks){const b=box(k),x=b[0],y=b[1],w=b[2],h=b[3],g=E("g",{});g.appendChild(E("rect",{x,y,width:w,height:h,rx:22,fill:roomFill(k),stroke:active?"#ff5e6c":"#46638d","stroke-width":active?4:2,filter:active?"url(#softGlow)":undefined}));g.appendChild(E("rect",{x:x+8,y:y+8,width:w-16,height:h-16,rx:17,fill:"url(#floor)",stroke:"#273f63", "stroke-width":1}));g.appendChild(E("text",{x:x+20,y:y+30,fill:"#edf4ff","font-size":17,"font-weight":900,"letter-spacing":".09em"},k.toUpperCase()));g.appendChild(E("text",{x:x+20,y:y+50,fill:"#7f96b9","font-size":9,"letter-spacing":".18em"},k==="Hub"?"CENTRAL HUB":"TASK ROOM"));
for(let q=0;q<3;q++){g.appendChild(E("rect",{x:x+20+q*58,y:y+h-33,width:42,height:9,rx:4,fill:q<tasks?"#51e0a4":"#2b4165",opacity:q<tasks?1:.7}))}s.appendChild(g)}
function drawCrew(s,a,x,y,dead){const coal=showRoles&&role(a)==="COALITION",col=showRoles?(coal?"#ff5e6c":"#4d8dff"):C[a%C.length],g=E("g",{transform:"translate("+x+","+y+")",opacity:dead?.25:1});g.appendChild(E("ellipse",{cx:0,cy:23,rx:18,ry:5,fill:"#050a13",opacity:.65}));g.appendChild(E("rect",{x:-18,y:-5,width:8,height:25,rx:4,fill:col,stroke:"#09111d","stroke-width":2}));g.appendChild(E("rect",{x:-14,y:-20,width:29,height:42,rx:13,fill:col,stroke:"#09111d","stroke-width":2}));g.appendChild(E("rect",{x:-4,y:-12,width:18,height:11,rx:5,fill:"#b9f6ff",stroke:"#0b263a","stroke-width":2}));g.appendChild(E("path",{d:"M-11 20 v8 h8 v-8 M3 20 v8 h8 v-8",fill:col,stroke:"#09111d","stroke-width":2}));if(dead)g.appendChild(E("text",{x:0,y:-30,"text-anchor":"middle",fill:"#ff7180","font-size":19,"font-weight":900},"×"));g.appendChild(E("text",{x:0,y:-34,"text-anchor":"middle",fill:"#f6fbff","font-size":11,"font-weight":900},who(a)));if(showRoles)g.appendChild(E("text",{x:0,y:43,"text-anchor":"middle",fill:coal?"#ff9da6":"#8fb9ff","font-size":8,"font-weight":800},coal?"IMPOSTOR":"CREW"));s.appendChild(g)}
function taskCount(f,k){return (f.events||[]).filter(e=>e.type==="TaskAttempted"&&e.payload&&e.payload.room===k&&e.payload.success).length}
function drawRoom(s,k,active,tasks){k=typeof k==="number"?rooms[k]:k;const b=box(k),x=b[0],y=b[1],w=b[2],h=b[3],g=E("g",{});g.appendChild(E("rect",{x,y,width:w,height:h,rx:22,fill:roomFill(k),stroke:active?"#ff5e6c":"#46638d","stroke-width":active?4:2,filter:active?"url(#softGlow)":undefined}));g.appendChild(E("rect",{x:x+8,y:y+8,width:w-16,height:h-16,rx:17,fill:"url(#floor)",stroke:"#273f63","stroke-width":1}));g.appendChild(E("text",{x:x+20,y:y+30,fill:"#edf4ff","font-size":17,"font-weight":900,"letter-spacing":".09em"},k.toUpperCase()));g.appendChild(E("text",{x:x+20,y:y+50,fill:"#7f96b9","font-size":9,"letter-spacing":".18em"},k==="Hub"?"CENTRAL HUB":"TASK ROOM"));for(let q=0;q<3;q++)g.appendChild(E("rect",{x:x+20+q*58,y:y+h-33,width:42,height:9,rx:4,fill:q<tasks?"#51e0a4":"#2b4165",opacity:q<tasks?1:.7}));s.appendChild(g)}
function drawMap(){const f=D.frames[frame],prev=frame?D.frames[frame-1]:null,s=document.getElementById("map");s.textContent="";gradientDefs(s);s.appendChild(E("rect",{x:0,y:0,width:920,height:610,fill:"#0c172a"}));drawCorridors(s);const incident=(f.events||[]).find(e=>e.type==="IncidentCreated"),report=(f.events||[]).find(e=>e.type==="ReportCalled"),eject=(f.events||[]).find(e=>e.type==="AgentEjected");rooms.forEach((r,k)=>drawRoom(s,k,!!(incident&&incident.payload&&incident.payload.room===k),taskCount(f,k)));if(report){const c=center(rooms.indexOf("Hub"));s.appendChild(E("circle",{cx:c[0],cy:c[1],r:102,fill:"none",stroke:"#51e0a4","stroke-width":4,opacity:.75,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:c[0],y:c[1]-98,"text-anchor":"middle",fill:"#51e0a4","font-size":13,"font-weight":900,"letter-spacing":".15em"},"MEETING CALLED"))}if(incident){const c=center(incident.payload.room);s.appendChild(E("circle",{cx:c[0],cy:c[1],r:95,fill:"none",stroke:"#ffd166","stroke-width":5,opacity:.9,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:c[0],y:c[1]-97,"text-anchor":"middle",fill:"#ffd166","font-size":15,"font-weight":900,"letter-spacing":".12em"},"INCIDENT"))}if(eject){const c=center(rooms.indexOf("Hub"));s.appendChild(E("text",{x:c[0],y:c[1]+102,"text-anchor":"middle",fill:"#ff5e6c","font-size":14,"font-weight":900,"letter-spacing":".12em"},eject.payload.agent==null?"VOTE SKIPPED":"EJECTED "+who(eject.payload.agent)))}
const at={};f.positions.forEach((r,a)=>{if(r>=0&&(f.alive[a]||showRoles)) (at[r]??=[]).push(a)});Object.keys(at).forEach(k=>{const arr=at[k],b=box(rooms[+k]),slots=[[.22,.40],[.43,.40],[.64,.40],[.32,.66],[.53,.66],[.74,.66],[.22,.78],[.43,.78],[.64,.78]];arr.forEach((a,j)=>{const q=slots[j%slots.length];drawCrew(s,a,b[0]+b[2]*q[0],b[1]+b[3]*q[1],!f.alive[a])})});if(prev){f.positions.forEach((to,a)=>{const from=prev.positions[a];if(f.alive[a]&&from>=0&&to>=0&&from!==to){const p=center(from),q=center(to);s.appendChild(E("path",{d:"M"+p[0]+" "+p[1]+" L"+q[0]+" "+q[1],stroke:C[a%C.length],"stroke-width":3,"stroke-dasharray":"7 8",opacity:.9,filter:"url(#softGlow)"}))}})}}
function drawCorridors(s){
const routes={
"Electrical|Navigation":"M335 165 H585",
"Electrical|Hub":"M335 225 H375 V285 H470 V315",
"Hub|Navigation":"M610 255 H635 V285 H590 V315",
"Hub|Medbay":"M335 475 H370 V390 H400"
};
const seen=new Set();D.config.edges.forEach(edge=>{const a=edge[0],b=edge[1],key=[a,b].sort().join("|");if(seen.has(key))return;seen.add(key);const d=routes[key];if(!d)return;s.appendChild(E("path",{d,stroke:"#060d19","stroke-width":68,"stroke-linecap":"round","stroke-linejoin":"round",fill:"none"}));s.appendChild(E("path",{d,stroke:"#29466f","stroke-width":52,"stroke-linecap":"round","stroke-linejoin":"round",fill:"none"}));s.appendChild(E("path",{d,stroke:"#7890b4","stroke-width":2,"stroke-dasharray":"6 10",fill:"none",opacity:.9}));});
}
function drawMap(){
const f=D.frames[frame],prev=frame?D.frames[frame-1]:null,s=document.getElementById("map");s.textContent="";gradientDefs(s);s.appendChild(E("rect",{x:0,y:0,width:920,height:610,fill:"#0c172a"}));drawCorridors(s);
const incident=(f.events||[]).find(e=>e.type==="IncidentCreated"),report=(f.events||[]).find(e=>e.type==="ReportCalled"),eject=(f.events||[]).find(e=>e.type==="AgentEjected");rooms.forEach((r,k)=>drawRoom(s,r,!!(incident&&incident.payload&&incident.payload.room===k),taskCount(f,k)));
if(incident){const b=box(rooms[incident.payload.room]);s.appendChild(E("circle",{cx:b[0]+b[2]-22,cy:b[1]+25,r:16,fill:"#ffd166",stroke:"#fff3bc","stroke-width":3,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-22,y:b[1]+31,"text-anchor":"middle",fill:"#1a2131","font-size":17,"font-weight":900},"!"));s.appendChild(E("text",{x:b[0]+b[2]-45,y:b[1]-12,"text-anchor":"end",fill:"#ffd166","font-size":12,"font-weight":900,"letter-spacing":".12em"},"INCIDENT"))}
if(report){const b=box("Hub");s.appendChild(E("circle",{cx:b[0]+b[2]-23,cy:b[1]+25,r:16,fill:"#51e0a4",stroke:"#c9ffe8","stroke-width":3,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-23,y:b[1]+31,"text-anchor":"middle",fill:"#06201b","font-size":15,"font-weight":900},"!"));s.appendChild(E("text",{x:b[0]+b[2]-45,y:b[1]-12,"text-anchor":"end",fill:"#51e0a4","font-size":12,"font-weight":900,"letter-spacing":".12em"},"MEETING"))}
if(eject){const b=box("Hub");s.appendChild(E("text",{x:b[0]+b[2]/2,y:b[1]+b[3]+25,"text-anchor":"middle",fill:"#ff7180","font-size":12,"font-weight":900,"letter-spacing":".12em"},eject.payload.agent==null?"VOTE SKIPPED":"EJECTED "+who(eject.payload.agent)))}
if(prev){f.positions.forEach((to,a)=>{const from=prev.positions[a];if(f.alive[a]&&from>=0&&to>=0&&from!==to){const p=center(from),q=center(to);s.appendChild(E("path",{d:"M"+p[0]+" "+p[1]+" L"+q[0]+" "+q[1],stroke:C[a%C.length],"stroke-width":3,"stroke-dasharray":"7 8",opacity:.9,filter:"url(#softGlow)"}))}})}
const at={};f.positions.forEach((r,a)=>{if(r>=0&&(f.alive[a]||showRoles))(at[r]??=[]).push(a)});Object.keys(at).forEach(k=>{const arr=at[k],b=box(rooms[+k]),slots=[[.22,.57],[.44,.57],[.66,.57],[.33,.78],[.55,.78],[.77,.78],[.22,.88],[.44,.88],[.66,.88]];arr.forEach((a,j)=>{const q=slots[j%slots.length];drawCrew(s,a,b[0]+b[2]*q[0],b[1]+b[3]*q[1],!f.alive[a])})});
}
function highlightType(e){return e.type==="IncidentCreated"?"incident":e.type==="ReportCalled"?"report":e.type==="AgentEjected"?"ejection":""}
function readable(e){if(e.type==="TaskAttempted")return e.label; if(e.type==="AgentMoved")return who(e.payload.agent)+" moved to "+rooms[e.payload.to]; return e.label}
function drawFeed(){const f=D.frames[frame],important=(f.events||[]).filter(e=>["IncidentCreated","ReportCalled","AgentEjected","ClaimMade","ResponseMade","VoteCast","RoundStarted","EpisodeEnded","TaskAttempted"].includes(e.type)),feed=document.getElementById("feed");feed.textContent="";if(!important.length){feed.innerHTML='<div class="quiet">Quiet turn — agents are moving and observing.</div>';return}important.slice(-6).forEach(e=>{const d=document.createElement("div");d.className="feed-row "+highlightType(e);const tag=document.createElement("span");tag.className="feed-tag";tag.textContent="Round "+(e.round+1)+" · "+e.phase;const text=document.createElement("span");text.textContent=readable(e);d.append(tag,text);feed.appendChild(d)})}
function drawMoment(){const f=D.frames[frame],events=f.events||[],box=document.getElementById("moment"),inc=events.find(e=>e.type==="IncidentCreated"),rep=events.find(e=>e.type==="ReportCalled"),ej=events.find(e=>e.type==="AgentEjected"),task=events.filter(e=>e.type==="TaskAttempted"&&e.payload.success).length;let title="FREE PLAY",copy=task?task+" task"+(task===1?"":"s")+" completed this turn. Agents are choosing where to move.":"Players are moving, observing, and choosing their next actions.";box.className="moment";if(inc){box.classList.add("incident");title="INCIDENT CREATED";copy=inc.label}else if(rep){box.classList.add("report");title="MEETING CALLED";copy=rep.label}else if(ej){box.classList.add("ejection");title=ej.payload.agent==null?"VOTE SKIPPED":"PLAYER EJECTED";copy=ej.label}box.innerHTML="<div class=moment-title>"+title+"</div><div class=moment-copy>"+copy+"</div>";document.getElementById("moment-label").textContent=title.toLowerCase()}
function drawStatus(){const f=D.frames[frame];document.getElementById("meta").textContent=D.config.name+" · "+n+" players · "+D.meetings.length+" recorded rounds";document.getElementById("step").textContent="ROUND "+(f.round+1);document.getElementById("phase").textContent=(f.events||[]).length?"decision point":"free play";document.getElementById("time").innerHTML="Round <strong>"+(f.round+1)+"</strong> · turn <strong>"+f.turn+"</strong>";document.getElementById("count").textContent="frame "+(frame+1)+" / "+D.frames.length}
function drawScrub(){const s=document.getElementById("scrub");s.textContent="";D.frames.forEach((f,k)=>{const b=document.createElement("button");b.className="tick "+(k===frame?"active ":"")+(k&&D.frames[k-1].round!==f.round?"round":"");b.title="Round "+(f.round+1)+", turn "+f.turn;b.onclick=()=>{frame=k;draw()};s.appendChild(b)})}
function drawRounds(){const s=document.getElementById("rounds");s.textContent="";D.meetings.forEach(m=>{const b=document.createElement("button");b.className="round";const k=D.frames.findIndex(f=>f.round===m.round-1);b.onclick=()=>{frame=Math.max(0,k);draw()};b.innerHTML='<span class=round-num>R'+m.round+'</span><span class=round-copy>'+ (m.incident==="none"?"No incident":m.incident)+'</span><span class=round-out>'+m.ejected+'</span>';s.appendChild(b)})}
function drawOutcome(){const o=D.outcome,s=document.getElementById("outcome");s.className="outcome "+(o.coalition_favorable?"":"bad");s.innerHTML="<b>"+(o.coalition_favorable?"COALITION FAVOURABLE":"CREW FAVOURABLE")+"</b><span>"+(o.ejected<0?"No player was ejected.":"Final ejection: "+who(o.ejected))+"<br>Role information is "+(showRoles?"visible":"hidden")+".</span>"}
function draw(){drawMap();drawMoment();drawFeed();drawStatus();drawScrub();drawOutcome();document.querySelectorAll(".round").forEach((x,k)=>x.classList.toggle("active",D.frames[frame].round===k))}
function toggle(){showRoles=!showRoles;document.getElementById("roles").textContent=showRoles?"Hide roles":"Show role labels";draw()}
function play(btn){if(timer){clearInterval(timer);timer=null;btn.textContent="▶ Play";document.getElementById("play").textContent="▶ Play";document.getElementById("play2").textContent="▶";return}timer=setInterval(()=>{frame=(frame+1)%D.frames.length;draw()},900);document.getElementById("play").textContent="Ⅱ Pause";document.getElementById("play2").textContent="Ⅱ"}
function viewPositions(f){return showAfter&&f.positions_after?f.positions_after:f.positions}
function viewAlive(f){return showAfter&&f.alive_after?f.alive_after:f.alive}
function drawMap(){
const f=D.frames[frame],s=document.getElementById("map"),positions=viewPositions(f),alive=viewAlive(f),moves=showAfter&&f.positions_after?f.positions_after.map((to,a)=>to!==f.positions[a]):[];s.textContent="";gradientDefs(s);s.appendChild(E("rect",{x:0,y:0,width:920,height:610,fill:"#0c172a"}));drawCorridors(s);
const incident=(f.events||[]).find(e=>e.type==="IncidentCreated"),report=(f.events||[]).find(e=>e.type==="ReportCalled"),eject=(f.events||[]).find(e=>e.type==="AgentEjected");rooms.forEach((r,k)=>drawRoom(s,r,!!(incident&&incident.payload&&incident.payload.room===k),taskCount(f,k)));
if(f.round_start&&!showAfter){s.appendChild(E("rect",{x:314,y:14,width:292,height:42,rx:21,fill:"#173c63",stroke:"#42e8ff","stroke-width":2,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:460,y:41,"text-anchor":"middle",fill:"#d9fbff","font-size":17,"font-weight":900,"letter-spacing":".15em"},"ROUND "+(f.round+1)+" START"))}
if(incident){const b=box(rooms[incident.payload.room]);s.appendChild(E("circle",{cx:b[0]+b[2]-22,cy:b[1]+25,r:16,fill:"#ffd166",stroke:"#fff3bc","stroke-width":3,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-22,y:b[1]+31,"text-anchor":"middle",fill:"#1a2131","font-size":17,"font-weight":900},"!"))}
if(report){const b=box("Hub");s.appendChild(E("circle",{cx:b[0]+b[2]-23,cy:b[1]+25,r:16,fill:"#51e0a4",stroke:"#c9ffe8","stroke-width":3,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-23,y:b[1]+31,"text-anchor":"middle",fill:"#06201b","font-size":15,"font-weight":900},"!"))}
if(eject){const b=box("Hub");s.appendChild(E("text",{x:b[0]+b[2]/2,y:b[1]+b[3]+25,"text-anchor":"middle",fill:"#ff7180","font-size":12,"font-weight":900,"letter-spacing":".12em"},eject.payload.agent==null?"VOTE SKIPPED":"EJECTED "+who(eject.payload.agent)))}
if(showAfter&&f.positions_after){f.positions_after.forEach((to,a)=>{if(!moves[a]||f.positions[a]<0||to<0)return;const p=center(f.positions[a]),q=center(to);s.appendChild(E("path",{d:"M"+p[0]+" "+p[1]+" L"+q[0]+" "+q[1],stroke:C[a%C.length],"stroke-width":4,"stroke-dasharray":"7 8",opacity:.95,filter:"url(#softGlow)"}))})}
const at={};positions.forEach((r,a)=>{if(r>=0&&(alive[a]||showAfter||showRoles))(at[r]??=[]).push(a)});Object.keys(at).forEach(k=>{const arr=at[k],b=box(rooms[+k]),slots=[[.22,.57],[.44,.57],[.66,.57],[.33,.78],[.55,.78],[.77,.78],[.22,.88],[.44,.88],[.66,.88]];arr.forEach((a,j)=>{const q=slots[j%slots.length];drawCrew(s,a,b[0]+b[2]*q[0],b[1]+b[3]*q[1],!alive[a])})});
(f.events||[]).filter(e=>e.type==="TaskAttempted"&&e.payload&&e.payload.success).forEach(e=>{const b=box(rooms[e.payload.room]);s.appendChild(E("circle",{cx:b[0]+b[2]-70,cy:b[1]+b[3]-29,r:9,fill:"#51e0a4",filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-70,y:b[1]+b[3]-25,"text-anchor":"middle",fill:"#06201b","font-size":12,"font-weight":900},"✓"))});
}
function drawMoment(){const f=D.frames[frame],events=f.events||[],boxEl=document.getElementById("moment"),inc=events.find(e=>e.type==="IncidentCreated"),rep=events.find(e=>e.type==="ReportCalled"),ej=events.find(e=>e.type==="AgentEjected"),task=events.filter(e=>e.type==="TaskAttempted"&&e.payload.success).length,moved=events.filter(e=>e.type==="AgentMoved").length;let title=showAfter?"ACTIONS RESOLVED":"TURN BEGINS",copy=moved?moved+" player"+(moved===1?"":"s")+" moved this turn.":"Players are choosing actions.";boxEl.className="moment";if(f.round_start&&!showAfter){title="ROUND "+(f.round+1)+" START";copy="Players enter the map and begin a new free-play round."}if(task){title="TASKS COMPLETE";copy=task+" task"+(task===1?"":"s")+" completed this turn."}if(inc&&showAfter){boxEl.classList.add("incident");title="INCIDENT CREATED";copy=inc.label}else if(rep){boxEl.classList.add("report");title="MEETING CALLED";copy=rep.label}else if(ej&&showAfter){boxEl.classList.add("ejection");title=ej.payload.agent==null?"VOTE SKIPPED":"PLAYER EJECTED";copy=ej.label}boxEl.innerHTML="<div class=moment-title>"+title+"</div><div class=moment-copy>"+copy+"</div>";document.getElementById("moment-label").textContent=title.toLowerCase()}
function drawStatus(){const f=D.frames[frame];document.getElementById("meta").textContent=D.config.name+" · "+n+" players · "+D.meetings.length+" recorded rounds";document.getElementById("step").textContent="ROUND "+(f.round+1);document.getElementById("phase").textContent=f.round_start&&!showAfter?"round start":showAfter?"actions resolved":"turn begins";document.getElementById("time").innerHTML="Round <strong>"+(f.round+1)+"</strong> · turn <strong>"+f.turn+"</strong> · "+(showAfter?"after actions":"before actions");document.getElementById("count").textContent="frame "+(frame+1)+" / "+D.frames.length}
function draw(){drawMap();drawMoment();drawFeed();drawStatus();drawScrub();drawOutcome();document.querySelectorAll(".round").forEach((x,k)=>x.classList.toggle("active",D.frames[frame].round===k))}
function toggle(){showRoles=!showRoles;document.getElementById("roles").textContent=showRoles?"Hide roles":"Show role labels";draw()}
function play(btn){if(timer){clearInterval(timer);timer=null;if(phaseTimer)clearTimeout(phaseTimer);btn.textContent="▶ Play";document.getElementById("play").textContent="▶ Play";document.getElementById("play2").textContent="▶";return}const resolve=()=>{showAfter=false;draw();phaseTimer=setTimeout(()=>{showAfter=true;draw()},420)};resolve();timer=setInterval(()=>{frame=(frame+1)%D.frames.length;resolve()},900);document.getElementById("play").textContent="Ⅱ Pause";document.getElementById("play2").textContent="Ⅱ"}
const tokenSlots=[[.22,.57],[.44,.57],[.66,.57],[.33,.78],[.55,.78],[.77,.78],[.22,.88],[.44,.88],[.66,.88]];
function tokenLayout(positions,alive,includeDead){const grouped={};positions.forEach((room,a)=>{if(room<0)return;if(!alive[a]&&!includeDead&&!showRoles)return;(grouped[room]??=[]).push(a)});const out={};Object.keys(grouped).forEach(k=>{const b=box(rooms[+k]);grouped[k].forEach((a,j)=>{const q=tokenSlots[j%tokenSlots.length];out[a]={x:b[0]+b[2]*q[0],y:b[1]+b[3]*q[1]}})});return out}
function routePoints(from,to){const direct={"Electrical|Hub":[[335,225],[375,225],[375,285],[470,285],[470,315]],"Electrical|Navigation":[[335,165],[585,165]],"Hub|Medbay":[[400,390],[370,390],[370,475],[335,475]],"Hub|Navigation":[[590,315],[635,285],[635,255],[610,255]]};const adj=Array.from({length:rooms.length},()=>[]);D.config.edges.forEach(edge=>{const a=rooms.indexOf(edge[0]),b=rooms.indexOf(edge[1]);if(a>=0&&b>=0){adj[a].push(b);adj[b].push(a)}});const prev=Array(rooms.length).fill(-1),queue=[from];prev[from]=from;for(let q=0;q<queue.length;q++){const u=queue[q];if(u===to)break;adj[u].forEach(v=>{if(prev[v]<0){prev[v]=u;queue.push(v)}})}if(prev[to]<0)return [center(from),center(to)];const path=[];for(let u=to;;u=prev[u]){path.push(u);if(u===from)break}path.reverse();let points=[];for(let j=0;j<path.length-1;j++){const a=path[j],b=path[j+1],key=[rooms[a],rooms[b]].sort().join("|"),base=direct[key];let seg=base?(rooms[a]===key.split("|")[0]?base:base.slice().reverse()):[center(a),center(b)];if(j>0)points.push(center(a));points=points.concat(seg)}return points}
function along(points,t){if(points.length<2)return points[0];let total=0;for(let j=1;j<points.length;j++)total+=Math.hypot(points[j][0]-points[j-1][0],points[j][1]-points[j-1][1]);if(!total)return points[0];let wanted=total*Math.max(0,Math.min(1,t));for(let j=1;j<points.length;j++){const a=points[j-1],b=points[j],len=Math.hypot(b[0]-a[0],b[1]-a[1]);if(wanted<=len){const u=wanted/len;return [a[0]+u*(b[0]-a[0]),a[1]+u*(b[1]-a[1])]}wanted-=len}return points[points.length-1]}
function drawMap(){const f=D.frames[frame],t=moveT,s=document.getElementById("map"),prePos=f.positions,postPos=f.positions_after||f.positions,preAlive=f.alive,postAlive=f.alive_after||f.alive; s.textContent="";gradientDefs(s);s.appendChild(E("rect",{x:0,y:0,width:920,height:610,fill:"#0c172a"}));drawCorridors(s);const incident=(f.events||[]).find(e=>e.type==="IncidentCreated"),report=(f.events||[]).find(e=>e.type==="ReportCalled"),eject=(f.events||[]).find(e=>e.type==="AgentEjected");rooms.forEach((r,k)=>drawRoom(s,r,!!(incident&&incident.payload&&incident.payload.room===k&&t>.55),t>.65?taskCount(f,k):0));if(f.round_start&&t<.22){s.appendChild(E("rect",{x:314,y:14,width:292,height:42,rx:21,fill:"#173c63",stroke:"#42e8ff","stroke-width":2,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:460,y:41,"text-anchor":"middle",fill:"#d9fbff","font-size":17,"font-weight":900,"letter-spacing":".15em"},"ROUND "+(f.round+1)+" START"))}if(incident&&t>.55){const b=box(rooms[incident.payload.room]);s.appendChild(E("circle",{cx:b[0]+b[2]-22,cy:b[1]+25,r:16,fill:"#ffd166",stroke:"#fff3bc","stroke-width":3,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-22,y:b[1]+31,"text-anchor":"middle",fill:"#1a2131","font-size":17,"font-weight":900},"!"))}if(report&&t>.6){const b=box("Hub");s.appendChild(E("circle",{cx:b[0]+b[2]-23,cy:b[1]+25,r:16,fill:"#51e0a4",stroke:"#c9ffe8","stroke-width":3,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-23,y:b[1]+b[1]+31,"text-anchor":"middle",fill:"#06201b","font-size":15,"font-weight":900},"!"))}if(eject&&t>.85){const b=box("Hub");s.appendChild(E("text",{x:b[0]+b[2]/2,y:b[1]+b[3]+25,"text-anchor":"middle",fill:"#ff7180","font-size":12,"font-weight":900,"letter-spacing":".12em"},eject.payload.agent==null?"VOTE SKIPPED":"EJECTED "+who(eject.payload.agent)))}const pre=tokenLayout(prePos,preAlive,false),post=tokenLayout(postPos,postAlive,true),ids=new Set([...Object.keys(pre),...Object.keys(post)]);ids.forEach(raw=>{const a=+raw;if(!pre[a]&&!showRoles&&t<.98)return;const p0=pre[a]||post[a],p1=post[a]||p0,from=prePos[a],to=postPos[a];let p;if(from>=0&&to>=0&&from!==to){p=along([[p0.x,p0.y],...routePoints(from,to),[p1.x,p1.y]],t)}else p=[p0.x+(p1.x-p0.x)*t,p0.y+(p1.y-p0.y)*t];drawCrew(s,a,p[0],p[1],t>.96&&!postAlive[a])});if(t>.65)(f.events||[]).filter(e=>e.type==="TaskAttempted"&&e.payload&&e.payload.success).forEach(e=>{const b=box(rooms[e.payload.room]);s.appendChild(E("circle",{cx:b[0]+b[2]-70,cy:b[1]+b[3]-29,r:9,fill:"#51e0a4",filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-70,y:b[1]+b[3]-25,"text-anchor":"middle",fill:"#06201b","font-size":12,"font-weight":900},"✓"))})}
function drawMoment(){const f=D.frames[frame],events=f.events||[],boxEl=document.getElementById("moment"),inc=events.find(e=>e.type==="IncidentCreated"),rep=events.find(e=>e.type==="ReportCalled"),ej=events.find(e=>e.type==="AgentEjected"),task=moveT>.65?events.filter(e=>e.type==="TaskAttempted"&&e.payload.success).length:0,moved=events.filter(e=>e.type==="AgentMoved").length;let title=moveT>.05&&moveT<.96?"PLAYERS MOVING":showAfter?"ACTIONS RESOLVED":"TURN BEGINS",copy=moved?moved+" player"+(moved===1?"":"s")+" moving between rooms.":"Players are choosing actions.";boxEl.className="moment";if(f.round_start&&moveT<.22){title="ROUND "+(f.round+1)+" START";copy="Players enter the map and begin a new free-play round."}if(task){title="TASKS COMPLETE";copy=task+" task"+(task===1?"":"s")+" completed this turn."}if(inc&&moveT>.55){boxEl.classList.add("incident");title="INCIDENT CREATED";copy=inc.label}else if(rep&&moveT>.6){boxEl.classList.add("report");title="MEETING CALLED";copy=rep.label}else if(ej&&moveT>.85){boxEl.classList.add("ejection");title=ej.payload.agent==null?"VOTE SKIPPED":"PLAYER EJECTED";copy=ej.label}boxEl.innerHTML="<div class=moment-title>"+title+"</div><div class=moment-copy>"+copy+"</div>";document.getElementById("moment-label").textContent=title.toLowerCase()}
function drawStatus(){const f=D.frames[frame];document.getElementById("meta").textContent=D.config.name+" · "+n+" players · "+D.meetings.length+" recorded rounds";document.getElementById("step").textContent="ROUND "+(f.round+1);document.getElementById("phase").textContent=f.round_start&&moveT<.22?"round start":moveT>.03&&moveT<.97?"players moving":showAfter?"actions resolved":"turn begins";document.getElementById("time").innerHTML="Round <strong>"+(f.round+1)+"</strong> · turn <strong>"+f.turn+"</strong> · "+Math.round(moveT*100)+"% resolved";document.getElementById("count").textContent="frame "+(frame+1)+" / "+D.frames.length}
function play(btn){if(timer){clearInterval(timer);timer=null;if(phaseTimer)cancelAnimationFrame(phaseTimer);btn.textContent="▶ Play";document.getElementById("play").textContent="▶ Play";document.getElementById("play2").textContent="▶";return}const resolve=()=>{showAfter=false;moveT=0;draw();const started=performance.now();const animate=now=>{moveT=Math.min(1,(now-started)/520);if(moveT>=1){showAfter=true;draw();return}drawMap();drawMoment();drawStatus();phaseTimer=requestAnimationFrame(animate)};phaseTimer=requestAnimationFrame(animate)};resolve();timer=setInterval(()=>{frame=(frame+1)%D.frames.length;resolve()},900);document.getElementById("play").textContent="Ⅱ Pause";document.getElementById("play2").textContent="Ⅱ"}
document.getElementById("roles").onclick=toggle;document.getElementById("play").onclick=e=>play(e.target);document.getElementById("play2").onclick=e=>play(e.target);document.getElementById("prev").onclick=()=>stepPhase(-1);document.getElementById("next").onclick=()=>stepPhase(1);document.addEventListener("keydown",e=>{const tag=e.target&&e.target.tagName;if(["INPUT","TEXTAREA","SELECT"].includes(tag)||e.target&&e.target.isContentEditable)return;if(e.key==="ArrowLeft"){e.preventDefault();stepPhase(-1)}else if(e.key==="ArrowRight"){e.preventDefault();stepPhase(1)}});drawRounds();showPhase(false);})();
</script></body></html>"""


def write_html(trace: dict[str, Any], path: Path) -> None:
    blob = json.dumps(trace, separators=(",", ":")).replace("</", "<\\/")
    html = HTML_TEMPLATE_V3.replace("__DATA__", blob)
    meeting_css = ".meeting-network{padding:10px 12px 0}.meeting-network svg{display:block;width:100%;height:auto;border-radius:10px}.meeting-actions{padding:10px 12px;display:flex;flex-direction:column;gap:6px;max-height:280px;overflow:auto}.meeting-action{padding:7px 8px;border-left:3px solid #42e8ff;border-radius:6px;background:#0d172b;font-size:11px}.meeting-action.claim{border-color:#a88cff}.meeting-action.response{border-color:#51e0a4}.meeting-action.vote{border-color:#ffd166}.meeting-action.ejection{border-color:#ff5e6c}.meeting-action .action-tag{color:#7e93b8;text-transform:uppercase;letter-spacing:.09em;font-size:9px}.meeting-action .action-body{color:#e8efff;margin-top:2px}.meeting-actions .quiet{padding:5px 2px}"
    html = html.replace("</style>", meeting_css + "</style>", 1)
    rounds_panel = '<section class="side-panel"><div class="side-head">Rounds <span style="float:right;color:var(--cyan);letter-spacing:0">click to jump</span></div><div id="rounds" class="rounds"></div></section>'
    meeting_panel = '<section class="side-panel"><div class="side-head">Meeting actions</div><div class="meeting-network"><svg id="meeting-network-svg" viewBox="0 0 760 250" role="img" aria-label="Meeting interaction network"></svg></div><div id="meeting-actions" class="meeting-actions"></div></section>'
    html = html.replace(rounds_panel, meeting_panel + rounds_panel, 1)
    # Keep the meeting marker inside the Hub when the browser renders the SVG.
    html = html.replace("y:b[1]+b[1]+31", "y:b[1]+31")
    html = html.replace("frame=(frame-1+D.frames.length)%D.frames.length;draw()", "frame=(frame-1+D.frames.length)%D.frames.length;moveT=0;showAfter=false;draw()")
    html = html.replace("frame=(frame+1)%D.frames.length;draw()", "frame=(frame+1)%D.frames.length;moveT=0;showAfter=false;draw()")
    html = html.replace("frame=k;draw()", "frame=k;moveT=0;showAfter=false;draw()")
    html = html.replace("frame=Math.max(0,k);draw()", "frame=Math.max(0,k);moveT=0;showAfter=false;draw()")
    helper = r'''function phaseRows(f,name){return f.phases&&f.phases[name]?f.phases[name]:[]}
function eventPhase(e){if(e.type==="RoundStarted")return "turn_start";if(e.type==="TaskAttempted")return "tasks";if(e.type==="IncidentCreated")return "incident";if(e.type==="ReportCalled")return "report";if(e.type==="AgentMoved")return "movement";if(e.type==="AgentObserved"||e.type==="MarkerObserved")return "observation";if(["ClaimMade","ResponseMade","VoteCast","AgentEjected"].includes(e.type))return "meeting";return "turn_start"}
function phaseSchedule(f){const p=[["turn_start",260]];if(phaseRows(f,"tasks").length)p.push(["tasks",520]);if(phaseRows(f,"incident").length)p.push(["incident",520]);if(phaseRows(f,"report").length){p.push(["report",650]);if(phaseRows(f,"meeting").length)p.push(["meeting",950])}else{if(phaseRows(f,"movement").length)p.push(["movement",750]);if(phaseRows(f,"observation").length)p.push(["observation",420])}p.push(["resolved",240]);return p.map(x=>({name:x[0],ms:x[1]*2}))}
function phaseTitle(f){const count=(name)=>phaseRows(f,name).length;if(currentPhase==="turn_start")return ["ROUND "+(f.round+1)+" START","Players begin a new free-play round."];if(currentPhase==="tasks")return ["TASKS",""+count("tasks")+" task action"+(count("tasks")===1?"":"s")+" resolve at the current rooms."];if(currentPhase==="incident"){const e=phaseRows(f,"incident")[0];return ["INCIDENT / DEATH",e?e.label:"Incident resolves before movement."]}if(currentPhase==="report"){const e=phaseRows(f,"report")[0];return ["MEETING CALLED",e?e.label:"Movement stops and the meeting begins."]}if(currentPhase==="movement")return ["MOVEMENT","Surviving players move simultaneously along legal routes."];if(currentPhase==="observation")return ["OBSERVATION","Players observe who is present after movement."];if(currentPhase==="meeting")return ["MEETING","Claims, responses, and votes resolve outside the map."];return ["TURN RESOLVED","The ordered turn is complete."]}
function drawMap(){const f=D.frames[frame],t=currentPhase==="movement"?moveT:["observation","resolved","meeting"].includes(currentPhase)?1:0,s=document.getElementById("map"),prePos=f.positions,postPos=f.positions_after||f.positions,preAlive=f.alive,postAlive=f.alive_after||f.alive,incident=(f.events||[]).find(e=>e.type==="IncidentCreated"),report=(f.events||[]).find(e=>e.type==="ReportCalled"),eject=(f.events||[]).find(e=>e.type==="AgentEjected");s.textContent="";gradientDefs(s);s.appendChild(E("rect",{x:0,y:0,width:920,height:610,fill:"#0c172a"}));drawCorridors(s);const taskVisible=currentPhase==="tasks"?moveT>.5:["incident","report","movement","observation","meeting","resolved"].includes(currentPhase);rooms.forEach((r,k)=>drawRoom(s,r,!!(incident&&incident.payload&&incident.payload.room===k&&["incident","report","meeting","resolved"].includes(currentPhase)),taskVisible?taskCount(f,k):0));if(currentPhase==="turn_start"){s.appendChild(E("rect",{x:314,y:14,width:292,height:42,rx:21,fill:"#173c63",stroke:"#42e8ff","stroke-width":2,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:460,y:41,"text-anchor":"middle",fill:"#d9fbff","font-size":17,"font-weight":900,"letter-spacing":".15em"},"ROUND "+(f.round+1)+" START"))}if(incident&&["incident","report","meeting","resolved"].includes(currentPhase)){const b=box(rooms[incident.payload.room]);s.appendChild(E("circle",{cx:b[0]+b[2]-22,cy:b[1]+25,r:16,fill:"#ffd166",stroke:"#fff3bc","stroke-width":3,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-22,y:b[1]+31,"text-anchor":"middle",fill:"#1a2131","font-size":17,"font-weight":900},"!"))}if(report&&["report","meeting","resolved"].includes(currentPhase)){const b=box("Hub");s.appendChild(E("circle",{cx:b[0]+b[2]-23,cy:b[1]+25,r:16,fill:"#51e0a4",stroke:"#c9ffe8","stroke-width":3,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-23,y:b[1]+31,"text-anchor":"middle",fill:"#06201b","font-size":15,"font-weight":900},"!"))}if(eject&&["meeting","resolved"].includes(currentPhase)){const b=box("Hub");s.appendChild(E("text",{x:b[0]+b[2]/2,y:b[1]+b[3]+25,"text-anchor":"middle",fill:"#ff7180","font-size":12,"font-weight":900,"letter-spacing":".12em"},eject.payload.agent==null?"VOTE SKIPPED":"EJECTED "+who(eject.payload.agent)))}const pre=tokenLayout(prePos,preAlive,false),post=tokenLayout(postPos,postAlive,true),ids=new Set([...Object.keys(pre),...Object.keys(post)]);ids.forEach(raw=>{const a=+raw;if(!pre[a]&&!showRoles&&!["incident","report","meeting","resolved"].includes(currentPhase))return;const p0=pre[a]||post[a],p1=post[a]||p0,from=prePos[a],to=postPos[a];let p;if(currentPhase==="movement"&&from>=0&&to>=0&&from!==to)p=along([[p0.x,p0.y],...routePoints(from,to),[p1.x,p1.y]],t);else p=[p0.x+(p1.x-p0.x)*t,p0.y+(p1.y-p0.y)*t];const alive=currentPhase==="incident"&&moveT>.55?postAlive:["report","meeting","observation","resolved"].includes(currentPhase)?postAlive:preAlive;drawCrew(s,a,p[0],p[1],!alive[a])});if(taskVisible)phaseRows(f,"tasks").filter(e=>e.payload&&e.payload.success).forEach(e=>{const b=box(rooms[e.payload.room]);s.appendChild(E("circle",{cx:b[0]+b[2]-70,cy:b[1]+b[3]-29,r:9,fill:"#51e0a4",filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-70,y:b[1]+b[3]-25,"text-anchor":"middle",fill:"#06201b","font-size":12,"font-weight":900},"✓"))});if(currentPhase==="observation")phaseRows(f,"observation").forEach(e=>{const room=e.payload&&e.payload.room;if(room==null)return;const c=center(room);s.appendChild(E("circle",{cx:c[0],cy:c[1],r:34,fill:"none",stroke:"#42e8ff","stroke-width":2,"stroke-dasharray":"5 6",opacity:.9}));})}
function drawMoment(){const f=D.frames[frame],boxEl=document.getElementById("moment"),pc=phaseTitle(f);boxEl.className="moment";if(currentPhase==="incident")boxEl.classList.add("incident");if(["report","meeting"].includes(currentPhase))boxEl.classList.add("report");if(currentPhase==="resolved"&&phaseRows(f,"meeting").length)boxEl.classList.add("ejection");boxEl.innerHTML="<div class=moment-title>"+pc[0]+"</div><div class=moment-copy>"+pc[1]+"</div>";document.getElementById("moment-label").textContent=pc[0].toLowerCase()}
function drawFeed(){const f=D.frames[frame],order={turn_start:0,tasks:1,incident:2,report:3,movement:4,observation:5,meeting:6},max=order[currentPhase]??0,important=(f.events||[]).filter(e=>order[eventPhase(e)]<=max&&["IncidentCreated","ReportCalled","AgentEjected","ClaimMade","ResponseMade","VoteCast","RoundStarted","EpisodeEnded","TaskAttempted","AgentMoved","AgentObserved","MarkerObserved"].includes(e.type)),feed=document.getElementById("feed");feed.textContent="";if(!important.length){feed.innerHTML='<div class="quiet">Waiting for the next phase.</div>';return}important.slice(-8).forEach(e=>{const d=document.createElement("div");d.className="feed-row "+highlightType(e);const tag=document.createElement("span");tag.className="feed-tag";tag.textContent="Round "+(e.round+1)+" · "+eventPhase(e);const text=document.createElement("span");text.textContent=readable(e);d.append(tag,text);feed.appendChild(d)})}
function drawStatus(){const f=D.frames[frame],pc=phaseTitle(f);document.getElementById("meta").textContent=D.config.name+" · "+n+" players · "+D.meetings.length+" recorded rounds";document.getElementById("step").textContent="ROUND "+(f.round+1);document.getElementById("phase").textContent=pc[0].toLowerCase();document.getElementById("time").innerHTML="Round <strong>"+(f.round+1)+"</strong> · turn <strong>"+f.turn+"</strong> · "+pc[0];document.getElementById("count").textContent="frame "+(frame+1)+" / "+D.frames.length}
function draw(){drawMap();drawMoment();drawFeed();drawStatus();drawScrub();drawOutcome();document.querySelectorAll(".round").forEach((x,k)=>x.classList.toggle("active",D.frames[frame].round===k))}
function animateFrame(){if(phaseTimer)cancelAnimationFrame(phaseTimer);showAfter=false;moveT=0;currentPhase="turn_start";draw();const schedule=phaseSchedule(D.frames[frame]);let index=0,started=performance.now();const step=now=>{const phase=schedule[index];currentPhase=phase.name;moveT=Math.min(1,(now-started)/phase.ms);showAfter=["observation","meeting","resolved"].includes(currentPhase);drawMap();drawMoment();drawFeed();drawStatus();if(moveT>=1){index++;if(index>=schedule.length){currentPhase="resolved";moveT=1;showAfter=true;draw();return}started=now}phaseTimer=requestAnimationFrame(step)};phaseTimer=requestAnimationFrame(step)}
function play(btn){if(timer){clearInterval(timer);timer=null;if(phaseTimer)cancelAnimationFrame(phaseTimer);btn.textContent="▶ Play";document.getElementById("play").textContent="▶ Play";document.getElementById("play2").textContent="▶";return}animateFrame();timer=setInterval(()=>{frame=(frame+1)%D.frames.length;animateFrame()},3600);document.getElementById("play").textContent="Ⅱ Pause";document.getElementById("play2").textContent="Ⅱ"}'''
    helper = helper.replace(
        'if(phaseRows(f,"report").length){p.push(["report",650]);if(phaseRows(f,"meeting").length)p.push(["meeting",950])}else{if(phaseRows(f,"movement").length)p.push(["movement",750]);if(phaseRows(f,"observation").length)p.push(["observation",420])}',
        'if(phaseRows(f,"movement").length)p.push(["movement",750]);if(phaseRows(f,"observation").length)p.push(["observation",420]);if(phaseRows(f,"report").length)p.push(["report",650]);if(phaseRows(f,"meeting").length)p.push(["meeting",950]);',
    )
    controls = r'''function assignedTaskCount(k){return D.tasks&&D.tasks.crew_assigned_by_room?Number(D.tasks.crew_assigned_by_room[k]||0):0}
function taskCount(f,k){return (f.events||[]).filter(e=>e.type==="TaskAttempted"&&e.payload&&e.payload.room===k&&e.payload.success&&D.roles["A"+e.payload.agent]==="CREW").length}
function drawRoom(s,k,active,completed){k=typeof k==="number"?rooms[k]:k;const b=box(k),x=b[0],y=b[1],w=b[2],h=b[3],g=E("g",{});g.appendChild(E("rect",{x,y,width:w,height:h,rx:22,fill:roomFill(k),stroke:active?"#ff5e6c":"#46638d","stroke-width":active?4:2,filter:active?"url(#softGlow)":undefined}));g.appendChild(E("rect",{x:x+8,y:y+8,width:w-16,height:h-16,rx:17,fill:"url(#floor)",stroke:"#273f63","stroke-width":1}));g.appendChild(E("text",{x:x+20,y:y+30,fill:"#edf4ff","font-size":17,"font-weight":900,"letter-spacing":".09em"},k.toUpperCase()));g.appendChild(E("text",{x:x+20,y:y+50,fill:"#7f96b9","font-size":9,"letter-spacing":".18em"},k==="Hub"?"CENTRAL HUB":"TASK ROOM"));const assigned=Math.max(0,Math.round(assignedTaskCount(rooms.indexOf(k)))),done=Math.min(assigned,Math.max(0,Math.round(Number(completed)||0))),gap=6,barW=assigned?Math.min(42,Math.max(10,(w-40-gap*Math.max(0,assigned-1))/assigned)):0,total=assigned?assigned*barW+(assigned-1)*gap:0,start=x+(w-total)/2;for(let q=0;q<assigned;q++)g.appendChild(E("rect",{x:start+q*(barW+gap),y:y+h-33,width:barW,height:9,rx:4,fill:q<done?"#51e0a4":"#ffd166",opacity:q<done?1:.85}));s.appendChild(g)}
function completedTaskCount(k){const roomIndex=typeof k==="number"?k:rooms.indexOf(k),last=currentPhase==="turn_start"?frame-1:frame;let total=0;for(let i=0;i<=last;i++)(D.frames[i].events||[]).forEach(e=>{const p=e.payload;if(e.type==="TaskAttempted"&&p&&Number(p.room)===roomIndex&&p.success&&D.roles["A"+p.agent]==="CREW")total++});return total}
function taskCount(f,k){return completedTaskCount(k)}
function drawMap(){const f=D.frames[frame],t=currentPhase==="movement"?moveT:["observation","resolved","meeting"].includes(currentPhase)?1:0,s=document.getElementById("map"),prePos=f.positions,postPos=f.positions_after||f.positions,preAlive=f.alive,postAlive=f.alive_after||f.alive,incident=(f.events||[]).find(e=>e.type==="IncidentCreated"),report=(f.events||[]).find(e=>e.type==="ReportCalled"),eject=(f.events||[]).find(e=>e.type==="AgentEjected");s.textContent="";gradientDefs(s);s.appendChild(E("rect",{x:0,y:0,width:920,height:610,fill:"#0c172a"}));drawCorridors(s);rooms.forEach((r,k)=>drawRoom(s,r,!!(incident&&incident.payload&&incident.payload.room===k&&["incident","report","meeting","resolved"].includes(currentPhase)),completedTaskCount(k)));if(currentPhase==="turn_start"){s.appendChild(E("rect",{x:314,y:14,width:292,height:42,rx:21,fill:"#173c63",stroke:"#42e8ff","stroke-width":2,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:460,y:41,"text-anchor":"middle",fill:"#d9fbff","font-size":17,"font-weight":900,"letter-spacing":".15em"},"ROUND "+(f.round+1)+" START"))}if(incident&&["incident","report","meeting","resolved"].includes(currentPhase)){const b=box(rooms[incident.payload.room]);s.appendChild(E("circle",{cx:b[0]+b[2]-22,cy:b[1]+25,r:16,fill:"#ffd166",stroke:"#fff3bc","stroke-width":3,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-22,y:b[1]+31,"text-anchor":"middle",fill:"#1a2131","font-size":17,"font-weight":900},"!"))}if(report&&["report","meeting","resolved"].includes(currentPhase)){const b=box("Hub");s.appendChild(E("circle",{cx:b[0]+b[2]-23,cy:b[1]+25,r:16,fill:"#51e0a4",stroke:"#c9ffe8","stroke-width":3,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:b[0]+b[2]-23,y:b[1]+31,"text-anchor":"middle",fill:"#06201b","font-size":15,"font-weight":900},"!"))}if(eject&&["meeting","resolved"].includes(currentPhase)){const b=box("Hub");s.appendChild(E("text",{x:b[0]+b[2]/2,y:b[1]+b[3]+25,"text-anchor":"middle",fill:"#ff7180","font-size":12,"font-weight":900,"letter-spacing":".12em"},eject.payload.agent==null?"VOTE SKIPPED":"EJECTED "+who(eject.payload.agent)))}const pre=tokenLayout(prePos,preAlive,false),post=tokenLayout(postPos,postAlive,true),ids=new Set([...Object.keys(pre),...Object.keys(post)]);ids.forEach(raw=>{const a=+raw;if(!pre[a]&&!showRoles&&t<.98)return;const p0=pre[a]||post[a],p1=post[a]||p0,from=prePos[a],to=postPos[a];let p;if(currentPhase==="movement"&&from>=0&&to>=0&&from!==to)p=along([[p0.x,p0.y],...routePoints(from,to),[p1.x,p1.y]],t);else p=[p0.x+(p1.x-p0.x)*t,p0.y+(p1.y-p0.y)*t];const alive=currentPhase==="incident"&&moveT>.55?postAlive:["report","meeting","observation","resolved"].includes(currentPhase)?postAlive:preAlive;drawCrew(s,a,p[0],p[1],!alive[a])});if(currentPhase==="observation")phaseRows(f,"observation").forEach(e=>{const room=e.payload&&e.payload.room;if(room==null)return;const c=center(room);s.appendChild(E("circle",{cx:c[0],cy:c[1],r:34,fill:"none",stroke:"#42e8ff","stroke-width":2,"stroke-dasharray":"5 6",opacity:.9}))})}
function drawMeetingNetwork(){const s=document.getElementById("meeting-network-svg");if(!s)return;s.textContent="";const W=760,H=250,cx=W/2,cy=132,rx=285,ry=88,defs=E("defs",{}),marker=E("marker",{id:"meeting-arrow",viewBox:"0 0 10 10",refX:8,refY:5,markerWidth:5,markerHeight:5,orient:"auto-start-reverse"});marker.appendChild(E("path",{d:"M 0 0 L 10 5 L 0 10 z",fill:"#9eb0cf"}));defs.appendChild(marker);s.appendChild(defs);s.appendChild(E("rect",{x:0,y:0,width:W,height:H,rx:12,fill:"#0c172a",stroke:"#263d63"}));s.appendChild(E("text",{x:16,y:20,fill:"#7e93b8","font-size":10,"font-weight":900,"letter-spacing":".14em"},"PUBLIC MEETING NETWORK"));if(!["meeting","resolved"].includes(currentPhase)){s.appendChild(E("text",{x:W/2,y:H/2+8,"text-anchor":"middle",fill:"#60769a","font-size":12},"Network appears when the meeting begins"));return}const pts={};for(let a=0;a<n;a++){const ang=-Math.PI/2+2*Math.PI*a/n;pts[a]={x:cx+rx*Math.cos(ang),y:cy+ry*Math.sin(ang)}}const events=(D.frames[frame].events||[]).filter(e=>e.type==="ResponseMade"||e.type==="VoteCast"),responseColor={ACCUSE:"#ff7180",CONTRADICT:"#ffd166",DEFEND:"#51e0a4",AGREE_WITH:"#51e0a4",QUESTION:"#42e8ff",NEUTRAL:"#9eb0cf"};let edgeIndex=0;events.forEach(e=>{const p=e.payload||{},from=e.type==="ResponseMade"?p.speaker:p.agent,to=e.type==="ResponseMade"?p.target:p.vote;if(from==null||to==null||from<0||to<0||!pts[from]||!pts[to])return;const a=pts[from],b=pts[to],dx=b.x-a.x,dy=b.y-a.y,len=Math.max(1,Math.hypot(dx,dy)),bend=((edgeIndex%3)-1)*18,mx=(a.x+b.x)/2-dy/len*bend,my=(a.y+b.y)/2+dx/len*bend,color=e.type==="VoteCast"?"#ffd166":(responseColor[p.response_type]||"#9eb0cf"),path="M"+a.x+" "+a.y+" Q"+mx+" "+my+" "+b.x+" "+b.y;s.appendChild(E("path",{d:path,fill:"none",stroke:color,"stroke-width":e.type==="VoteCast"?2:2.5,"stroke-dasharray":e.type==="VoteCast"?"6 5":undefined,opacity:.82,"marker-end":"url(#meeting-arrow)"}));edgeIndex++});const ejected=(D.frames[frame].events||[]).find(e=>e.type==="AgentEjected"),ejectedAgent=ejected&&ejected.payload?ejected.payload.agent:null;for(let a=0;a<n;a++){const p=pts[a],coal=showRoles&&role(a)==="COALITION",color=showRoles?(coal?"#ff5e6c":"#4d8dff"):C[a%C.length];s.appendChild(E("circle",{cx:p.x,cy:p.y,r:19,fill:"#0b1224",stroke:a===ejectedAgent?"#ff5e6c":color,"stroke-width":a===ejectedAgent?4:2}));s.appendChild(E("text",{x:p.x,y:p.y+4,"text-anchor":"middle",fill:"#f7fbff","font-size":11,"font-weight":900},who(a)));if(showRoles)s.appendChild(E("text",{x:p.x,y:p.y+33,"text-anchor":"middle",fill:coal?"#ff9da6":"#8fb9ff","font-size":8},coal?"IMPOSTOR":"CREW"))}s.appendChild(E("text",{x:16,y:H-12,fill:"#7186aa","font-size":10},"solid = response   ·   dashed = vote   ·   arrows point to the target"))}
function meetingActionText(e){const p=e.payload||{},clean=x=>String(x||"").replace(/_/g," ").toLowerCase();if(e.type==="ClaimMade"){const parts=[clean(p.claim_type)];if(p.subject!=null)parts.push(who(p.subject));if(p.room!=null)parts.push("in "+(rooms[Number(p.room)]||"unknown room"));if(p.time!=null)parts.push("at turn "+p.time);return parts.join(" · ")}if(e.type==="ResponseMade")return clean(p.response_type)+(p.target!=null?" · "+who(p.target):"");if(e.type==="VoteCast")return "voted for "+(p.vote==null?"skip":who(p.vote));if(e.type==="AgentEjected")return p.agent==null?"vote skipped":"ejected "+who(p.agent);return e.label||e.type}
function drawMeetingActions(){const panel=document.getElementById("meeting-actions");if(!panel)return;panel.textContent="";if(!["meeting","resolved"].includes(currentPhase)){panel.innerHTML='<div class="quiet">Meeting actions appear here once the meeting begins.</div>';return}const types={ClaimMade:["claim","CLAIM"],ResponseMade:["response","RESPONSE"],VoteCast:["vote","VOTE"],AgentEjected:["ejection","RESULT"]},items=(D.frames[frame].events||[]).filter(e=>types[e.type]);if(!items.length){panel.innerHTML='<div class="quiet">No recorded meeting actions.</div>';return}items.forEach(e=>{const d=document.createElement("div"),meta=types[e.type],speaker=e.type==="ClaimMade"?e.payload.speaker:e.type==="ResponseMade"?e.payload.speaker:e.type==="VoteCast"?e.payload.agent:null;d.className="meeting-action "+meta[0];const tag=document.createElement("div");tag.className="action-tag";tag.textContent=meta[1]+(speaker!=null?" · "+who(speaker):"");const body=document.createElement("div");body.className="action-body";body.textContent=meetingActionText(e);d.append(tag,body);panel.appendChild(d)})}
function draw(){drawMap();drawMoment();drawFeed();drawMeetingNetwork();drawMeetingActions();drawStatus();drawScrub();drawOutcome();document.querySelectorAll(".round").forEach((x,k)=>x.classList.toggle("active",D.frames[frame].round===k))}
let phaseIndex=0;
function phaseList(){return phaseSchedule(D.frames[frame])}
function cancelPhaseAnimation(){if(phaseTimer!==null){cancelAnimationFrame(phaseTimer);phaseTimer=null}}
function pausePlayback(){if(timer!==null){clearTimeout(timer);timer=null}cancelPhaseAnimation();document.getElementById("play").textContent="▶ Play";document.getElementById("play2").textContent="▶"}
function showPhase(animateMovement){const schedule=phaseList();if(!schedule.length)return;const phase=schedule[Math.max(0,Math.min(phaseIndex,schedule.length-1))];currentPhase=phase.name;showAfter=["observation","meeting","resolved"].includes(currentPhase);if(currentPhase==="movement"&&animateMovement){moveT=0;draw();const started=performance.now();const run=now=>{moveT=Math.min(1,(now-started)/phase.ms);drawMap();drawMoment();drawFeed();drawStatus();if(moveT>=1){phaseTimer=null;return}phaseTimer=requestAnimationFrame(run)};phaseTimer=requestAnimationFrame(run)}else{moveT=currentPhase==="turn_start"?0:1;draw()}}
function drawStatus(){const f=D.frames[frame],pc=phaseTitle(f),schedule=phaseList();document.getElementById("meta").textContent=D.config.name+" · "+n+" players · "+D.meetings.length+" recorded rounds";document.getElementById("step").textContent="ROUND "+(f.round+1);document.getElementById("phase").textContent=pc[0].toLowerCase();document.getElementById("time").innerHTML="Round <strong>"+(f.round+1)+"</strong> · turn <strong>"+f.turn+"</strong> · "+pc[0];document.getElementById("count").textContent="frame "+(frame+1)+" / "+D.frames.length+" · step "+(phaseIndex+1)+" / "+schedule.length}
function animateFrame(){phaseIndex=0;showPhase(false)}
function stepPhase(delta,fromPlayback){if(!fromPlayback)pausePlayback();let schedule=phaseList();if(delta>0){if(phaseIndex<schedule.length-1)phaseIndex+=1;else{frame=(frame+1)%D.frames.length;phaseIndex=0}}else{if(phaseIndex>0)phaseIndex-=1;else{frame=(frame-1+D.frames.length)%D.frames.length;phaseIndex=phaseList().length-1}}schedule=phaseList();const target=schedule[phaseIndex];showPhase(delta>0&&target.name==="movement")}
function play(btn){if(timer!==null){pausePlayback();return}const tick=()=>{if(timer===null)return;stepPhase(1,true);const schedule=phaseList(),phase=schedule[phaseIndex];timer=setTimeout(tick,phase.ms+360)};timer=setTimeout(tick,420);document.getElementById("play").textContent="Ⅱ Pause";document.getElementById("play2").textContent="Ⅱ"}'''
    controls += r'''
function drawMainMeetingNetwork(s){const W=920,H=610,cx=460,cy=315,rx=330,ry=210,defs=E("defs",{}),marker=E("marker",{id:"meeting-arrow-main",viewBox:"0 0 10 10",refX:8,refY:5,markerWidth:6,markerHeight:6,orient:"auto-start-reverse"});marker.appendChild(E("path",{d:"M 0 0 L 10 5 L 0 10 z",fill:"#9eb0cf"}));defs.appendChild(marker);s.appendChild(defs);s.appendChild(E("rect",{x:0,y:0,width:W,height:H,fill:"#0c172a"}));s.appendChild(E("text",{x:40,y:42,fill:"#42e8ff","font-size":18,"font-weight":900,"letter-spacing":".16em"},"MEETING // PUBLIC INTERACTION MAP"));s.appendChild(E("text",{x:40,y:65,fill:"#7e93b8","font-size":11},"Claims, responses, and votes are shown as directed interactions."));const pts={};for(let a=0;a<n;a++){const ang=-Math.PI/2+2*Math.PI*a/n;pts[a]={x:cx+rx*Math.cos(ang),y:cy+ry*Math.sin(ang)}}const all=D.frames[frame].events||[],events=all.filter(e=>e.type==="ResponseMade"||e.type==="VoteCast"),claimers=new Set(all.filter(e=>e.type==="ClaimMade").map(e=>e.payload&&e.payload.speaker).filter(a=>a!=null)),responseColor={ACCUSE:"#ff7180",CONTRADICT:"#ffd166",DEFEND:"#51e0a4",AGREE_WITH:"#51e0a4",QUESTION:"#42e8ff",NEUTRAL:"#9eb0cf"};let edgeIndex=0;events.forEach(e=>{const p=e.payload||{},from=e.type==="ResponseMade"?p.speaker:p.agent,to=e.type==="ResponseMade"?p.target:p.vote;if(from==null||to==null||from<0||to<0||!pts[from]||!pts[to])return;const a=pts[from],b=pts[to],dx=b.x-a.x,dy=b.y-a.y,len=Math.max(1,Math.hypot(dx,dy)),bend=((edgeIndex%3)-1)*25,mx=(a.x+b.x)/2-dy/len*bend,my=(a.y+b.y)/2+dx/len*bend,color=e.type==="VoteCast"?"#ffd166":(responseColor[p.response_type]||"#9eb0cf");s.appendChild(E("path",{d:"M"+a.x+" "+a.y+" Q"+mx+" "+my+" "+b.x+" "+b.y,fill:"none",stroke:color,"stroke-width":e.type==="VoteCast"?3:4,"stroke-dasharray":e.type==="VoteCast"?"10 8":undefined,opacity:.82,"marker-end":"url(#meeting-arrow-main)"}));edgeIndex++});const ejected=all.find(e=>e.type==="AgentEjected"),ejectedAgent=ejected&&ejected.payload?ejected.payload.agent:null;s.appendChild(E("circle",{cx,cy,r:58,fill:"#17335a",stroke:"#42e8ff","stroke-width":2,filter:"url(#softGlow)"}));s.appendChild(E("text",{x:cx,y:cy-4,"text-anchor":"middle",fill:"#e8f7ff","font-size":15,"font-weight":900,"letter-spacing":".12em"},"MEETING"));s.appendChild(E("text",{x:cx,y:cy+17,"text-anchor":"middle",fill:"#7e93b8","font-size":10},"public transcript"));for(let a=0;a<n;a++){const p=pts[a],coal=showRoles&&role(a)==="COALITION",color=showRoles?(coal?"#ff5e6c":"#4d8dff"):C[a%C.length];if(claimers.has(a))s.appendChild(E("circle",{cx:p.x,cy:p.y,r:34,fill:"none",stroke:"#a88cff","stroke-width":2,"stroke-dasharray":"5 6",opacity:.9}));s.appendChild(E("circle",{cx:p.x,cy:p.y,r:27,fill:"#0b1224",stroke:a===ejectedAgent?"#ff5e6c":color,"stroke-width":a===ejectedAgent?5:3}));s.appendChild(E("text",{x:p.x,y:p.y+5,"text-anchor":"middle",fill:"#f7fbff","font-size":14,"font-weight":900},who(a)));if(showRoles)s.appendChild(E("text",{x:p.x,y:p.y+48,"text-anchor":"middle",fill:coal?"#ff9da6":"#8fb9ff","font-size":10},coal?"IMPOSTOR":"CREW"))}s.appendChild(E("text",{x:40,y:H-24,fill:"#a88cff","font-size":11},"ring = made a claim"));s.appendChild(E("text",{x:200,y:H-24,fill:"#51e0a4","font-size":11},"solid = response"));s.appendChild(E("text",{x:360,y:H-24,fill:"#ffd166","font-size":11},"dashed = vote"));s.appendChild(E("text",{x:520,y:H-24,fill:"#7186aa","font-size":11},"arrows point to the target"))}
const roomMapRenderer=drawMap;
function drawMeetingEdgesOnMap(s){const f=D.frames[frame],positions=f.positions_after||f.positions,alive=f.alive_after||f.alive,layout=tokenLayout(positions,alive,true),all=f.events||[],events=all.filter(e=>e.type==="ResponseMade"||e.type==="VoteCast"),responseColor={ACCUSE:"#ff7180",CONTRADICT:"#ffd166",DEFEND:"#51e0a4",AGREE_WITH:"#51e0a4",QUESTION:"#42e8ff",NEUTRAL:"#9eb0cf"},defs=E("defs",{}),marker=E("marker",{id:"meeting-arrow-map",viewBox:"0 0 10 10",refX:8,refY:5,markerWidth:5,markerHeight:5,orient:"auto-start-reverse"}),group=E("g",{id:"meeting-edges",opacity:.84});marker.appendChild(E("path",{d:"M 0 0 L 10 5 L 0 10 z",fill:"#9eb0cf"}));defs.appendChild(marker);s.appendChild(defs);let edgeIndex=0;events.forEach(e=>{const p=e.payload||{},from=e.type==="ResponseMade"?p.speaker:p.agent,to=e.type==="ResponseMade"?p.target:p.vote;if(from==null||to==null||from<0||to<0||!layout[from]||!layout[to])return;const a=layout[from],b=layout[to],dx=b.x-a.x,dy=b.y-a.y,len=Math.max(1,Math.hypot(dx,dy)),bend=((edgeIndex%3)-1)*12,mx=(a.x+b.x)/2-dy/len*bend,my=(a.y+b.y)/2+dx/len*bend,color=e.type==="VoteCast"?"#ffd166":(responseColor[p.response_type]||"#9eb0cf");group.appendChild(E("path",{d:"M"+a.x+" "+a.y+" Q"+mx+" "+my+" "+b.x+" "+b.y,fill:"none",stroke:color,"stroke-width":e.type==="VoteCast"?2.5:3,"stroke-dasharray":e.type==="VoteCast"?"7 5":undefined,"marker-end":"url(#meeting-arrow-map)"}));edgeIndex++});const firstToken=[...s.children].find(x=>x.getAttribute&&x.getAttribute("transform"));if(firstToken)s.insertBefore(group,firstToken);else s.appendChild(group);s.appendChild(E("text",{x:460,y:34,"text-anchor":"middle",fill:"#d9fbff","font-size":13,"font-weight":900,"letter-spacing":".12em"},"MEETING INTERACTIONS"))}
drawMap=function(){roomMapRenderer();if(["meeting","resolved"].includes(currentPhase))drawMeetingEdgesOnMap(document.getElementById("map"))};
const drawMeetingEdgesBase=drawMeetingEdgesOnMap;
drawMeetingEdgesOnMap=function(s){drawMeetingEdgesBase(s);const f=D.frames[frame],positions=f.positions_after||f.positions,alive=f.alive_after||f.alive,layout=tokenLayout(positions,alive,true),all=f.events||[],events=all.filter(e=>e.type==="ResponseMade"||e.type==="VoteCast"),responseNames={ACCUSE:"ACCUSE",CONTRADICT:"CONTRADICT",DEFEND:"DEFEND",AGREE_WITH:"AGREE",QUESTION:"QUESTION",NEUTRAL:"NEUTRAL"},labels=E("g",{id:"meeting-edge-labels","pointer-events":"none"});let edgeIndex=0;events.forEach(e=>{const p=e.payload||{},from=e.type==="ResponseMade"?p.speaker:p.agent,to=e.type==="ResponseMade"?p.target:p.vote;if(from==null||to==null||from<0||to<0||!layout[from]||!layout[to])return;const a=layout[from],b=layout[to],dx=b.x-a.x,dy=b.y-a.y,len=Math.max(1,Math.hypot(dx,dy)),bend=((edgeIndex%3)-1)*12,mx=(a.x+b.x)/2-dy/len*bend,my=(a.y+b.y)/2+dx/len*bend,label=e.type==="VoteCast"?"VOTE":(responseNames[p.response_type]||"RESPONSE"),color=e.type==="VoteCast"?"#ffd166":"#b9c8e3",width=Math.max(34,label.length*6+12);labels.appendChild(E("rect",{x:mx-width/2,y:my-8,width,height:15,rx:7,fill:"#081222",stroke:color,"stroke-width":1,opacity:.94}));labels.appendChild(E("text",{x:mx,y:my+3,"text-anchor":"middle",fill:color,"font-size":8,"font-weight":900,"letter-spacing":".04em"},label));edgeIndex++});const firstToken=[...s.children].find(x=>x.getAttribute&&x.getAttribute("transform"));if(firstToken)s.insertBefore(labels,firstToken);else s.appendChild(labels)};
const drawMeetingEdgesWithLabels=drawMeetingEdgesOnMap;
drawMeetingEdgesOnMap=function(s){drawMeetingEdgesWithLabels(s);const f=D.frames[frame],positions=f.positions_after||f.positions,alive=f.alive_after||f.alive,layout=tokenLayout(positions,alive,true),events=meetingEdgeEvents(),responseColor={ACCUSE:"#ff7180",CONTRADICT:"#ffd166",DEFEND:"#51e0a4",AGREE_WITH:"#51e0a4",QUESTION:"#42e8ff",NEUTRAL:"#b9c8e3"},heads=E("g",{id:"meeting-arrowheads","pointer-events":"none"});let edgeIndex=0;events.forEach(e=>{const p=e.payload||{},from=e.type==="ResponseMade"?p.speaker:p.agent,to=e.type==="ResponseMade"?p.target:p.vote;if(from==null||to==null||from<0||to<0||!layout[from]||!layout[to])return;const a=layout[from],b=layout[to],dx=b.x-a.x,dy=b.y-a.y,len=Math.max(1,Math.hypot(dx,dy)),bend=((edgeIndex%3)-1)*12,mx=(a.x+b.x)/2-dy/len*bend,my=(a.y+b.y)/2+dx/len*bend,angle=Math.atan2(b.y-my,b.x-mx),tipX=b.x-Math.cos(angle)*23,tipY=b.y-Math.sin(angle)*23,backX=tipX-Math.cos(angle)*11,backY=tipY-Math.sin(angle)*11,wing=7,color=e.type==="VoteCast"?"#ffd166":(responseColor[p.response_type]||"#b9c8e3"),points=tipX+","+tipY+" "+(backX+Math.sin(angle)*wing)+","+(backY-Math.cos(angle)*wing)+" "+(backX-Math.sin(angle)*wing)+","+(backY+Math.cos(angle)*wing);heads.appendChild(E("polygon",{points,fill:color,opacity:.98}));edgeIndex++});const firstToken=[...s.children].find(x=>x.getAttribute&&x.getAttribute("transform"));if(firstToken)s.insertBefore(heads,firstToken);else s.appendChild(heads)};
'''
    controls = r'''function meetingEdgeEvents(){const events=D.frames[frame].events||[];if(currentPhase==="meeting")return events.filter(e=>e.type==="ResponseMade");if(currentPhase==="resolved")return events.filter(e=>e.type==="VoteCast");return []}
function meetingVisibleAction(e){return currentPhase==="meeting"?["ClaimMade","ResponseMade"].includes(e.type):currentPhase==="resolved"?["VoteCast","AgentEjected"].includes(e.type):false}
''' + controls
    controls = controls.replace('const events=(D.frames[frame].events||[]).filter(e=>e.type==="ResponseMade"||e.type==="VoteCast")', 'const events=meetingEdgeEvents()')
    controls = controls.replace('events=all.filter(e=>e.type==="ResponseMade"||e.type==="VoteCast")', 'events=meetingEdgeEvents()')
    controls = controls.replace('items=(D.frames[frame].events||[]).filter(e=>types[e.type])', 'items=(D.frames[frame].events||[]).filter(e=>types[e.type]&&meetingVisibleAction(e))')
    # Report and meeting phases display the post-movement positions. Without
    # this, the replay briefly jumps back to pre-movement positions when a
    # report is called, which looks like a teleport before the meeting.
    phase_view = 'currentPhase==="movement"?moveT:["observation","resolved","meeting"].includes(currentPhase)?1:0'
    phase_view_fixed = 'currentPhase==="movement"?moveT:["report","observation","resolved","meeting"].includes(currentPhase)?1:0'
    helper = helper.replace(phase_view, phase_view_fixed)
    controls = controls.replace(phase_view, phase_view_fixed)
    html = html.replace('document.getElementById("roles").onclick=toggle;', helper + '\n' + controls + '\ndocument.getElementById("roles").onclick=toggle;')
    html = html.replace("frame=(frame-1+D.frames.length)%D.frames.length;moveT=0;showAfter=false;draw()", "frame=(frame-1+D.frames.length)%D.frames.length;animateFrame()")
    html = html.replace("frame=(frame+1)%D.frames.length;moveT=0;showAfter=false;draw()", "frame=(frame+1)%D.frames.length;animateFrame()")
    html = html.replace("frame=k;moveT=0;showAfter=false;draw()", "frame=k;animateFrame()")
    html = html.replace("frame=Math.max(0,k);moveT=0;showAfter=false;draw()", "frame=Math.max(0,k);animateFrame()")
    path.write_text(html)


def choose_replay(path: str | None) -> Path:
    if path:
        return Path(path)
    return Path(DEFAULT_REPLAY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=None, help="saved .json or .json.gz multi-round replay")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    limit_threads(args.threads)

    replay = choose_replay(args.replay)
    if not replay.exists():
        raise FileNotFoundError(replay)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace = build_trace(replay)
    trace_path = out / "f4_full_game_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2))
    plot_static(trace, out / "fig_f4_full_game_privileged.png", privileged=True)
    plot_static(trace, out / "fig_f4_full_game_public.png", privileged=False)
    write_html(trace, out / "replay_f4_full_game.html")
    provenance = {
        "script": "plot_f4_full_game.py",
        "replay": str(replay),
        "frames": len(trace["frames"]),
        "meetings": len(trace["meetings"]),
        "outcome": trace["outcome"],
        "artifacts": [
            "f4_full_game_trace.json",
            "fig_f4_full_game_privileged.png",
            "fig_f4_full_game_public.png",
            "replay_f4_full_game.html",
        ],
    }
    (out / "f4_full_game_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
