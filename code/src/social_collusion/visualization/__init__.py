"""Renderers. Every one of them consumes a replay/state and never mutates it (plan sec.14)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from social_collusion.visualization.export_json import episode_to_view, record_to_view, write_view

TEMPLATE = Path(__file__).parent / "web_replay" / "template.html"


def export_html(view: dict[str, Any], path: str | Path) -> Path:
    """Write a **self-contained** HTML replay: no CDN, no sibling files, no server.

    This is the plan's Stage-14 acceptance gate - a replay produced on one machine must open on
    another with neither the training code nor a checkpoint present.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text()
    blob = json.dumps(view).replace("</", "<\\/")  # keep the JSON from closing the script tag
    path.write_text(html.replace("__REPLAY_DATA__", blob))
    return path


__all__ = ["episode_to_view", "record_to_view", "write_view", "export_html", "TEMPLATE"]
