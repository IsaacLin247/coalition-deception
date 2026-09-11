#!/usr/bin/env python3
"""Verify the documented bridge between original runs and portable study code.

This checks source and protocol identity, not the completeness of result data.
Run analyze_submission.py separately to validate original or reproduced results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORIGINAL_SOURCE = "691f7687435424185064b24a9fee1832c6e7b32b9661efe20a10e35ca0c17d35"
PORTABLE_SOURCE = "cdaa1c031b4e2e6e690771c9a884ed70ed78653814863f99bb2fe0def367ae68"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root=REPO):
    root = Path(root)
    packaging = json.loads((root / "provenance/packaging.json").read_text())
    protocols = {}
    for kind, name, expected_source in (
        ("original", "provenance/original_study_protocol.json", ORIGINAL_SOURCE),
        ("portable", "submission_protocol.json", PORTABLE_SOURCE),
    ):
        path = root / name
        if sha256(path) != packaging[f"{kind}_protocol_sha256"]:
            raise ValueError(f"{kind} protocol differs from the packaging record")
        protocol = json.loads(path.read_text())
        manifest_hash = hashlib.sha256(json.dumps(protocol["sources"], sort_keys=True).encode()).hexdigest()
        if not (protocol["source_sha256"] == manifest_hash == expected_source
                == packaging[f"{kind}_source_sha256"]):
            raise ValueError(f"{kind} source fingerprint differs from the frozen inventory")
        protocols[kind] = protocol

    original, portable = protocols["original"], protocols["portable"]
    if original["jobs"] != portable["jobs"] or len(portable["jobs"]) != 240:
        raise ValueError("Original and portable job designs must match exactly")
    original_sources, portable_sources = original["sources"], portable["sources"]
    omitted = sorted(set(original_sources) - set(portable_sources))
    added = sorted(set(portable_sources) - set(original_sources))
    changed = sorted(name for name in set(original_sources) & set(portable_sources)
                     if original_sources[name] != portable_sources[name])
    if (omitted != sorted(packaging["source_paths_omitted"]) or added
            or added != packaging["new_source_paths"]
            or changed != ["scripts/plot_f4_full_game.py"]
            or changed != packaging["shared_paths_changed"]):
        raise ValueError("Scientific source differences exceed the documented packaging changes")

    paths = [root / "pyproject.toml"]
    for directory in ("src", "scripts", "configs", "experiments"):
        paths.extend(path for path in (root / directory).rglob("*")
                     if path.is_file() and path.suffix in (".py", ".yaml", ".yml")
                     and "__pycache__" not in path.parts)
    actual = {path.relative_to(root).as_posix(): sha256(path) for path in sorted(paths)}
    if actual != portable_sources:
        raise ValueError("Portable source files differ from the exact frozen inventory")
    return dict(status="verified", original_source_sha256=ORIGINAL_SOURCE,
                portable_source_sha256=PORTABLE_SOURCE, jobs=240,
                unchanged_shared_scientific_files=len(portable_sources) - len(changed),
                changed_presentation_files=changed, omitted_source_files=omitted,
                note="Use the original protocol for original runs and the portable protocol for new runs; never relabel results.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
