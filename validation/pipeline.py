"""Run the frozen development -> selection -> final-test -> analysis sequence.

The process can run as a desktop scheduled task after SSH disconnects. There is
no result-dependent seed expansion, budget change, or checkpoint selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--slots", type=int, default=12)
    args = ap.parse_args()
    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    logs = results / "_logs"
    logs.mkdir(exist_ok=True)
    protocol = args.protocol.resolve()
    selection = results / "selection.json"
    state_file = results / "pipeline_status.json"
    state = dict(protocol=str(protocol), results=str(results), slots=args.slots,
                 pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                 status="running", completed_steps=[])

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = state_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_file)

    def run_step(name, arguments):
        state["step"] = name
        save()
        command = [sys.executable, "-u", "-B", *map(str, arguments)]
        with (logs / f"pipeline_{name}.log").open("a", encoding="utf-8") as out:
            out.write("\n" + json.dumps(dict(started_utc=state["updated_utc"], command=command)) + "\n")
            out.flush()
            subprocess.run(command, cwd=HERE.parent, stdout=out, stderr=subprocess.STDOUT, check=True)
        state["completed_steps"].append(name)
        save()

    try:
        run_step("development", [HERE / "runner.py", "run", "--protocol", protocol,
                 "--results", results, "--phase", "development", "--slots", args.slots])
        run_step("selection", [HERE / "selector.py", "--protocol", protocol,
                 "--results", results, "--out", selection])
        run_step("final", [HERE / "runner.py", "run", "--protocol", protocol,
                 "--results", results, "--phase", "final", "--slots", args.slots,
                 "--selection", selection])
        run_step("analysis", [HERE / "analyze.py", "--protocol", protocol,
                 "--results", results, "--selection", selection, "--out", results / "analysis"])
        state["status"] = "complete"
        state["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        return 0
    except BaseException as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
