#!/usr/bin/env python3
"""Execute the separately frozen 30-job mechanism supplement, with two workers."""
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    out = HERE / 'mechanism_results'
    logs = out / '_logs'
    logs.mkdir(parents=True, exist_ok=True)
    lock = out / 'queue.lock'
    with lock.open('x') as stream:
        json.dump(dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat()), stream)
    jobs = [(crew, seed) for crew in (3, 5, 7) for seed in range(10)]
    results = []

    def run(job):
        crew, seed = job
        folder = out / f'mechanism_crew{crew}_s{seed}'
        if (folder / 'complete.json').exists():
            return dict(crew=crew, replicate=seed, returncode=0, existing=True)
        cmd = [sys.executable, '-u', '-B', str(HERE / 'replicate_mechanisms.py'),
               'run', '--crew', str(crew), '--replicate', str(seed)]
        with (logs / f'mechanism_crew{crew}_s{seed}.log').open('a') as stream:
            code = subprocess.run(cmd, cwd=HERE.parents[1], stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        return dict(crew=crew, replicate=seed, returncode=code, existing=False)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                payload = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                               total=30, finished=len(results), results=results)
                temp = out / 'queue_status.json.tmp'
                temp.write_text(json.dumps(payload, indent=2) + '\n')
                temp.replace(out / 'queue_status.json')
                print(result, flush=True)
        if any(row['returncode'] for row in results):
            return 1
        # The analysis independently verifies every completion record and raw file.
        return subprocess.run([sys.executable, '-B', str(HERE / 'replicate_mechanisms.py'),
                               'analyze'], cwd=HERE.parents[1], check=False).returncode
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
