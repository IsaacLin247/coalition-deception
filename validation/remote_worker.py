"""Portable transport checks for a separately frozen validation workspace.

This helper performs no research selection. It verifies transferred bytes and
exports completed-job artifacts without editing a running source snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import zipfile


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_relative(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or "\\" in name or ":" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"Unsafe artifact path: {name!r}")
    return p


def unpack(archive: Path, destination: Path, expected_hash: str) -> dict:
    if sha256(archive) != expected_hash:
        raise ValueError("Transferred bundle checksum differs")
    if destination.exists():
        raise ValueError("Refusing to replace an existing source workspace")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entries")
        manifest = json.loads(z.read("bundle_manifest.json"))
        if set(names) != set(manifest["files"]) | {"bundle_manifest.json"}:
            raise ValueError("Bundle inventory differs from manifest")
        for name in names:
            safe_relative(name)
        destination.mkdir(parents=True)
        for name, record in manifest["files"].items():
            raw = z.read(name)
            if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"Bundle member checksum differs: {name}")
            target = destination / safe_relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        (destination / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(source=str(destination), files=len(manifest["files"]),
                archive_sha256=expected_hash, status="unpacked_and_verified")


def environment() -> dict:
    packages = {}
    for name in ("numpy", "torch", "PyYAML", "pandas", "pytest", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, executable=sys.executable,
                platform=platform.platform(), processor=platform.processor(),
                logical_cpus=os.cpu_count(), free_disk_bytes=shutil.disk_usage(Path.cwd()).free,
                packages=packages)


def progress(directory: Path) -> dict:
    """Read operational progress only, without exposing unblinded outcome values."""
    result = {"pipeline": None, "phases": {}, "running_jobs": []}
    pipeline = directory / "pipeline_status.json"
    if pipeline.exists():
        result["pipeline"] = json.loads(pipeline.read_text())
    for phase in ("development", "final"):
        path = directory / "_control" / f"{phase}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        result["phases"][phase] = dict(
            updated_utc=data["updated_utc"], total=data["total"], completed=data["completed"],
            pending=len(data["pending"]), running=len(data["running"]), failed=data["failed"])
        for job in data["running"]:
            folder = directory / safe_relative(job["name"])
            stages = []
            for history in sorted(folder.glob("*/history.json")):
                raw = json.loads(history.read_text())
                rows = raw.get("history", [])
                stages.append(dict(name=history.parent.name, updates=raw.get("updates_completed"),
                    checkpoint_final=(history.parent / "checkpoint_final.pt").is_file(),
                    elapsed_seconds=rows[-1].get("wall_s") if rows else None))
            result["running_jobs"].append(dict(name=job["name"], pid=job["pid"], stages=stages,
                completed_evaluation_cells=len(list(folder.glob("*__*.json")))))
    return result


def export_directory(directory: Path, output: Path) -> dict:
    """Transport a completed tree; research validation runs separately on both hosts."""
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Expected a real result directory")
    if output.exists():
        return dict(path=str(output), sha256=sha256(output), bytes=output.stat().st_size,
                    status="existing_archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise ValueError("An unfinished export already exists")
    manifest = {}
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(directory.rglob("*")):
            if p.is_symlink():
                raise ValueError("Symlink found in result tree")
            if not p.is_file():
                continue
            name = p.relative_to(directory).as_posix()
            safe_relative(name)
            manifest[name] = dict(bytes=p.stat().st_size, sha256=sha256(p))
            z.write(p, name)
        z.writestr("transport_manifest.json", json.dumps(manifest, sort_keys=True, indent=2))
    partial.replace(output)
    return dict(path=str(output), sha256=sha256(output), bytes=output.stat().st_size,
                files=len(manifest), status="exported")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("unpack", "environment", "export", "progress"))
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--destination", type=Path)
    ap.add_argument("--sha256")
    ap.add_argument("--directory", type=Path)
    args = ap.parse_args()
    if args.action == "environment":
        result = environment()
    elif args.action == "progress":
        result = progress(args.directory)
    elif args.action == "unpack":
        result = unpack(args.archive, args.destination, args.sha256)
    else:
        result = export_directory(args.directory, args.archive)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
