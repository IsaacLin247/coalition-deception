"""Run provenance: everything sec.21 of the plan asks to record for every run."""

from __future__ import annotations

import json
import hashlib
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def source_manifest() -> dict[str, str]:
    """Hash executable/configuration inputs, including uncommitted repairs."""
    if not (REPO_ROOT / "pyproject.toml").is_file():
        raise RuntimeError(f"Cannot identify research source root: {REPO_ROOT}")
    paths = [REPO_ROOT / "pyproject.toml"]
    for directory in ("src", "scripts", "configs", "experiments"):
        paths.extend(p for p in (REPO_ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".yaml", ".yml")
                     and "__pycache__" not in p.parts)
    return {p.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.exists()}


def source_digest(manifest: dict[str, str] | None = None) -> str:
    manifest = source_manifest() if manifest is None else manifest
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def collect(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "argv": sys.argv,
        "source_sha256": source_digest(),
    }
    for mod in ("numpy", "torch", "pettingzoo", "gymnasium"):
        try:
            m = __import__(mod)
            meta[f"{mod}_version"] = getattr(m, "__version__", "?")
        except ImportError:
            meta[f"{mod}_version"] = None
    try:
        import torch

        meta["cuda_available"] = torch.cuda.is_available()
        meta["cuda_version"] = torch.version.cuda
        meta["device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except ImportError:
        meta["cuda_available"] = False
    if extra:
        meta.update(extra)
    return meta


def write(run_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "runmeta.json"
    with open(path, "w") as fh:
        json.dump(collect(extra), fh, indent=2, sort_keys=True)
    return path
