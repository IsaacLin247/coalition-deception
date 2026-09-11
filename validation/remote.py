"""Bundle and inspect validation work over an existing OpenSSH host alias.

Run registration/launch only for a user-authorized remote computation. Commands
use encoded PowerShell and explicit argument quoting rather than shell data
interpolation. Machine-specific paths and receipts stay outside the repository.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def ssh(host: str, script: str, timeout: int = 60) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", host):
        raise ValueError("Use an SSH host alias without shell syntax")
    script = "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; " + script
    encoded = base64.b64encode(script.encode("utf-16le")).decode()
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "--", host,
                             "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand " + encoded],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[:4000] or result.stdout[:4000])
    return result.stdout.lstrip("\ufeff").strip()


def remote_python(host: str, python: str, arguments: list[str], timeout: int = 60) -> str:
    command = "& " + " ".join(ps_quote(x) for x in [python, "-B", *arguments])
    return ssh(host, command + "; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }", timeout)


def bundle(output: Path) -> dict:
    if output.exists():
        raise ValueError("Refusing to overwrite a prepared bundle")
    paths = []
    for folder in (ROOT / "code", ROOT / "validation"):
        for p in sorted(folder.rglob("*")):
            if any(x in (".venv", "__pycache__", ".pytest_cache", ".hypothesis", "results")
                   or x.endswith(".egg-info") for x in p.parts):
                continue
            if p.is_symlink():
                raise ValueError("Symlink found in source inventory")
            if p.is_file() and (p.suffix in (".py", ".yaml", ".yml", ".toml", ".json", ".md")
                                or p.name == "LICENSE"):
                paths.append(p)
    inputs = json.loads((ROOT / "validation/inputs/development_attacks.json").read_text())
    for attack in inputs["attacks"]:
        p = ROOT / attack["path"]
        if not p.resolve().is_relative_to(ROOT) or p.is_symlink():
            raise ValueError("Development input escaped source workspace")
        if hashlib.sha256(p.read_bytes()).hexdigest() != attack["sha256"]:
            raise ValueError("Development checkpoint changed")
        paths.append(p)
    inventory = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(set(paths)):
            raw = p.read_bytes()
            name = p.relative_to(ROOT).as_posix()
            inventory[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            z.writestr(name, raw)
        z.writestr("bundle_manifest.json", json.dumps(dict(files=inventory), indent=2, sort_keys=True))
    return dict(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                bytes=output.stat().st_size, files=len(inventory))


def upload(host: str, local: Path, remote: str) -> None:
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "--",
                    str(local), host + ":" + remote], check=True, timeout=120)


def launch(host: str, base: str, python: str, task: str, principal_task: str,
           protocol: str, slots: int) -> dict:
    """Start the already prepared pipeline as a task that survives SSH exit."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task) or not re.fullmatch(r"[A-Za-z0-9_-]+", principal_task):
        raise ValueError("Use a simple scheduled-task name")
    if not 1 <= slots <= 128:
        raise ValueError("Invalid worker count")
    for path in (base, python, protocol):
        if not re.fullmatch(r"[A-Za-z0-9_./:\\-]+", path):
            raise ValueError("Task wrapper requires simple paths without shell metacharacters")
    arguments = [python, "-u", "-B", base + "/source/validation/pipeline.py",
                 "--protocol", base + "/source/" + protocol,
                 "--results", base + "/results", "--slots", str(slots)]
    wrapper = "\r\n".join(["@echo off", "set OMP_NUM_THREADS=1", "set MKL_NUM_THREADS=1",
        "set OPENBLAS_NUM_THREADS=1", "set PYTHONUNBUFFERED=1", "set PYTHONDONTWRITEBYTECODE=1",
        "set PYTHONUTF8=1", "cd /d " + base.replace("/", "\\") + "\\source",
        " ".join('"' + a + '"' for a in arguments) + ' > "' + base + '/pipeline.log" 2>&1',
        "exit /b %ERRORLEVEL%", ""])
    ssh(host, f"if (Get-ScheduledTask -TaskName {ps_quote(task)} -ErrorAction SilentlyContinue) "
              "{ throw 'Task already exists; inspect before reuse' }")
    with tempfile.TemporaryDirectory(prefix="validation-launch-") as temporary:
        path = Path(temporary) / "run_validation.cmd"
        path.write_bytes(wrapper.encode("ascii"))
        upload(host, path, base + "/run_validation.cmd")
    wrapper_hash = hashlib.sha256(wrapper.encode("ascii")).hexdigest()
    command_path = (base + "/run_validation.cmd").replace("/", "\\")
    script = "; ".join([
        f"if ((Get-FileHash -Algorithm SHA256 {ps_quote(base + '/run_validation.cmd')}).Hash.ToLower() "
        f"-ne {ps_quote(wrapper_hash)}) {{ throw 'Task wrapper hash mismatch' }}",
        f"$principal=(Get-ScheduledTask -TaskName {ps_quote(principal_task)}).Principal",
        "$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) "
        "-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew",
        "$action=New-ScheduledTaskAction -Execute 'cmd.exe' "
        f"-Argument {ps_quote('/d /c ' + command_path)} -WorkingDirectory {ps_quote(base + '/source')}",
        "$definition=New-ScheduledTask -Action $action -Principal $principal -Settings $settings "
        "-Description 'Frozen development and independent validation pipeline'",
        f"Register-ScheduledTask -TaskName {ps_quote(task)} -InputObject $definition | Out-Null",
        f"Start-ScheduledTask -TaskName {ps_quote(task)}",
        f"Get-ScheduledTask -TaskName {ps_quote(task)} | Select-Object TaskName,State | ConvertTo-Json",
    ])
    return dict(task=json.loads(ssh(host, script)), command=arguments, wrapper_sha256=wrapper_hash,
                host=host, source=base + "/source", results=base + "/results", slots=slots)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("bundle", "deploy", "environment", "status", "launch", "progress"))
    ap.add_argument("--host", default="desktop")
    ap.add_argument("--base")
    ap.add_argument("--python")
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--protocol", default="validation/protocol.json")
    ap.add_argument("--phase", default="development")
    ap.add_argument("--task")
    ap.add_argument("--principal-task")
    ap.add_argument("--slots", type=int, default=12)
    args = ap.parse_args()
    if args.action == "bundle":
        result = bundle(args.archive)
    elif args.action == "deploy":
        if not args.base or not args.python or not args.archive:
            ap.error("deploy needs --base, --python and --archive")
        ssh(args.host, f"if (Test-Path {ps_quote(args.base)}) {{ throw 'Remote workspace already exists' }}; "
                       f"[IO.Directory]::CreateDirectory({ps_quote(args.base)}) | Out-Null")
        helper = ROOT / "validation/remote_worker.py"
        upload(args.host, helper, args.base + "/transport.py")
        helper_hash = hashlib.sha256(helper.read_bytes()).hexdigest()
        ssh(args.host, f"if ((Get-FileHash -Algorithm SHA256 {ps_quote(args.base + '/transport.py')}).Hash.ToLower() "
                       f"-ne {ps_quote(helper_hash)}) {{ throw 'Transport helper hash mismatch' }}")
        upload(args.host, args.archive, args.base + "/source.zip")
        result = json.loads(remote_python(args.host, args.python,
            [args.base + "/transport.py", "unpack", "--archive", args.base + "/source.zip",
             "--destination", args.base + "/source", "--sha256",
             hashlib.sha256(args.archive.read_bytes()).hexdigest()]))
    elif args.action == "environment":
        result = json.loads(remote_python(args.host, args.python,
                            [args.base + "/transport.py", "environment"]))
    elif args.action == "launch":
        result = launch(args.host, args.base, args.python, args.task, args.principal_task,
                        args.protocol, args.slots)
    elif args.action == "progress":
        result = json.loads(remote_python(args.host, args.python,
            [args.base + "/transport.py", "progress", "--directory", args.base + "/results"]))
    else:
        arguments = (
            [args.base + "/source/validation/runner.py", "status", "--protocol",
             args.base + "/source/" + args.protocol, "--results", args.base + "/results",
             "--phase", args.phase])
        if args.phase == "final":
            arguments += ["--selection", args.base + "/results/selection.json"]
        result = json.loads(remote_python(args.host, args.python, arguments))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
