import hashlib
import json
import zipfile

import pytest

from validation.remote_worker import safe_relative, unpack


@pytest.mark.parametrize("name", ["../secret", "/absolute", "C:/drive", "a\\b", "a//b", "./x"])
def test_refuses_unconfined_paths(name):
    with pytest.raises(ValueError):
        safe_relative(name)


def make_bundle(path, payload=b"pinned source\n", declared_payload=None):
    declared_payload = payload if declared_payload is None else declared_payload
    manifest = {"files": {"code/src/example.py": {
        "bytes": len(declared_payload), "sha256": hashlib.sha256(declared_payload).hexdigest()}}}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("code/src/example.py", payload)
        z.writestr("bundle_manifest.json", json.dumps(manifest))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checks_archive_and_members_and_refuses_replacement(tmp_path):
    archive = tmp_path / "bundle.zip"
    digest = make_bundle(archive)
    destination = tmp_path / "source"
    result = unpack(archive, destination, digest)
    assert result["files"] == 1
    assert (destination / "code/src/example.py").read_bytes() == b"pinned source\n"
    with pytest.raises(ValueError, match="existing"):
        unpack(archive, destination, digest)
    with pytest.raises(ValueError, match="checksum"):
        unpack(archive, tmp_path / "other", "0" * 64)
    altered = tmp_path / "altered.zip"
    altered_hash = make_bundle(altered, b"tampered", b"expected")
    with pytest.raises(ValueError, match="checksum"):
        unpack(altered, tmp_path / "altered_source", altered_hash)
