"""The portable distribution must not relabel historical runs or change science."""
import json
import shutil

import pytest

from analysis.verify_packaging import REPO, verify


def test_published_package_preserves_original_job_design_and_scientific_source():
    result = verify()
    assert result["jobs"] == 240
    assert result["unchanged_shared_scientific_files"] == 71
    assert result["original_source_sha256"] != result["portable_source_sha256"]


@pytest.fixture
def package(tmp_path):
    manifest = json.loads((REPO / "submission_protocol.json").read_text())["sources"]
    for name in [*manifest, "submission_protocol.json", "provenance/packaging.json",
                 "provenance/original_study_protocol.json"]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
    return tmp_path


@pytest.mark.parametrize("change", ["source", "added_source", "original_protocol"])
def test_changed_source_or_protocol_cannot_pass_the_packaging_bridge(package, change):
    if change == "source":
        path = package / "src/social_collusion/config.py"
        path.write_text(path.read_text() + "\n# changed\n")
    elif change == "added_source":
        (package / "scripts/unplanned.py").write_text("raise RuntimeError('unplanned')\n")
    else:
        path = package / "provenance/original_study_protocol.json"
        protocol = json.loads(path.read_text())
        protocol["jobs"][0]["name"] += "_changed"
        path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="(inventory|protocol differs)"):
        verify(package)
