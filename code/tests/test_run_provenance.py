"""A source fingerprint must identify real research inputs, including dirty edits."""
from social_collusion import runmeta


def test_actual_repository_root_covers_the_training_implementation():
    manifest=runmeta.source_manifest()
    assert 'src/social_collusion/rl/train.py' in manifest
    assert 'configs/env/full_short_game.yaml' in manifest
    assert 'experiments/submission/run_study.py' in manifest
    assert 'experiments/remote/remote.env' not in manifest


def test_fingerprint_changes_with_uncommitted_source(tmp_path, monkeypatch):
    (tmp_path/'pyproject.toml').write_text('[project]\nname="test"\n')
    (tmp_path/'src').mkdir()
    source=tmp_path/'src'/'model.py'
    source.write_text('def model(): return 1\n')
    monkeypatch.setattr(runmeta,'REPO_ROOT',tmp_path)
    before=runmeta.source_digest()
    source.write_text('def model(): return 2\n')
    assert runmeta.source_digest()!=before
