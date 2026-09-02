"""Classifier unit tests for the conflict gate against real git fixtures.

Covers conflict_gate.evaluate (pure classifier) plus read-only / error / timeout
behaviour. Integration with the dispatcher is covered in
tests/plugins/test_kanban_conflict_gate.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli.conflict_gate import (
    PARALLEL_SAFE,
    SERIALIZE,
    UNKNOWN,
    GateInput,
    evaluate,
    ConflictGateError,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "dev")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "a.txt").write_text("BASE\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "base")
    _git(r, "branch", "base")
    _git(r, "branch", "origin/dev", "base")

    _git(r, "checkout", "-qb", "w-x")
    (r / "a.txt").write_text("BASE\nX\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "x")

    _git(r, "checkout", "-qb", "w-y", "base")
    (r / "a.txt").write_text("BASE\nY\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "y")

    _git(r, "checkout", "-qb", "w-z", "base")
    (r / "b.txt").write_text("new\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "z")
    _git(r, "checkout", "-q", "dev")
    return r


def test_first_developer_no_active(repo):
    assert evaluate(GateInput(str(repo), "origin/dev", [], "w-y")).cls == PARALLEL_SAFE


def test_second_disjoint_parallel(repo):
    assert evaluate(GateInput(str(repo), "origin/dev", ["w-x"], "w-z")).cls == PARALLEL_SAFE


def test_second_conflict_serialize(repo):
    v = evaluate(GateInput(str(repo), "origin/dev", ["w-x"], "w-y"))
    assert v.cls == SERIALIZE
    assert any("merge-tree" in r for r in v.reasons)


def test_migration_manifest_serialize(repo):
    v = evaluate(GateInput(str(repo), "origin/dev", ["w-x"], "dev", ["supabase/migrations/002.sql"]))
    assert v.cls == SERIALIZE
    assert any("migration" in r for r in v.reasons)


def test_no_commits_no_manifest_unknown(repo):
    assert evaluate(GateInput(str(repo), "origin/dev", ["w-x"], "dev", None)).cls == UNKNOWN


def test_no_commits_manifest_disjoint_parallel(repo):
    assert evaluate(GateInput(str(repo), "origin/dev", ["w-x"], "dev", ["zzz/x.txt"])).cls == PARALLEL_SAFE


def test_no_commits_manifest_overlap_serialize(repo):
    assert evaluate(GateInput(str(repo), "origin/dev", ["w-x"], "dev", ["a.txt"])).cls == SERIALIZE


def test_same_domain_serialize(repo):
    v = evaluate(GateInput(str(repo), "origin/dev", ["w-x"], "w-z", same_domain_running=["t-2"]))
    assert v.cls == SERIALIZE


def test_read_only(repo):
    before = _git(repo, "status", "--porcelain")
    evaluate(GateInput(str(repo), "origin/dev", ["w-x"], "w-y"))
    after = _git(repo, "status", "--porcelain")
    assert before == after
    for b in ("w-x", "w-y", "w-z", "origin/dev"):
        assert _git(repo, "rev-parse", "--verify", b).strip()


def test_missing_repo_safe(repo, tmp_path):
    from hermes_cli.conflict_gate import _merge_tree_has_conflicts
    assert _merge_tree_has_conflicts(str(tmp_path / "missing"), "w-x", "w-y") is True
    v = evaluate(GateInput(str(tmp_path / "missing"), "origin/dev", ["w-x"], "w-y"))
    assert v.cls in (UNKNOWN, SERIALIZE)


def test_bad_base_surfaces_error(repo):
    from hermes_cli.conflict_gate import _branch_has_commits
    with pytest.raises(ConflictGateError):
        _branch_has_commits(str(repo), "origin/does-not-exist", "w-x")


def test_merge_tree_error_degrades_to_unknown(repo, monkeypatch):
    """Regression: a merge-tree error/timeout must degrade to UNKNOWN (never
    propagate and crash the dispatcher). QA found this was previously raised."""
    from hermes_cli import conflict_gate as cg

    def _boom(repo_root, a, b):
        raise ConflictGateError("git timed out: merge-tree")

    monkeypatch.setattr(cg, "_merge_tree_has_conflicts", _boom)
    v = evaluate(GateInput(str(repo), "origin/dev", ["w-x"], "w-y"))
    assert v.cls == UNKNOWN
    assert any("merge-tree" in r for r in v.reasons)
