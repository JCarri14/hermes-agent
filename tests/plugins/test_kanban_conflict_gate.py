"""Integration tests for the dispatcher-side conflict-gate wiring.

Drives ``kanban_db._conflict_gate_should_serialize`` (the exact function the
dispatcher calls before ``claim_task`` when ``kanban.conflict_gate`` is on)
against a real, isolated board DB and real git fixtures.

Unit coverage of ``hermes_cli.conflict_gate.evaluate`` (classifier) lives in
tests/hermes_cli/test_conflict_gate_classifier.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


@pytest.fixture()
def board_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    r = tmp_path / "proj"
    r.mkdir()
    _git(r, "init", "-q", "-b", "dev")
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "a.txt").write_text("BASE\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "base")
    _git(r, "branch", "base")
    # The gate resolves the real base as origin/dev (remote-based deployment);
    # expose an equivalent local ref so the isolated fixture matches production.
    _git(r, "branch", "origin/dev", "base")

    _git(r, "checkout", "-qb", "wt-active")
    (r / "a.txt").write_text("BASE\nACTIVE\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "active touches a.txt")

    _git(r, "checkout", "-qb", "wt-cand-conflict", "base")
    (r / "a.txt").write_text("BASE\nCAND\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "candidate also touches a.txt")

    _git(r, "checkout", "-qb", "wt-cand-disjoint", "base")
    (r / "b.txt").write_text("new\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "candidate adds b.txt")
    return r


def _insert(conn, tid, *, project, branch, status, workspace):
    conn.execute(
        "INSERT INTO tasks (id, title, assignee, status, created_at, workspace_kind, "
        "workspace_path, branch_name, project_id) VALUES (?,?,?,?,?, 'scratch',?,?,?)",
        (tid, tid, "Developer", status, 1, workspace, branch, project),
    )


def _run_gate(board_home, git_repo, active, candidate):
    with kb.connect() as conn:
        _insert(conn, "active", project="p1", branch=active, status="running",
                workspace=str(git_repo))
        _insert(conn, "cand", project="p1", branch=candidate, status="ready",
                workspace=str(git_repo))
        return kb._conflict_gate_should_serialize(conn, "cand", "Developer", board="default")


def test_first_slot_allow(board_home, git_repo):
    # No same-base running card -> gate returns None (allow, normal dispatch).
    with kb.connect() as conn:
        _insert(conn, "cand", project="p1", branch="wt-cand-disjoint", status="ready",
                workspace=str(git_repo))
        assert kb._conflict_gate_should_serialize(conn, "cand", "Developer", board="default") is None


def test_second_disjoint_allow(board_home, git_repo):
    # Active on a.txt; candidate adds b.txt -> no git conflict, no overlap -> allow.
    assert _run_gate(board_home, git_repo, "wt-active", "wt-cand-disjoint") is None


def test_second_conflict_serialize(board_home, git_repo):
    # Active and candidate both modify a.txt -> SERIALIZE tuple.
    out = _run_gate(board_home, git_repo, "wt-active", "wt-cand-conflict")
    assert out is not None
    cls, reason = out
    assert cls == "SERIALIZE"
    assert "merge-tree" in reason or "overlap" in reason


def test_unknown_no_commit_no_manifest(board_home, git_repo):
    # Candidate branch == base/dev (no commits); no target_paths manifest -> UNKNOWN.
    out = _run_gate(board_home, git_repo, "wt-active", "dev")
    assert out is not None
    cls, _ = out
    assert cls == "UNKNOWN"
