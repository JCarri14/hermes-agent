"""Integration tests for the dispatcher-side pre-action destructive gate.

Drives ``kanban_db._destructive_gate_requires_go`` (the function the dispatcher
calls before ``claim_task`` when ``kanban.destructive_gate`` is on), plus a
real ``dispatch_once`` against an isolated board DB, to prove:

  * a ready destructive-live card with NO human GO is NOT claimed/spawned
    (the gate fires BEFORE the destructive action can start); and
  * once a human GO comment is recorded on the card, it IS dispatched.

Top-of-module unit coverage of the classifier lives in
``tests/hermes_cli/test_destructive_gate_classifier.py``.
"""

from __future__ import annotations

import os
import sys

import pytest

# Force-import against the caller's HERMES_HOME-agnostic module.
from hermes_cli import kanban_db as kb


@pytest.fixture()
def board_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _insert(conn, tid, *, title, body="", assignee="default", status="ready"):
    conn.execute(
        "INSERT INTO tasks (id, title, body, assignee, status, created_at, "
        "workspace_kind) VALUES (?,?,?,?,?,?, 'scratch')",
        (tid, title, body, assignee, status, int(__import__("time").time())),
    )


def _insert_go(conn, tid, body, author="human"):
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?,?,?,?)",
        (tid, author, body, int(__import__("time").time())),
    )


def _fake_spawn(*args, **kwargs):
    return 12345


# ─────────────── unit: _destructive_gate_requires_go (helper) ────────────────


def test_destructive_no_go_blocked(board_home):
    with kb.connect() as conn:
        _insert(conn, "destructive", title="Cleanup client bucket",
                body="Delete the R2 bucket erp-client-a-docs.", status="ready")
        out = kb._destructive_gate_requires_go(conn, "destructive", board="default")
    assert out is not None
    cls, reason = out
    assert cls == "DESTRUCTIVE_LIVE"
    assert "NO human GO recorded" in reason


def test_destructive_with_human_go_allowed(board_home):
    with kb.connect() as conn:
        _insert(conn, "destructive", title="Cleanup client bucket",
                body="Delete the R2 bucket erp-client-a-docs.", status="ready")
        _insert_go(conn, "destructive", "@go destructive destructive", author="human")
        out = kb._destructive_gate_requires_go(conn, "destructive", board="default")
    assert out is None  # human GO recorded -> allow


def test_safe_card_allowed(board_home):
    with kb.connect() as conn:
        _insert(conn, "safe", title="Add a new endpoint",
                body="Implements /v1/health.", status="ready")
        out = kb._destructive_gate_requires_go(conn, "safe", board="default")
    assert out is None


def test_executor_self_go_not_counted(board_home):
    # A GO comment authored by the card's assignee (the executor) must NOT be
    # treated as a human GO — a worker cannot self-authorize.
    with kb.connect() as conn:
        _insert(conn, "destructive", title="Drop table",
                body="Drop the live orders table.", assignee="dev", status="ready")
        _insert_go(conn, "destructive", "@go destructive destructive", author="dev")
        out = kb._destructive_gate_requires_go(conn, "destructive", board="default")
    assert out is not None
    cls, _ = out
    assert cls == "DESTRUCTIVE_LIVE"


# ──────────────────── integration: real dispatch_once ────────────────────────


def test_dispatch_blocks_destructive_without_go(board_home):
    """Pre-action proof: with destructive_gate=on, a destructive-live card
    without human GO is NOT spawned (the gate fires BEFORE the action)."""
    with kb.connect() as conn:
        _insert(conn, "destructive", title="Cleanup client bucket",
                body="Delete the R2 bucket erp-client-a-docs.", assignee="default", status="ready")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False, destructive_gate=True,
        )
    assert res.skipped_destructive_gate == [("destructive", "DESTRUCTIVE_LIVE")]
    assert res.spawned == []  # NOT spawned
    # Card stays ready (never claimed / executed).
    with kb.connect() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id='destructive'").fetchone()
    assert row["status"] == "ready"
    # Diagnostic event recorded (guarded by not dry_run).
    with kb.connect() as conn:
        evs = list(conn.execute(
            "SELECT kind FROM task_events WHERE task_id='destructive' AND kind='destructive_gate_held'",
        ))
    assert len(evs) == 1


def test_dispatch_spawns_destructive_with_go(board_home):
    """Once a human GO is recorded, the destructive-live card IS dispatched."""
    with kb.connect() as conn:
        _insert(conn, "destructive", title="Cleanup client bucket",
                body="Delete the R2 bucket erp-client-a-docs.", assignee="default", status="ready")
        _insert_go(conn, "destructive", "@go destructive destructive", author="human")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False, destructive_gate=True,
        )
    assert res.skipped_destructive_gate == []
    assert [s[0] for s in res.spawned] == ["destructive"]
    with kb.connect() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id='destructive'").fetchone()
    assert row["status"] == "running"


def test_gate_off_is_legacy_identical(board_home):
    """Gate OFF (default) -> destructive-live card spawns immediately,
    byte-for-byte as before (no gate interference)."""
    with kb.connect() as conn:
        _insert(conn, "destructive", title="Cleanup client bucket",
                body="Delete the R2 bucket erp-client-a-docs.", assignee="default", status="ready")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False, destructive_gate=False,
        )
    assert res.skipped_destructive_gate == []
    assert [s[0] for s in res.spawned] == ["destructive"]


from pathlib import Path  # noqa: E402  (needed by the board_home fixture)