"""Tests for the Claude -p kanban executor wrapper (kanban_executor_fallback).

Covers the claude-p attempt contract: run ``claude -p`` against the same card
context, terminate the card only through the real board protocol
(complete_task / block_task), and leave the position (exit code 75) open for
the dispatcher to queue the lower-priority fallback on a QUALIFYING failure.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_executor_fallback as kef


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (mirrors test_kanban_db)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def kanban_env(kanban_home, monkeypatch):
    """HERMES_HOME isolated board + DB env pins for the wrapper process."""
    monkeypatch.setenv("HERMES_HOME", str(kanban_home))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    return kanban_home


def _seed_task(conn, *, title="Build the thing", body="Implement feature X"):
    tid = kb.create_task(conn, title=title, assignee="alice", body=body)
    kb.claim_task(conn, tid, claimer="worker:openai")
    return tid


def _fake_claude_run(monkeypatch, payload: dict, returncode: int = 0):
    """Stub ``subprocess.run`` so the wrapper sees a canned claude -p JSON."""

    def _fake_run(cmd, *args, **kwargs):
        assert cmd[0].endswith("claude") or cmd[0] == "claude"
        return types.SimpleNamespace(
            returncode=returncode,
            stdout=json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(kef.subprocess, "run", _fake_run)


def _task_state(conn, tid):
    row = conn.execute(
        "SELECT status, result, block_kind FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    return {"status": row["status"], "result": row["result"], "block_kind": row["block_kind"]}


def test_claude_success_with_marker_completes_same_card(
    kanban_env, tmp_path, monkeypatch,
):
    """A successful claude -p result carrying the completion marker closes the
    ORIGINAL card as done with the fallback attempt's executor identity."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    payload = {
        "type": "result",
        "subtype": "success",
        "result": "EXECUTOR_FALLBACK_COMPLETE: implemented feature X\nnew file added",
        "modelUsage": {"claude-sonnet-4-6": {"costUSD": 0.01}},
    }
    _fake_claude_run(monkeypatch, payload)

    with kb.connect() as conn:
        tid = _seed_task(conn, title="fallback card", body="Add feature X")
        kef.run_fallback_attempt(tid, str(workspace))

        state = _task_state(conn, tid)
        assert state["status"] == "done"
        assert "implemented feature X" in (state["result"] or "")
        run = conn.execute(
            "SELECT metadata, outcome FROM task_runs WHERE task_id=? "
            "ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        meta = json.loads(run["metadata"] or "{}")
        assert run["outcome"] == "completed"
        assert meta["actual_executor"] == "claude-p"
        assert meta["actual_provider"] == "anthropic"
        assert meta["actual_model"] == "claude-sonnet-4-6"


def test_claude_success_missing_marker_blocks_nonqualifying(
    kanban_env, tmp_path, monkeypatch,
):
    """A claude run that succeeded but never emitted the completion marker is a
    task-side contract violation (the agent narrated instead of delivering) —
    must block, never reroute."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _fake_claude_run(
        monkeypatch,
        {"type": "result", "subtype": "success",
         "result": "I would implement that, but let me stop here."},
    )

    with kb.connect() as conn:
        tid = _seed_task(conn)
        kef.run_fallback_attempt(tid, str(workspace))

        assert _task_state(conn, tid)["status"] == "blocked"
        events = [e.kind for e in kb.list_events(conn, tid)]
        assert "executor_fallback_stopped" in events
        # No queued lower fallback: non-qualifying -> no reroute.
        assert "executor_fallback_queued" not in events


def test_claude_rate_limit_exits_75_leaves_card_reopenable(
    kanban_env, tmp_path, monkeypatch,
):
    """A claude-side rate limit / billing wall is qualifying: exit 75 so the
    dispatcher's reap classifier queues the lower fallback; the card must NOT
    be touched by the wrapper (no complete, no block)."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    payload = {
        "type": "result",
        "subtype": "error",
        "error": {"code": "error_budget", "message": "billing limit reached"},
    }
    _fake_claude_run(monkeypatch, payload, returncode=1)

    with kb.connect() as conn:
        tid = _seed_task(conn)
        exit_code = kef.run_fallback_attempt(tid, str(workspace))
        assert exit_code == kb.KANBAN_RATE_LIMIT_EXIT_CODE
        state = _task_state(conn, tid)
        assert state["status"] == "running"
        events = [e.kind for e in kb.list_events(conn, tid)]
        assert "completed" not in events
        assert "blocked" not in events


def test_claude_max_turns_blocks_not_reroutes(
    kanban_env, tmp_path, monkeypatch,
):
    """``error_max_turns`` is a task-complexity bound, not provider
    unavailability — block, do not queue the lower fallback."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    payload = {
        "type": "result",
        "subtype": "error",
        "error": {"code": "error_max_turns", "message": "max turn count reached"},
    }
    _fake_claude_run(monkeypatch, payload, returncode=1)

    with kb.connect() as conn:
        tid = _seed_task(conn)
        exit_code = kef.run_fallback_attempt(tid, str(workspace))
        assert exit_code == 0
        assert _task_state(conn, tid)["status"] == "blocked"
        events = [e.kind for e in kb.list_events(conn, tid)]
        assert "executor_fallback_queued" not in events


def test_claude_binary_missing_exits_75(kanban_env, tmp_path, monkeypatch):
    """``claude`` not on PATH = executor backend unavailable (qualifying)."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def _raise(cmd, *args, **kwargs):
        raise FileNotFoundError("claude not found")

    monkeypatch.setattr(kef.subprocess, "run", _raise)

    with kb.connect() as conn:
        tid = _seed_task(conn)
        exit_code = kef.run_fallback_attempt(tid, str(workspace))
        assert exit_code == kb.KANBAN_RATE_LIMIT_EXIT_CODE
        assert _task_state(conn, tid)["status"] == "running"


def test_unparseable_claude_output_fails_closed(
    kanban_env, tmp_path, monkeypatch,
):
    """Unknown output/error -> block (no blind reroute on UNKNOWN)."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    def _weird(cmd, *args, **kwargs):
        return types.SimpleNamespace(returncode=3, stdout="", stderr="boom")

    monkeypatch.setattr(kef.subprocess, "run", _weird)

    with kb.connect() as conn:
        tid = _seed_task(conn)
        exit_code = kef.run_fallback_attempt(tid, str(workspace))
        assert exit_code == 0
        assert _task_state(conn, tid)["status"] == "blocked"


def test_wrapper_entrypoint_main(kanban_env, tmp_path, monkeypatch,
                                 capsys):
    """``python -m hermes_cli.kanban_executor_fallback <id> <ws>`` routes to
    run_fallback_attempt and exits with its code (verifiable from the CLI,
    which is what the dispatcher actually spawns)."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _fake_claude_run(
        monkeypatch,
        {"type": "result", "subtype": "success",
         "result": "EXECUTOR_FALLBACK_COMPLETE: done"},
    )
    with kb.connect() as conn:
        tid = _seed_task(conn)

    code = kef.main([tid, str(workspace)])
    assert code == 0
    with kb.connect() as conn:
        assert _task_state(conn, tid)["status"] == "done"