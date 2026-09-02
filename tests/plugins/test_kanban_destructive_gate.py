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

import json
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


def test_productive_dispatch_blocks_ambiguous_destructive_work_without_go(board_home):
    """Strict productive context closes the old verb-without-live-marker gap."""
    with kb.connect() as conn:
        _insert(conn, "ambiguous", title="Delete a temp file",
                body="Delete /tmp/scratch.txt locally", assignee="default", status="ready")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False, destructive_gate=True,
            destructive_strict=True,
        )
    assert res.skipped_destructive_gate == [("ambiguous", "DESTRUCTIVE_LIVE")]
    assert res.spawned == []


def test_productive_dispatch_allows_explicit_benign_allowlist_match(board_home):
    with kb.connect() as conn:
        _insert(conn, "benign", title="Delete a temp file",
                body="Delete /tmp/scratch.txt locally", assignee="default", status="ready")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False, destructive_gate=True,
            destructive_strict=True,
            destructive_allowlist=[{"pattern": r"delete /tmp/.*", "reason": "local scratch"}],
        )
    assert res.skipped_destructive_gate == []
    assert [spawn[0] for spawn in res.spawned] == ["benign"]


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


# ═══════════════════════════════════════════════════════════════════════════════
# v1.1 deterministic human-GO — claim_task / complete_task guards (bounded
# test fixture: isolated board DB, config injected via load_config patch).
# ═══════════════════════════════════════════════════════════════════════════════

def _set_destructive_config(monkeypatch, **overrides):
    """Patch ``hermes_cli.config.load_config`` to drive the claim/complete guards
    deterministically (same pattern as test_kanban_cli_dispatch_passthrough)."""
    cfg = {
        "kanban": {
            "destructive_gate": True,
            "destructive_require_preverify": False,
            "destructive_tenants": [],
            "destructive_authorized_ttl_seconds": 604800,
            **overrides,
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)


def _destructive_card(conn, tid, *, title="Cleanup client bucket",
                      body="Delete the R2 bucket erp-client-a-docs.",
                      assignee="dev", tenant="CLIENT_A", extra=""):
    full_body = body + ("\n" + extra if extra else "")
    _insert(conn, tid, title=title, body=full_body, assignee=assignee, status="ready")
    conn.execute("UPDATE tasks SET tenant = ? WHERE id = ?", (tenant, tid))


def _derived_action_id(conn, tid):
    from hermes_cli.destructive_gate import compute_destructive_action_id

    row = conn.execute("SELECT title, body, tenant, assignee FROM tasks WHERE id=?", (tid,)).fetchone()
    return compute_destructive_action_id(
        row["title"] or "", row["body"] or "",
        tenant=row["tenant"] or None, assignee=row["assignee"] or None,
    )


def _record_full_sequence(conn, tid, *, author="operator@lab", action_id=None,
                          with_postcondition=False, preverify=True):
    action_id = action_id or _derived_action_id(conn, tid)
    if preverify:
        kb.record_destructive_event(conn, tid, "destructive_preverified", action_id,
                                    author="operator@lab", by="operator@lab")
    kb.add_comment(conn, tid, author, f"@go destructive {tid} {action_id}",
                   destructive_action_id=action_id)
    if with_postcondition:
        kb.record_destructive_event(conn, tid, "destructive_postcondition_posted",
                                    action_id, by="dev", evidence="read returned 404")
    return action_id


def test_claim_blocked_no_preverify_no_go(board_home, monkeypatch):
    """Strict mode: no destructive_preverified event -> claim refused."""
    _set_destructive_config(monkeypatch, destructive_require_preverify=True)
    with kb.connect() as conn:
        _destructive_card(conn, "d1", tenant="CLIENT_A")
    with kb.connect_closing() as conn:
        claimed = kb.claim_task(conn, "d1")
    assert claimed is None
    with kb.connect() as conn:
        held = list(conn.execute(
            "SELECT payload FROM task_events WHERE task_id='d1' AND kind='destructive_gate_held'",
        ))
    assert len(held) == 1
    assert "pre-verification" in (held[0]["payload"] or "")
    with kb.connect() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id='d1'").fetchone()
    assert row["status"] == "ready"


def test_claim_blocked_go_without_event(board_home, monkeypatch):
    """A legacy GO comment with no canonical event does not count when the card
    declares an action binding (fail-closed: event is the source of truth)."""
    _set_destructive_config(monkeypatch)
    from hermes_cli.destructive_gate import compute_destructive_action_id

    with kb.connect() as conn:
        # Declared binding on the card.
        _destructive_card(conn, "d2", body="Delete the R2 bucket erp-client-a-docs.\n"
                         "destructive_action_id: sha256:declared2")
        _insert_go(conn, "d2", "@go destructive d2", author="human")
    with kb.connect_closing() as conn:
        claimed = kb.claim_task(conn, "d2")
    assert claimed is None
    with kb.connect() as conn:
        held = list(conn.execute(
            "SELECT payload FROM task_events WHERE task_id='d2' AND kind='destructive_gate_held'",
        ))
    assert len(held) == 1
    assert "mismatched GO" in (held[0]["payload"] or "")


def test_claim_blocked_stale_go_edited_after(board_home, monkeypatch):
    """GO valid, then the card is edited (title/body) after the GO -> claim
    refused with a stale reason."""
    _set_destructive_config(monkeypatch, destructive_require_preverify=True)
    with kb.connect() as conn:
        _destructive_card(conn, "d3", tenant="CLIENT_A")
        action_id = _record_full_sequence(conn, "d3")
        # Card edited AFTER the GO (event with a higher id).
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, 'edited', ?, ?)",
            ("d3", '{"fields": ["title"]}', int(__import__("time").time())),
        )
    with kb.connect_closing() as conn:
        claimed = kb.claim_task(conn, "d3")
    assert claimed is None
    with kb.connect() as conn:
        held = list(conn.execute(
            "SELECT payload FROM task_events WHERE task_id='d3' AND kind='destructive_gate_held'",
        ))
    assert held and "stale GO" in (held[0]["payload"] or "")
    assert action_id.startswith("sha256:")


def test_claim_blocked_stale_go_ttl_expired(board_home, monkeypatch):
    """GO older than the TTL -> claim refused (reason stale)."""
    _set_destructive_config(monkeypatch, destructive_require_preverify=True,
                            destructive_authorized_ttl_seconds=10)
    with kb.connect() as conn:
        _destructive_card(conn, "d4", tenant="CLIENT_A")
        action_id = _record_full_sequence(conn, "d4")
        # Re-age the authorized event beyond the TTL (10s).
        conn.execute(
            "UPDATE task_events SET created_at = ? "
            "WHERE kind='destructive_authorized' AND task_id='d4'",
            (int(__import__("time").time()) - 11,),
        )
    with kb.connect_closing() as conn:
        claimed = kb.claim_task(conn, "d4")
    assert claimed is None
    with kb.connect() as conn:
        held = list(conn.execute(
            "SELECT payload FROM task_events WHERE task_id='d4' AND kind='destructive_gate_held'",
        ))
    assert held and "stale" in (held[0]["payload"] or "")
    assert action_id.startswith("sha256:")


def test_claim_blocked_mismatched_action_id(board_home, monkeypatch):
    """GO bound to action A, card declares action B -> claim refused
    (the GO does not authorize the requested action)."""
    _set_destructive_config(monkeypatch)
    with kb.connect() as conn:
        _destructive_card(conn, "d5", body="Delete the R2 bucket erp-client-a-docs.\n"
                         "destructive_action_id: sha256:bbbb")
        # Canonical event recorded against a DIFFERENT action.
        kb.record_destructive_event(conn, "d5", "destructive_authorized",
                                    "sha256:aaaa", author="operator@lab")
    with kb.connect_closing() as conn:
        claimed = kb.claim_task(conn, "d5")
    assert claimed is None
    with kb.connect() as conn:
        held = list(conn.execute(
            "SELECT payload FROM task_events WHERE task_id='d5' AND kind='destructive_gate_held'",
        ))
    assert len(held) == 1
    assert "mismatched GO" in (held[0]["payload"] or "")


def test_claim_allowed_full_sequence(board_home, monkeypatch):
    """pre-verification + canonical human GO -> claim admitted, task runs."""
    _set_destructive_config(monkeypatch, destructive_require_preverify=True)
    with kb.connect() as conn:
        _destructive_card(conn, "d6", tenant="CLIENT_A")
        _record_full_sequence(conn, "d6")
    with kb.connect_closing() as conn:
        claimed = kb.claim_task(conn, "d6")
    assert claimed is not None
    assert claimed.status == "running"
    with kb.connect() as conn:
        held = list(conn.execute(
            "SELECT 1 FROM task_events WHERE task_id='d6' AND kind='destructive_gate_held'",
        ))
    assert held == []


def test_completion_blocked_no_postcondition(board_home, monkeypatch):
    """Full claim sequence but no postcondition verification -> completion
    refused (DestructiveGateError), card stays running in-flight."""
    _set_destructive_config(monkeypatch, destructive_require_preverify=True)
    with kb.connect() as conn:
        _destructive_card(conn, "d7", tenant="CLIENT_A")
        _record_full_sequence(conn, "d7")
    with kb.connect_closing() as conn:
        claimed = kb.claim_task(conn, "d7")
    assert claimed is not None
    with kb.connect_closing() as conn:
        with pytest.raises(kb.DestructiveGateError) as excinfo:
            kb.complete_task(conn, "d7", result="done")
    assert "postcondition verification missing" in str(excinfo.value)
    with kb.connect() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id='d7'").fetchone()
    assert row["status"] == "running"  # rolled back — still in-flight


def test_completion_allowed_with_postcondition(board_home, monkeypatch):
    """Full sequence including postcondition -> completion admitted and the
    closing run metadata carries the GO traceability ids."""
    _set_destructive_config(monkeypatch, destructive_require_preverify=True)
    with kb.connect() as conn:
        _destructive_card(conn, "d8", tenant="CLIENT_A")
        action_id = _record_full_sequence(conn, "d8", with_postcondition=True)
    with kb.connect_closing() as conn:
        claimed = kb.claim_task(conn, "d8")
    assert claimed is not None
    with kb.connect_closing() as conn:
        ok = kb.complete_task(conn, "d8", result="done", metadata={"note": "x"})
    assert ok is True
    with kb.connect() as conn:
        row = conn.execute("SELECT status FROM tasks WHERE id='d8'").fetchone()
        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id='d8' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row["status"] == "done"
    md = json.loads(run["metadata"]) if run and run["metadata"] else {}
    assert md.get("approved_action_id") == action_id
    assert isinstance(md.get("authorized_event_id"), int)
    assert isinstance(md.get("postverified_event_id"), int)


def test_worker_spoofed_author_rejected(board_home, monkeypatch):
    """A worker/dashboard/system author can never create the canonical event —
    only a genuine human author can (GO event created exclusively via the
    canonical surface with a passable author)."""
    _set_destructive_config(monkeypatch)
    with kb.connect() as conn:
        _destructive_card(conn, "d9", tenant="CLIENT_A")
        action_id = _derived_action_id(conn, "d9")
    for spoof in ("worker", "dashboard", "hermes-system", "system", "dev"):
        with kb.connect() as conn:
            kb.add_comment(conn, "d9", spoof, f"@go destructive d9 {action_id}",
                           destructive_action_id=action_id)
    with kb.connect() as conn:
        n_spoofed = list(conn.execute(
            "SELECT 1 FROM task_events WHERE task_id='d9' AND kind='destructive_authorized'",
        ))
    assert n_spoofed == []  # no event from any denied author
    # A real human author creates the event.
    with kb.connect() as conn:
        kb.add_comment(conn, "d9", "operator@lab", f"@go destructive d9 {action_id}",
                       destructive_action_id=action_id)
    with kb.connect() as conn:
        evs = list(conn.execute(
            "SELECT payload FROM task_events WHERE task_id='d9' AND kind='destructive_authorized'",
        ))
    assert len(evs) == 1
    assert action_id in (evs[0]["payload"] or "")


def test_legacy_go_admitted_when_require_preverify_false(board_home, monkeypatch):
    """Backward compatibility: with require_preverify=false and NO declared
    action binding, a legacy GO comment (author != executor) still admits the
    claim — the existing opt-in flow keeps working."""
    _set_destructive_config(monkeypatch)  # require_preverify=False (default)
    with kb.connect() as conn:
        _destructive_card(conn, "d10", tenant="CLIENT_A")
        _insert_go(conn, "d10", "@go destructive d10", author="human")
    with kb.connect_closing() as conn:
        claimed = kb.claim_task(conn, "d10")
    assert claimed is not None
    assert claimed.status == "running"


def test_allowlist_benign_still_allowed(board_home, monkeypatch):
    """Allowlisted benign operation -> SAFE -> claim admitted (no regression)."""
    _set_destructive_config(monkeypatch,
                            destructive_allowlist=[{"pattern": r"delete /tmp/.*",
                                                    "reason": "local scratch"}])
    with kb.connect() as conn:
        _insert(conn, "d11", title="Delete a temp file",
                body="Delete /tmp/scratch.txt locally", assignee="dev", status="ready")
    with kb.connect_closing() as conn:
        claimed = kb.claim_task(conn, "d11")
    assert claimed is not None
    assert claimed.status == "running"


def test_cli_preverify_and_approve_record_events(board_home, monkeypatch, capsys):
    """The canonical CLI flow records the events end-to-end: preverification,
    human GO (comment + event), and postcondition."""
    from hermes_cli import kanban as kb_cli

    _set_destructive_config(monkeypatch)
    with kb.connect() as conn:
        _destructive_card(conn, "d12", tenant="CLIENT_A")
        action_id = _derived_action_id(conn, "d12")
    import argparse

    rc = kb_cli._cmd_preverify_destructive(
        argparse.Namespace(task_id="d12", action_id=action_id, author="operator@lab"))
    assert rc == 0
    rc = kb_cli._cmd_approve_destructive(
        argparse.Namespace(task_id="d12", action_id=action_id, author="operator@lab",
                           postcondition=False, evidence=None)
    )
    assert rc == 0
    rc = kb_cli._cmd_approve_destructive(
        argparse.Namespace(task_id="d12", action_id=action_id, author="dev",
                           postcondition=True, evidence="404")
    )
    assert rc == 0
    with kb.connect() as conn:
        kinds = [r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id='d12' ORDER BY id ASC"
        )]
        comments = [r["body"] for r in conn.execute(
            "SELECT body FROM task_comments WHERE task_id='d12'"
        )]
        row = conn.execute("SELECT status FROM tasks WHERE id='d12'").fetchone()
    assert "destructive_preverified" in kinds
    assert "destructive_authorized" in kinds
    assert "destructive_postcondition_posted" in kinds
    assert f"@go destructive d12 {action_id}" in comments
    assert row["status"] == "ready"  # CLI records; dispatcher admits later


def test_cli_approve_rejects_spoofed_author(board_home, monkeypatch, capsys):
    """The CLI refuses to record a GO for a denied author (fail fast with
    actionable output before anything is written)."""
    from hermes_cli import kanban as kb_cli

    _set_destructive_config(monkeypatch)
    with kb.connect() as conn:
        _destructive_card(conn, "d13", tenant="CLIENT_A")
        action_id = _derived_action_id(conn, "d13")
    rc = kb_cli._cmd_approve_destructive(
        __import__("argparse").Namespace(
            task_id="d13", action_id=action_id, author="dev",  # executor
            postcondition=False, evidence=None)
    )
    assert rc == 1
    with kb.connect() as conn:
        evs = list(conn.execute(
            "SELECT 1 FROM task_events WHERE task_id='d13' AND kind='destructive_authorized'",
        ))
    assert evs == []


from pathlib import Path  # noqa: E402  (needed by the board_home fixture)