"""Tests for the kanban observation feed (AGENT_OBSERVATION_CONTRACT_V1, P2+P4).

Covers the acceptance criteria of the design doc
``research/metagpt-foundationagents/AGENT_OBSERVATION_CONTRACT_V1.md``:

- AC-1 regression: profile without rules → worker context has no observed block
- AC-2 filtering: a profile with rules sees EXACTLY the matching events
- AC-3 cursor/dedup: consecutive reads do not re-project observed events;
  cursor advances monotonically and is rebuildable
- AC-4 no-execution: applying observe rules spawns no worker/card
- AC-5 rollback: removing rules/env restores pre-V1 output
- AC-6 fail-open: invalid rule / failing query → log + no block, no crash
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import watch_rules as wr

_NOW = int(time.time())


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (same shape as
    tests/hermes_cli/test_kanban_db.py)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_OBSERVATIONS", raising=False)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect_closing() as c:
        yield c


def _make_task(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    title: str = "task",
    assignee: object = "monitor",
    tenant: object = None,
) -> str:
    assert isinstance(assignee, (str, type(None)))
    assert isinstance(tenant, (str, type(None)))
    conn.execute(
        "INSERT INTO tasks (id, title, body, assignee, status, created_at, tenant) "
        "VALUES (?, ?, NULL, ?, 'done', ?, ?)",
        (task_id, title, assignee, _NOW, tenant),
    )
    conn.commit()
    return task_id


def _add_event(
    conn: sqlite3.Connection,
    task_id: str,
    kind: str,
    *,
    age_s: int = 60,
    payload: object = None,
    run_id: object = None,
) -> int:
    """Insert an event ``age_s`` seconds before the shared test clock."""
    import json as _json

    cur = conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, run_id, kind, _json.dumps(payload) if payload else None, _NOW - age_s),
    )
    conn.commit()
    return int(cur.lastrowid)


def _write_watch_yaml(home: Path, profile: str, rules_yaml: str) -> Path:
    """Write a watch.yaml for a profile under the isolated hermes home."""
    if profile == "default":
        profile_dir = home
    else:
        profile_dir = home / "profiles" / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
    watch_file = profile_dir / "watch.yaml"
    watch_file.write_text(rules_yaml, encoding="utf-8")
    return watch_file


# ---------------------------------------------------------------------------
# AC-1 — regression: no rules → no observation block
# ---------------------------------------------------------------------------


def test_ac1_no_rules_no_observation_block(kanban_home, conn):
    _make_task(conn, task_id="t1", assignee="monitor")
    _add_event(conn, "t1", "blocked")

    ctx = kb.build_worker_context(conn, "t1")
    assert "## Observable events (watch match)" not in ctx
    # Structural regression: the core sections are still present.
    assert "# Kanban task t1" in ctx
    assert "Assignee: monitor" in ctx


def test_ac1_task_without_assignee_never_observes(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        "watch:\n  - name: r1\n    match: {kinds: [blocked]}\n",
    )
    _make_task(conn, task_id="t1", assignee=None)
    _add_event(conn, "t1", "blocked")

    ctx = kb.build_worker_context(conn, "t1")
    assert "## Observable events (watch match)" not in ctx


# ---------------------------------------------------------------------------
# AC-2 — filtering: only matching events are projected
# ---------------------------------------------------------------------------


def test_ac2_only_matching_kinds_projected(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        "watch:\n  - name: blocked-watch\n    match: {kinds: [blocked, block_loop_detected]}\n",
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _make_task(conn, task_id="t2", assignee="other")
    _add_event(conn, "t1", "blocked")
    _add_event(conn, "t2", "completed")

    feed = kb.build_observation_feed(conn, "monitor", now=_NOW)
    assert "blocked on t1" in feed
    assert "completed on t2" not in feed
    # The block header + rule header present.
    assert "## Observable events (watch match)" in feed
    assert "### rule `blocked-watch`" in feed


def test_ac2_scope_filters_tenants_and_assignees(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        (
            "watch:\n"
            "  - name: scoped\n"
            "    match:\n"
            "      kinds: [blocked]\n"
            "      scope: {tenants: [acme], assignees: [alice]}\n"
        ),
    )
    _make_task(conn, task_id="t1", assignee="alice", tenant="acme")
    _make_task(conn, task_id="t2", assignee="bob", tenant="acme")
    _make_task(conn, task_id="t3", assignee="alice", tenant="globex")
    _add_event(conn, "t1", "blocked")  # match
    _add_event(conn, "t2", "blocked")  # wrong assignee
    _add_event(conn, "t3", "completed")  # wrong tenant+kind

    feed = kb.build_observation_feed(conn, "monitor", now=_NOW)
    assert "blocked on t1" in feed
    assert "t2" not in feed
    assert "t3" not in feed


def test_ac2_payload_contains_filters(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        (
            "watch:\n"
            "  - name: payload-watch\n"
            "    match:\n"
            "      kinds: [blocked]\n"
            "      payload_contains: rotating\n"
        ),
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _make_task(conn, task_id="t2", assignee="monitor")
    _add_event(conn, "t1", "blocked", payload={"reason": "rotating lock"})
    _add_event(conn, "t2", "blocked", payload={"reason": "other"})

    feed = kb.build_observation_feed(conn, "monitor", now=_NOW)
    assert "blocked on t1" in feed
    assert "blocked on t2" not in feed


def test_ac2_window_excludes_old_events(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        "watch:\n  - name: windowed\n    match: {kinds: [blocked]}\n    window_s: 100\n",
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _make_task(conn, task_id="t2", assignee="monitor")
    _add_event(conn, "t1", "blocked", age_s=50)  # inside window (100s)
    _add_event(conn, "t2", "blocked", age_s=500)  # outside window

    feed = kb.build_observation_feed(conn, "monitor", now=_NOW)
    assert "blocked on t1" in feed
    assert "blocked on t2" not in feed


# ---------------------------------------------------------------------------
# AC-3 — cursor/dedup: no re-projection of observed events
# ---------------------------------------------------------------------------


def test_ac3_cursor_dedup_and_rebuild(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        "watch:\n  - name: r1\n    match: {kinds: [blocked]}\n",
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _add_event(conn, "t1", "blocked")

    # First read projects the event; cursor advances to the event id.
    feed1 = kb.build_observation_feed(conn, "monitor", now=_NOW, advance_cursor=True)
    assert "blocked on t1" in feed1
    cursor = kb._read_observation_cursor(conn, "monitor", "r1")
    assert cursor > 0
    ev = conn.execute("SELECT MAX(id) AS m FROM task_events").fetchone()
    assert cursor == int(ev["m"])

    # Second read with advance → nothing new → empty block.
    feed2 = kb.build_observation_feed(conn, "monitor", now=_NOW, advance_cursor=True)
    assert feed2 == ""

    # Rebuildable cache: delete the cursor row → window re-projected.
    conn.execute("DELETE FROM observation_cursors WHERE profile='monitor' AND rule_id='r1'")
    conn.commit()
    feed3 = kb.build_observation_feed(conn, "monitor", now=_NOW, advance_cursor=False)
    assert "blocked on t1" in feed3


def test_ac3_new_events_after_cursor_are_seen(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        "watch:\n  - name: r1\n    match: {kinds: [blocked]}\n",
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _add_event(conn, "t1", "blocked")

    kb.build_observation_feed(conn, "monitor", now=_NOW, advance_cursor=True)
    _add_event(conn, "t1", "blocked")

    feed = kb.build_observation_feed(conn, "monitor", now=_NOW, advance_cursor=True)
    assert "blocked on t1" in feed  # the NEW event surfaces, the old one doesn't
    # Exactly one event line (the old one is not re-projected).
    assert feed.count("- blocked on t1") == 1


# ---------------------------------------------------------------------------
# AC-4 — no-execution: observing spawns nothing
# ---------------------------------------------------------------------------


def test_ac4_observation_never_creates_tasks_or_runs(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        "watch:\n  - name: r1\n    match: {kinds: [blocked]}\n",
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _add_event(conn, "t1", "blocked")

    kb.build_observation_feed(conn, "monitor", now=_NOW, advance_cursor=True)
    kb.build_worker_context(conn, "t1")

    rows = conn.execute("SELECT COUNT(*) AS n FROM tasks WHERE id != 't1'").fetchone()
    assert int(rows["n"]) == 0  # no new cards
    runs = conn.execute("SELECT COUNT(*) AS n FROM task_runs").fetchone()
    assert int(runs["n"]) == 0  # no new runs/workers
    events = conn.execute(
        "SELECT COUNT(*) AS n FROM task_events WHERE kind NOT IN ('blocked')"
    ).fetchone()
    assert int(events["n"]) == 0  # observation writes no events


# ---------------------------------------------------------------------------
# AC-5 — rollback: disabling restores pre-V1 output
# ---------------------------------------------------------------------------


def test_ac5_env_flag_disables_observation(kanban_home, conn, monkeypatch):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        "watch:\n  - name: r1\n    match: {kinds: [blocked]}\n",
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _add_event(conn, "t1", "blocked")

    monkeypatch.setenv("HERMES_KANBAN_OBSERVATIONS", "off")
    ctx = kb.build_worker_context(conn, "t1")
    assert "## Observable events (watch match)" not in ctx
    assert kb.build_observation_feed(conn, "monitor", now=_NOW) == ""

    monkeypatch.delenv("HERMES_KANBAN_OBSERVATIONS")
    assert "## Observable events (watch match)" in kb.build_worker_context(conn, "t1")


def test_ac5_removing_rules_restores_output(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        "watch:\n  - name: r1\n    match: {kinds: [blocked]}\n",
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _add_event(conn, "t1", "blocked")

    ctx_with = kb.build_worker_context(conn, "t1")
    assert "## Observable events (watch match)" in ctx_with

    (kanban_home / "profiles" / "monitor" / "watch.yaml").unlink()
    ctx_without = kb.build_worker_context(conn, "t1")
    assert "## Observable events (watch match)" not in ctx_without


# ---------------------------------------------------------------------------
# AC-6 — fail-open: invalid rules / failed queries never crash the worker
# ---------------------------------------------------------------------------


def test_ac6_invalid_rule_is_warned_not_fatal(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        (
            "watch:\n"
            "  - name: good\n"
            "    match: {kinds: [blocked]}\n"
            "  - name: bad\n"
            "    match: {kinds: 42}\n"
        ),
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _add_event(conn, "t1", "blocked")

    # Parsing the rules warns and keeps the valid one — no exception.
    rules, warnings = wr.load_watch_rules("monitor")
    assert len(rules) == 1
    assert any("bad" in w for w in warnings)

    # Worker context still renders with the valid rule's block.
    ctx = kb.build_worker_context(conn, "t1")
    assert "## Observable events (watch match)" in ctx


def test_ac6_failing_query_is_fail_open(kanban_home, conn, monkeypatch):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        "watch:\n  - name: r1\n    match: {kinds: [blocked]}\n",
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _add_event(conn, "t1", "blocked")

    def boom(*args, **kwargs):
        raise sqlite3.OperationalError("simulated db failure")

    monkeypatch.setattr(kb, "_observation_feed_rows", boom)
    # Direct feed call is fail-open.
    assert kb.build_observation_feed(conn, "monitor", now=_NOW) == ""
    # Worker context survives and drops the block.
    ctx = kb.build_worker_context(conn, "t1")
    assert "## Observable events (watch match)" not in ctx
    assert "# Kanban task t1" in ctx


# ---------------------------------------------------------------------------
# watch_rules parser unit coverage
# ---------------------------------------------------------------------------


def test_parse_watch_rules_shapes():
    rules, warnings = wr.parse_watch_rules(
        {
            "watch": [
                {"name": "a", "match": {"kinds": ["blocked", "*"]}, "window_s": 60},
                {"name": "b", "match": {"kinds": "blocked"}},
                {"name": "c", "match": {"scope": {"tenants": ["acme"]}}},
                {"name": "surface-rule", "mode": "surface"},
                {"name": 42},
                "not-a-dict",
            ]
        }
    )
    assert warnings  # surface + invalid entries produce warnings
    by_name = {r.name: r for r in rules}
    assert set(by_name) == {"a", "b", "c"}
    assert by_name["a"].window_s == 60
    # "*" in kinds collapsed to None (match anything).
    assert by_name["a"].kinds == frozenset({"blocked"})
    assert by_name["b"].kinds == frozenset({"blocked"})
    assert "surface-rule" not in by_name  # P5 out of scope → skipped
    assert by_name["c"].tenants == frozenset({"acme"})
    assert by_name["c"].assignees is None


def test_parse_watch_rules_never_raises():
    for junk in (None, 42, "x", [], {}, {"watch": "nope"}, [{"name": ""}]):
        rules, warnings = wr.parse_watch_rules(junk)
        assert isinstance(rules, list)
        assert isinstance(warnings, list)


# ---------------------------------------------------------------------------
# Regression — HALLAZGO-1 (QA t_28cd552a): scope/payload filters must never
# starve the top-N. >80 same-kind events that DON'T match, arriving more
# recently, must not hide matching events buried below. Pre-fix the SQL
# pre-filtered only by kind with LIMIT 80, so a full page of non-matching
# rows came back and the feed was EMPTY (0/6-style repro — violates AC-2).
# ---------------------------------------------------------------------------


def test_regression_scope_match_buried_under_many_nonmatching(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        (
            "watch:\n"
            "  - name: scoped-buried\n"
            "    match:\n"
            "      kinds: [blocked]\n"
            "      scope: {tenants: [acme]}\n"
        ),
    )
    _make_task(conn, task_id="t1", assignee="monitor", tenant="acme")
    _make_task(conn, task_id="t2", assignee="monitor", tenant="globex")
    # The matching event is OLDER than the flood: pre-fix the LIMIT 80 page
    # was filled by the 100 newer globex rows and the feed came back empty.
    _add_event(conn, "t1", "blocked", payload={"reason": "rotating"})
    for i in range(100):
        _add_event(conn, "t2", "blocked", payload={"seq": i})

    feed = kb.build_observation_feed(conn, "monitor", now=_NOW)
    assert "blocked on t1" in feed  # the buried acme match MUST surface
    assert "blocked on t2" not in feed  # non-matching tenant never projected


def test_regression_payload_match_buried_under_many_nonmatching(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        (
            "watch:\n"
            "  - name: payload-buried\n"
            "    match:\n"
            "      kinds: [blocked]\n"
            "      payload_contains: rotating\n"
        ),
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _make_task(conn, task_id="t2", assignee="monitor")
    # Matching event buried under >80 newer same-kind events WITHOUT the
    # needle: pre-fix the LIMIT 80 page was entirely non-matching and the
    # feed came back EMPTY.
    _add_event(conn, "t1", "blocked", payload={"reason": "rotating lock"})
    for i in range(100):
        _add_event(conn, "t2", "blocked", payload={"reason": f"other {i}"})

    feed = kb.build_observation_feed(conn, "monitor", now=_NOW)
    assert "blocked on t1" in feed  # the buried needle match MUST surface
    assert "blocked on t2" not in feed


def test_regression_buried_match_dedup_after_pagination(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        (
            "watch:\n"
            "  - name: payload-buried\n"
            "    match:\n"
            "      kinds: [blocked]\n"
            "      payload_contains: rotating\n"
        ),
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    _make_task(conn, task_id="t2", assignee="monitor")
    _add_event(conn, "t1", "blocked", payload={"reason": "rotating lock"})
    for i in range(100):
        _add_event(conn, "t2", "blocked", payload={"reason": f"other {i}"})

    feed1 = kb.build_observation_feed(conn, "monitor", now=_NOW, advance_cursor=True)
    assert "blocked on t1" in feed1
    # The cursor advanced to the surfaced match: a second read must NOT
    # re-project it, even though >80 non-matching rows sit above it (the
    # pagination walk also stays monotonic — no row examined twice).
    feed2 = kb.build_observation_feed(conn, "monitor", now=_NOW, advance_cursor=True)
    assert feed2 == ""


def test_regression_flood_still_capped_at_top_n(kanban_home, conn):
    _write_watch_yaml(
        kanban_home,
        "monitor",
        "watch:\n  - name: flood\n    match: {kinds: [blocked]}\n",
    )
    _make_task(conn, task_id="t1", assignee="monitor")
    for i in range(60):
        _add_event(conn, "t1", "blocked", payload={"seq": i})

    feed = kb.build_observation_feed(conn, "monitor", now=_NOW)
    # Only the top-N most recent matching events are projected — caps hold.
    lines = [ln for ln in feed.splitlines() if ln.startswith("- blocked on t1")]
    assert 0 < len(lines) <= kb._CTX_MAX_OBSERVED_EVENTS
    # Feed-level byte cap still applies to the whole block.
    assert len(feed.encode("utf-8")) <= kb._CTX_MAX_OBSERVATION_BYTES