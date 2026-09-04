"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    # decompose_task now routes through call_llm (see #35566) — mock it at
    # the source module so task config, extra_body, and retries stay out of
    # unit-test scope.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    # No-op shim retained for call-site compatibility: extra_body plumbing
    # now lives inside call_llm, which _patch_aux_client already mocks.
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist. The decomposer uses
    profiles_mod.list_profiles() to build the roster + valid-set, and
    profiles_mod.profile_exists() to resolve orchestrator/default."""
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def test_decompose_with_fanout_creates_children(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"


def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="route me safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to fallback.",
        "assignee": "made_up",
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x")  # ready, not triage

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    assert "not in triage" in outcome.reason


def _park_in_triage_via_block_loop(kanban_home, *, recurrences: int = 3):
    """Create a triage card and stamp it the way ``block_loop_detected`` does
    (status='triage' + block_recurrences > 0) — the ONLY path that raises
    block_recurrences while a task sits in triage."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="loop card", triage=True)
        conn.execute(
            "UPDATE tasks SET block_kind=?, block_recurrences=? WHERE id=?",
            ("needs_input", recurrences, tid),
        )
    return tid


def test_decompose_refuses_loop_triaged_without_touching_llm(kanban_home):
    tid = _park_in_triage_via_block_loop(kanban_home)

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client("unused") as llm, _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="auto-decomposer")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert "block_loop_detected" in outcome.reason
    # The guard fires before the auxiliary LLM is even invoked — the card is
    # awaiting a human decision, not a fresh spec.
    assert llm.mock_calls == []
    # Card stays parked in triage; it must NOT be re-specified / re-promoted.
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "triage"
    assert task.block_recurrences == 3


def test_decompose_fresh_triage_still_runs_composer(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened",
        "body": "Spec.",
    })
    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload) as llm, _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert llm.mock_calls  # fresh triage still reaches the decomposer LLM
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    # Single-task promotion: triage -> todo -> ready (recompute_ready runs on
    # the parent-free card right after specify).
    assert task.status == "ready"


def test_list_triage_ids_excludes_loop_triaged_only_when_requested(kanban_home):
    # Fresh triage card (block_recurrences stays 0).
    with kb.connect() as conn:
        fresh = kb.create_task(conn, title="fresh idea", triage=True)
    looped = _park_in_triage_via_block_loop(kanban_home)

    assert fresh != looped
    all_ids = decomp.list_triage_ids()
    assert fresh in all_ids and looped in all_ids

    fresh_only = decomp.list_triage_ids(exclude_loop_triaged=True)
    assert fresh in fresh_only
    assert looped not in fresh_only


