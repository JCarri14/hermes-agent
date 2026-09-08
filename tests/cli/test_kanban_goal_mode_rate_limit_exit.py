"""Goal-mode kanban workers must exit with the EX_TEMPFAIL rate-limit
sentinel when the provider hits a 429 / usage-limit wall INSIDE the goal
loop (a later turn, not the first one).

Regression: a kanban goal_mode worker whose first turn succeeded and whose
second turn hit HTTP 429 used to either crash (exit 1, counted as a
failure → gave_up) or spin on empty responses until its turn budget blocked
the card. The sentinel exit code (``KANBAN_RATE_LIMIT_EXIT_CODE`` = 75)
lets the dispatcher's ``detect_crashed_workers`` classify the run as
``rate_limited`` and release the card back to ``ready`` without counting a
failure.

Covered here (all through the real ``cli.main`` quiet path):
1. a later goal-loop turn returning a failed result dict with
   ``failure_reason="rate_limit"`` → SystemExit(75);
2. a later goal-loop turn raising a provider 429 exception → SystemExit(75);
3. a later goal-loop turn raising an ordinary (non-rate-limit) exception →
   the existing "stopped" contract (exit 0 from the succeeded first turn,
   never the sentinel).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb


class _Fake429(Exception):
    """Minimal stand-in for a provider HTTP 429 with a usage-limit body."""

    status_code = 429

    def __init__(self, message):
        super().__init__(message)
        self.body = {"error": {"type": "usage_limit_reached", "message": message}}


@pytest.fixture
def kanban_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a real kanban DB containing one task."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(name, raising=False)
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="goal task", assignee="worker")
    return task_id


def _make_fake_cli(scripted):
    """FakeCLI whose agent replays ``scripted`` run_conversation outcomes.

    Each entry is either a result dict or an exception instance to raise.
    """

    def run_conversation(*, user_message, conversation_history):
        entry = scripted.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return entry

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.provider = "test-provider"
            self.model = "test-model"
            self.session_id = "quiet-session"
            self.conversation_history = []
            self._active_agent_route_signature = "same-route"
            self.agent = SimpleNamespace(
                session_id="quiet-session",
                platform="cli",
                quiet_mode=False,
                suppress_status_output=False,
                stream_delta_callback=object(),
                tool_gen_callback=object(),
                run_conversation=run_conversation,
            )

        def _claim_active_session(self, surface, *, stderr=False):
            return True

        def _ensure_runtime_credentials(self):
            return True

        def _resolve_turn_agent_config(self, effective_query):
            return {
                "signature": "same-route",
                "model": None,
                "runtime": None,
                "request_overrides": None,
            }

        def _init_agent(self, **kwargs):
            return True

    return FakeCLI


def _run_goal_mode_main(monkeypatch, FakeCLI, task_id):
    import cli as cli_mod
    from hermes_cli import goals

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda _cli: None)
    # Judge keeps saying "continue" so the goal loop drives a second turn.
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda goal, response, **_kw: (
            "continue",
            "scripted:continue",
            False,
            None,
            False,
        ),
    )
    cli_mod.main(query="work kanban task", quiet=True, toolsets="terminal")


def test_goal_loop_429_result_dict_exits_with_sentinel(monkeypatch, kanban_env):
    """Later turn returns a failed dict with failure_reason=rate_limit."""
    scripted = [
        {"final_response": "started", "failed": False},
        {
            "final_response": "",
            "error": "API usage limit has been reached",
            "failed": True,
            "failure_reason": "rate_limit",
        },
    ]
    FakeCLI = _make_fake_cli(scripted)
    with pytest.raises(SystemExit) as exc:
        _run_goal_mode_main(monkeypatch, FakeCLI, kanban_env)
    assert exc.value.code == kb.KANBAN_RATE_LIMIT_EXIT_CODE


def test_goal_loop_429_exception_exits_with_sentinel(monkeypatch, kanban_env):
    """Later turn raises a provider 429 (usage-limit body)."""
    scripted = [
        {"final_response": "started", "failed": False},
        _Fake429(
            "Execution is blocked because the API usage limit "
            "has been reached (HTTP 429)."
        ),
    ]
    FakeCLI = _make_fake_cli(scripted)
    with pytest.raises(SystemExit) as exc:
        _run_goal_mode_main(monkeypatch, FakeCLI, kanban_env)
    assert exc.value.code == kb.KANBAN_RATE_LIMIT_EXIT_CODE


def test_goal_loop_other_error_never_exits_sentinel(monkeypatch, kanban_env):
    """A non-rate-limit failure keeps the generic stopped → exit 0 path."""
    scripted = [
        {"final_response": "started", "failed": False},
        ValueError("tool exploded"),
    ]
    FakeCLI = _make_fake_cli(scripted)
    with pytest.raises(SystemExit) as exc:
        _run_goal_mode_main(monkeypatch, FakeCLI, kanban_env)
    assert exc.value.code == 0
    assert exc.value.code != kb.KANBAN_RATE_LIMIT_EXIT_CODE


def test_turn_rate_limit_classifier(monkeypatch, kanban_env):
    """Only provider quota walls classify as rate-limited turns."""
    import cli as cli_mod

    assert (
        cli_mod._kanban_goal_turn_rate_limited(
            _Fake429(
                "Execution is blocked because the API usage limit "
                "has been reached (HTTP 429)."
            )
        )
        is True
    )
    assert (
        cli_mod._kanban_goal_turn_rate_limited(
            _Fake429("Weekly usage limit reached. Resets in 6hr 29min.")
        )
        is True
    )
    # Never auth / tool / task / server errors.
    assert cli_mod._kanban_goal_turn_rate_limited(Exception("unauthorized")) is False
    assert cli_mod._kanban_goal_turn_rate_limited(ValueError("boom")) is False
    assert (
        cli_mod._kanban_goal_turn_rate_limited(Exception("gateway timeout 504"))
        is False
    )
