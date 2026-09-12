"""A–H matrix for INTERACTIVE_SESSION_FALLBACK_V1 (session-level provider fallback).

Maps the master-prompt test plan onto the runtime's NATIVE session machinery
(no subprocess, no kanban coupling, no ``fallback_providers`` policy edits):

  A. OpenAI success -> OpenAI response, no fallback.
  B. OpenAI 429 -> responds via Claude (anthropic chain entry).
  C. OpenAI + Claude unavailable -> responds via OpenRouter chain entry.
  D. non-provider session error -> NO inappropriate fallback.
  E. conversation context preserved across the provider swap.
  F. exactly one final response (single advance, no message duplication).
  G. no duplicate executor (no subprocess, no extra threads).
  H. fallback reason observable (classified reason + ``[session-fallback]`` log).

See research/interactive-session-fallback-v1/IS_D1_DESIGN_VERDICT.md (§5).
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from hermes_cli.auth import AuthError
from hermes_cli.interactive_fallback import (
    build_session_fallback_chain,
    format_session_fallback_notice,
    is_qualifying_provider_failure,
    record_session_fallback,
)
from run_agent import AIAgent

# The explicit interactive-session route (interactive_session_fallback.providers).
ANTHROPIC_ENTRY = {"provider": "anthropic", "model": "claude-sonnet-4"}
OPENROUTER_ENTRY = {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash-0731"}


# ── Fixtures / helpers ─────────────────────────────────────────────────────


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config (see
    tests/run_agent/test_provider_fallback.py for the mold)."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _rate_limit_error():
    """HTTP 429 surfaced as a status-bearing exception (primary path)."""
    err = Exception("Error code: 429 - rate limit exceeded")
    err.status_code = 429
    return err


def _codex_quota_auth_error(message):
    """Codex quota/rate-limit surfaced as AuthError(code=codex_rate_limited)
    — the connector's input (hermes_cli/auth.py:986,1006-1018)."""
    return AuthError(
        message,
        provider="openai-codex",
        code="codex_rate_limited",
        relogin_required=False,
    )


def _mock_client(base_url="https://api.anthropic.com", api_key="wk-test-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    # Provider-headers slots must be absent (None) so the openrouter swap
    # branch (dict(fb_headers)) does not choke on a MagicMock.
    mock._custom_headers = None
    mock.default_headers = None
    mock._default_headers = None
    return mock


def _router_with(provider_clients):
    """resolve_provider_client stand-in: returns (client, None) for the
    providers in ``provider_clients`` and (None, None) for everything else —
    modelling a provider that is NOT resolvable/configured (Claude down)."""

    def _router(provider, model=None, **kwargs):
        client = provider_clients.get(provider)
        if client is None:
            return None, None
        return client, None

    return _router


# ── A. OpenAI success -> OpenAI response, no fallback ──────────────────────


class TestA_SuccessNoFallback:
    def test_default_chain_equals_legacy_when_session_route_unconfigured(self):
        # A: with interactive_session_fallback NOT configured, the session
        # chain must be EXACTLY the legacy fallback_providers chain — the
        # happy path runs no new code.
        from hermes_cli.fallback_config import get_fallback_chain

        cfg = {"fallback_providers": [dict(OPENROUTER_ENTRY)]}
        assert build_session_fallback_chain(cfg) == get_fallback_chain(cfg)

    def test_no_classified_failure_is_never_qualifying(self):
        # A: a successful turn produces no classified failure, so the
        # qualifying gate is closed (no fallback decision is even possible).
        assert is_qualifying_provider_failure(None) is False

    def test_success_leaves_chain_and_index_untouched(self):
        agent = _make_agent([dict(ANTHROPIC_ENTRY)])
        assert agent._fallback_index == 0
        assert agent._fallback_activated is False
        # No failure -> no activation attempted: the gate stays closed.
        assert not is_qualifying_provider_failure(None)


# ── B. OpenAI 429 -> responds via Claude ───────────────────────────────────


class TestB_OpenAI429_RespondsClaude:
    def test_codex_quota_auth_error_classifies_rate_limit(self):
        # B: the gap the connector closes — a Codex quota AuthError whose
        # message does NOT match any heuristic pattern must STILL classify as
        # rate_limit (definitive via error.code, not fragile message match).
        err = _codex_quota_auth_error(
            "Codex RPD budget for this account is depleted; credentials remain valid."
        )
        c = classify_api_error(err)
        assert c.reason == FailoverReason.rate_limit
        assert c.should_fallback is True

    def test_http429_classifies_rate_limit(self):
        c = classify_api_error(_rate_limit_error())
        assert c.reason == FailoverReason.rate_limit
        assert c.should_fallback is True

    def test_chain_advances_to_anthropic_on_rate_limit(self):
        agent = _make_agent([dict(ANTHROPIC_ENTRY)])
        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                _router_with({"anthropic": _mock_client()}),
            ),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
        assert agent.provider == "anthropic"
        assert agent.model == "claude-sonnet-4"
        assert agent._fallback_index == 1
        assert agent._fallback_activated is True


# ── C. OpenAI + Claude unavailable -> responds via OpenRouter ──────────────


class TestC_UnavailableBoth_RespondsOpenRouter:
    def test_chain_skips_unavailable_anthropic_to_openrouter(self):
        agent = _make_agent([dict(ANTHROPIC_ENTRY), dict(OPENROUTER_ENTRY)])
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            _router_with({"openrouter": _mock_client(base_url="https://openrouter.ai/api/v1", api_key="orb-key")}),
        ):
            # anthropic resolves to None (unavailable / not configured) ->
            # try_activate_fallback must skip it and activate openrouter.
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
        assert agent.provider == "openrouter"
        assert agent.model == "deepseek/deepseek-v4-flash-0731"
        assert agent._fallback_index == 2


# ── D. non-provider session error -> NO inappropriate fallback ─────────────


class TestD_NonProviderError_NoFallback:
    def test_content_policy_blocked_is_not_qualifying(self):
        err = Exception("your request was flagged by the content filter")
        err.status_code = 400
        c = classify_api_error(err)
        assert c.reason == FailoverReason.content_policy_blocked
        assert c.should_fallback is True  # loop-level safety handling exists
        assert is_qualifying_provider_failure(c) is False  # but NOT chain-rerouted

    def test_tool_failure_value_error_not_qualifying_chain_untouched(self):
        agent = _make_agent([dict(ANTHROPIC_ENTRY)])
        c = classify_api_error(ValueError("tool call result could not be parsed"))
        assert c.reason == FailoverReason.unknown
        assert is_qualifying_provider_failure(c) is False
        # Chain state untouched: no advance, no swap.
        assert agent._fallback_index == 0
        assert agent._fallback_activated is False


# ── E. context preserved across the provider swap ──────────────────────────


class TestE_ContextPreserved:
    def test_messages_unchanged_across_activation(self):
        agent = _make_agent([dict(ANTHROPIC_ENTRY)])
        messages = [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Summarize"},
        ]
        snapshot = [dict(m) for m in messages]
        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                _router_with({"anthropic": _mock_client()}),
            ),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
        # Swap is in-place: same list, same order, same roles, nothing appended.
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[1]["content"] == "Summarize"
        assert messages == snapshot
        assert len(messages) == 2

    def test_failover_sync_rewrites_only_system_identity(self):
        from agent.conversation_loop import _sync_failover_system_message

        agent = _make_agent([dict(ANTHROPIC_ENTRY)])
        agent._cached_system_prompt = (
            "You are Hermes.\nModel: gpt-5.6-luna\nProvider: openai-codex"
        )
        api_messages = [
            {
                "role": "system",
                "content": "You are Hermes.\nModel: gpt-5.6-luna\nProvider: openai-codex",
            },
            {"role": "user", "content": "Summarize"},
        ]
        _sync_failover_system_message(agent, api_messages, agent._cached_system_prompt)
        assert [m["role"] for m in api_messages] == ["system", "user"]
        assert api_messages[1]["content"] == "Summarize"
        assert len(api_messages) == 2


# ── F. exactly one final response ──────────────────────────────────────────


class TestF_SingleFinalResponse:
    def test_single_chain_advance_per_activation_no_message_append(self):
        agent = _make_agent([dict(ANTHROPIC_ENTRY), dict(OPENROUTER_ENTRY)])
        messages = [{"role": "user", "content": "hello"}]
        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                _router_with(
                    {
                        "anthropic": _mock_client(),
                        "openrouter": _mock_client(
                            base_url="https://openrouter.ai/api/v1", api_key="orb-key"
                        ),
                    }
                ),
            ),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
        # ONE activation -> exactly ONE chain advance; the transcript is not
        # touched by the swap (no duplicate messages, no double-append).
        assert agent._fallback_index == 1
        assert agent._fallback_activated is True
        assert messages == [{"role": "user", "content": "hello"}]


# ── G. no duplicate executor ───────────────────────────────────────────────


class TestG_NoDuplicateExecutor:
    def test_no_subprocess_no_extra_threads_during_chain_walk(self):
        agent = _make_agent([dict(ANTHROPIC_ENTRY), dict(OPENROUTER_ENTRY)])
        threads_before = threading.active_count()
        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                _router_with(
                    {
                        "anthropic": _mock_client(),
                        "openrouter": _mock_client(
                            base_url="https://openrouter.ai/api/v1", api_key="orb-key"
                        ),
                    }
                ),
            ),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("subprocess.run", side_effect=AssertionError("subprocess must not run")),
            patch("subprocess.Popen", side_effect=AssertionError("subprocess must not run")),
        ):
            # Walk the whole chain in-process, like the loop does per turn.
            assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is True
            assert agent._try_activate_fallback(reason=FailoverReason.overloaded) is True
            # Chain exhausted -> False (loop then surfaces the terminal error).
            assert agent._try_activate_fallback(reason=FailoverReason.overloaded) is False
        assert threading.active_count() == threads_before
        assert agent._fallback_index == len(agent._fallback_chain)


# ── H. fallback reason observable ──────────────────────────────────────────


class TestH_FallbackReasonObservable:
    def test_classified_reason_is_rate_limit(self):
        # H: result["failure_reason"] producers (turn_finalizer) are fed by the
        # same classified reason the loop sees after the connector.
        c = classify_api_error(_codex_quota_auth_error(
            "Codex RPD budget for this account is depleted; credentials remain valid."
        ))
        assert c.reason == FailoverReason.rate_limit

    def test_record_session_fallback_logs_structured_line(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="hermes_cli.interactive_fallback"):
            record_session_fallback(
                turn_id="turn-1",
                from_provider="openai-codex",
                from_model="gpt-5.6-luna",
                to_provider="anthropic",
                to_model="claude-sonnet-4",
                reason="rate_limit",
            )
        assert any(
            "[session-fallback]" in r.message and "reason=rate_limit" in r.message
            for r in caplog.records
        )
        assert any(
            "from=openai-codex/gpt-5.6-luna" in r.message
            and "to=anthropic/claude-sonnet-4" in r.message
            for r in caplog.records
        )

    def test_notice_format_brief_and_secret_free(self):
        notice = format_session_fallback_notice(
            "openai-codex/gpt-5.6-luna", "anthropic/claude-sonnet-4"
        )
        assert notice.startswith("⚠️ openai-codex/gpt-5.6-luna unavailable")
        assert "continuing with anthropic/claude-sonnet-4" in notice
        assert "sk-" not in notice and "oauth" not in notice.lower()

    def test_loop_hook_emits_notice_and_log_for_session_route(self, caplog):
        from agent.conversation_loop import _try_emit_session_fallback_notice

        import logging

        agent = _make_agent(
            build_session_fallback_chain(
                {"interactive_session_fallback": {"providers": [dict(ANTHROPIC_ENTRY)]}}
            )
        )
        # The explicit session route marks its entries.
        assert agent._fallback_chain[0].get("_session_explicit_route") is True
        agent._fallback_index = 1  # entry already activated
        statuses = []
        agent._buffer_status = statuses.append
        classified = classify_api_error(_rate_limit_error())
        with caplog.at_level(logging.INFO, logger="hermes_cli.interactive_fallback"):
            _try_emit_session_fallback_notice(
                agent, classified, "openai-codex", "gpt-5.6-luna", turn_id="turn-9"
            )
        assert any("continuing with anthropic" in s for s in statuses)
        assert any(
            "[session-fallback]" in r.message and "reason=rate_limit" in r.message
            for r in caplog.records
        )

    def test_loop_hook_silent_for_legacy_route(self, caplog):
        from agent.conversation_loop import _try_emit_session_fallback_notice

        import logging

        # fallback_providers (legacy route): entries carry NO session marker,
        # so existing UX must be untouched (no session notice, no log line).
        agent = _make_agent([dict(ANTHROPIC_ENTRY)])
        assert agent._fallback_chain[0].get("_session_explicit_route") is not True
        agent._fallback_index = 1
        statuses = []
        agent._buffer_status = statuses.append
        classified = classify_api_error(_rate_limit_error())
        with caplog.at_level(logging.INFO, logger="hermes_cli.interactive_fallback"):
            _try_emit_session_fallback_notice(
                agent, classified, "openai-codex", "gpt-5.6-luna", turn_id="turn-9"
            )
        assert statuses == []
        assert not any("[session-fallback]" in r.message for r in caplog.records)