"""Tests for hermes_cli/fallback_config.py — fallback entry API-key resolution."""

from agent.secret_scope import reset_secret_scope, set_secret_scope
from hermes_cli.fallback_config import resolve_entry_api_key


class TestResolveEntryApiKey:
    def test_inline_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"provider": "custom", "api_key": "inline-key", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "inline-key"


    def test_no_key_fields_returns_none(self):
        assert resolve_entry_api_key({"provider": "openrouter", "model": "glm"}) is None


    def test_whitespace_inline_key_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"api_key": "   ", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "env-key"

    def test_key_env_resolves_from_active_secret_scope_not_raw_env(self, monkeypatch):
        # Multiplexed gateway: os.environ holds another profile's key, but the
        # active per-turn secret scope holds this profile's key. The scoped
        # value must win — a raw os.getenv() would leak the other profile's
        # credential (issue #74311).
        monkeypatch.setenv("FB_KEY", "fake-other-profile-key")
        token = set_secret_scope({"FB_KEY": "fake-active-profile-key"})
        try:
            assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "fake-active-profile-key"
        finally:
            reset_secret_scope(token)

    def test_key_env_falls_back_to_env_when_no_active_scope(self, monkeypatch):
        # Non-multiplexed / single-profile behavior must be unchanged: with no
        # secret scope installed, resolution still reads os.environ.
        monkeypatch.setenv("FB_KEY", "env-key")
        assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "env-key"


# ── INTERACTIVE_SESSION_FALLBACK_V1 ────────────────────────────────────────
# build_session_fallback_chain (hermes_cli/interactive_fallback.py): session
# chain seeding precedence — explicit interactive_session_fallback.providers,
# env override, then the legacy fallback_providers chain. NEVER edits policy.

from hermes_cli.interactive_fallback import build_session_fallback_chain

FALLBACK_PROVIDERS_CFG = {
    "fallback_providers": [{"provider": "openrouter", "model": "deepseek/deepseek-v4-flash-0731"}],
}
SESSION_ROUTE_CFG = {
    "interactive_session_fallback": {
        "enabled": True,
        "providers": [{"provider": "anthropic", "model": "claude-sonnet-4"}],
    },
    **FALLBACK_PROVIDERS_CFG,
}


class TestBuildSessionFallbackChain:
    def test_no_session_config_returns_legacy_chain_exactly(self):
        chain = build_session_fallback_chain(FALLBACK_PROVIDERS_CFG)
        assert chain == [
            {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash-0731"}
        ]
        # Legacy route entries must NOT carry the session-explicit marker.
        assert chain[0].get("_session_explicit_route") is not True

    def test_explicit_session_providers_win_over_fallback_providers(self):
        chain = build_session_fallback_chain(SESSION_ROUTE_CFG)
        assert [e["provider"] for e in chain] == ["anthropic"]
        assert chain[0].get("_session_explicit_route") is True

    def test_empty_session_providers_falls_back_to_legacy(self):
        cfg = {"interactive_session_fallback": {"providers": []}, **FALLBACK_PROVIDERS_CFG}
        chain = build_session_fallback_chain(cfg)
        assert [e["provider"] for e in chain] == ["openrouter"]

    def test_master_switch_off_falls_back_to_legacy(self, monkeypatch):
        monkeypatch.delenv("HERMES_INTERACTIVE_FALLBACK", raising=False)
        cfg = {"interactive_session_fallback": {"enabled": False, "providers": [{"provider": "anthropic", "model": "claude-sonnet-4"}]}, **FALLBACK_PROVIDERS_CFG}
        chain = build_session_fallback_chain(cfg)
        assert [e["provider"] for e in chain] == ["openrouter"]

    def test_env_master_switch_disables_session_route(self, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE_FALLBACK", "0")
        chain = build_session_fallback_chain(SESSION_ROUTE_CFG)
        assert [e["provider"] for e in chain] == ["openrouter"]
        monkeypatch.delenv("HERMES_INTERACTIVE_FALLBACK")

    def test_env_provider_override_replaces_config(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_INTERACTIVE_FALLBACK_PROVIDERS",
            '[{"provider": "nous", "model": "hermes-3"}]',
        )
        chain = build_session_fallback_chain(SESSION_ROUTE_CFG)
        assert [e["provider"] for e in chain] == ["nous"]
        assert chain[0].get("_session_explicit_route") is True
        monkeypatch.delenv("HERMES_INTERACTIVE_FALLBACK_PROVIDERS")

    def test_invalid_env_override_ignored(self, monkeypatch, caplog):
        monkeypatch.setenv("HERMES_INTERACTIVE_FALLBACK_PROVIDERS", "not-json{{")
        chain = build_session_fallback_chain(SESSION_ROUTE_CFG)
        # Falls through to the explicit session route (still enabled).
        assert [e["provider"] for e in chain] == ["anthropic"]
        monkeypatch.delenv("HERMES_INTERACTIVE_FALLBACK_PROVIDERS")

    def test_always_list_safe(self):
        assert build_session_fallback_chain(None) == []
        assert build_session_fallback_chain({}) == []
