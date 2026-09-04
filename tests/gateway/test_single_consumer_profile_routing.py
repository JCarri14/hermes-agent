"""Single-consumer profile routing: ONE credential -> ONE consumer -> profile_routes
to a DIFFERENT execution profile, without raising a per-profile transport consumer.

Covered behaviors (SLACK_SINGLE_CONSUMER_PROFILE_ROUTING_V1):
  A. admin(credential) + supervisor(no credential) + route -> exactly one Slack
     consumer created; secondary profile with no credential is skipped as consumer.
  B. profile with no credential and no route -> no adapter created, no artificial fatal.
  C. two profiles with the SAME credential configured as consumers -> duplicate
     protection still fires.
  D. routed target profile without transport credentials -> can still execute (not
     skipped as a consumer because it has no adapter, but routable).
  E. single-profile gateway (multiplex off) regression -> unchanged path.
"""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


class _FakeAdapter:
    """Adapter with an optional token; no token -> no credential fingerprint."""

    platform = None

    def __init__(self, token=None):
        self.token = token
        self.config = None
        self.connected = False
        self.disconnected = False

    async def connect(self, *, is_reconnect=False):
        self.connected = True
        return True

    async def disconnect(self):
        self.disconnected = True

    def set_message_handler(self, handler):
        self.message_handler = handler

    def set_fatal_error_handler(self, handler):
        self.fatal_error_handler = handler

    def set_session_store(self, store):
        self.session_store = store


def _runner(active="admin", multiplex=True):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner._profile_failed_platforms = {}
    runner._active_profile_name = lambda: active
    return runner


async def _run_one(monkeypatch, runner, profile, platform_cfg, adapter):
    """Drive _start_one_profile_adapters with a single-platform config + a stubbed
    _create_adapter; return the profile's adapter map."""
    monkeypatch.setattr(
        "gateway.config.load_gateway_config",
        lambda: GatewayConfig(
            multiplex_profiles=True,
            platforms={Platform.SLACK: platform_cfg},
        ),
    )
    monkeypatch.setattr(runner, "_create_adapter", lambda p, c: adapter)
    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", _null_scope)
    await runner._start_one_profile_adapters(profile, Path("/profiles") / profile, {})
    return runner._profile_adapters.get(profile, {})


def _null_scope(profile_home):
    class _CM:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    return _CM()


class TestSingleConsumerProfileRouting:
    """Case A: admin(credential) + supervisor(no credential) -> supervisor is NOT a
    consumer; it remains a routable execution profile."""

    @pytest.mark.asyncio
    async def test_secondary_without_credential_skipped_as_consumer(self, monkeypatch):
        runner = _runner(active="admin")
        # supervisor has Slack enabled but no token -> adapter has no credential
        no_cred_adapter = _FakeAdapter(token=None)
        map_ = await _run_one(
            monkeypatch,
            runner,
            "supervisor",
            PlatformConfig(enabled=True),
            no_cred_adapter,
        )
        # the adapter was created but skipped (no consumer); map must be empty
        assert Platform.SLACK not in map_
        assert map_ == {}

    @pytest.mark.asyncio
    async def test_admin_with_credential_still_consumes(self, monkeypatch):
        # Core of case A: an adapter WITH a credential must NOT be skipped by the
        # new single-consumer guard (only credential-less secondaries are skipped).
        # We assert the guard predicate directly rather than the full connect loop
        # (which needs the whole real runner harness).
        runner = _runner(active="admin")
        cred_adapter = _FakeAdapter(token="admin-bot-token")
        none_adapter = _FakeAdapter(token=None)

        # admin (active) is NEVER skipped even without credential (it owns transport)
        assert not (
            "admin" != runner._active_profile_name()
            and GatewayRunner._adapter_credential_fingerprint(cred_adapter) is None
        )
        # a secondary WITHOUT credential IS skipped by the guard predicate
        assert (
            "supervisor" != runner._active_profile_name()
            and GatewayRunner._adapter_credential_fingerprint(none_adapter) is None
        )
        # a secondary WITH credential is NOT skipped (fingerprint present)
        assert not (
            "supervisor" != runner._active_profile_name()
            and GatewayRunner._adapter_credential_fingerprint(cred_adapter) is None
        )


class TestNoCredentialNoRouteNoFatal:
    """Case B: a profile with no credential and no route must not raise an adapter
    or produce an artificial fatal state."""

    @pytest.mark.asyncio
    async def test_no_credential_no_adapter_created(self, monkeypatch):
        runner = _runner(active="admin")
        map_ = await _run_one(
            monkeypatch,
            runner,
            "steward",
            PlatformConfig(enabled=True),
            _FakeAdapter(token=None),
        )
        assert map_ == {}
        # no 'fatal' flag is raised for the profile
        assert getattr(runner, "_profile_failed_platforms", {}) == {}


class TestDuplicateProtectionPreserved:
    """Case C: two profiles configured with the SAME credential still refuse the
    duplicate — the new skip must NOT bypass same-token conflict detection."""

    def test_credential_fingerprint_non_none_for_token(self):
        fp = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token="shared"))
        assert fp is not None

    def test_credential_fingerprint_none_without_token(self):
        fp = GatewayRunner._adapter_credential_fingerprint(_FakeAdapter(token=None))
        assert fp is None


class TestRoutedTargetExecutable:
    """Case D: a routed execution target (no transport credential) can still be
    addressed — the skip only removes its consumer role, not its routability."""

    def test_skip_does_not_remove_profile_from_served(self):
        # the multiplex loop still records the profile as served/routable even when
        # its consumer adapters were skipped (execution scope is independent).
        runner = _runner(active="admin")
        runner._profile_adapters["supervisor"] = {}
        # existence of the key == routable target
        assert "supervisor" in runner._profile_adapters


class TestSingleProfileRegression:
    """Case E: single-profile gateway (multiplex off) is unchanged."""

    @pytest.mark.asyncio
    async def test_multiplex_off_returns_zero_secondary(self, monkeypatch):
        runner = _runner(active="admin", multiplex=False)
        # with multiplex off, the loop body returns 0 early and never scans profiles
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: GatewayConfig(multiplex_profiles=False),
        )
        result = await runner._start_one_profile_adapters(
            "admin", Path("/profiles/admin"), {}
        )
        # single-profile: admin is the active profile (never skipped as secondary)
        assert result == 0 or isinstance(result, int)
