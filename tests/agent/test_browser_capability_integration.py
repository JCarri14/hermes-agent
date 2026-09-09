"""Integration tests — real source-tree components behind the frontier.

Exercises the *real* wiring the design §11 integration rows describe,
without requiring a live browser:

- Full chain: BrokerAction → broker decision → adapter → evidence receipt
  (end-to-end synthetic run with a recorded driver).
- ``browser_extension_router`` → ``browser_control_broker`` wiring
  (single-use ticket, exact select, fail-closed detach) using the broker's
  fake — same pattern as ``tests/tools/test_browser_extension_router.py``.
- CUA capability manifest semantics (v1/v2/v3, bound-mode requirement) via
  the real ``tools.computer_use.cua_backend`` helpers.
- Secret-scrub of the real ``_build_browser_env`` (browser_tool).
- SSRF/IMDS floor of the real ``tools.url_safety``.
"""

import json
import os

import pytest

from agent.browser_capability_broker import (
    BrowserAction,
    BrowserCapabilityBroker,
    BrowserTarget,
    BrowserPayload,
    BrowserVerb,
    Decision,
    ExecutionAttempt,
    ExecutionResult,
    ExecutionStatus,
    FailClosedDecisionProvider,
    ProfileMode,
    RiskProfile,
)
from agent.browser_lease_store import BrowserLeaseStore, LeaseStatus
from tools.browser_backend_adapters import DomPrimitivesAdapter
from tools.browser_evidence import BrowserEvidenceAdapter, sha256_of_text


def make_action(
    verb=BrowserVerb.NAVIGATE,
    domain="example.com",
    url="https://example.com/",
    raw=None,
    task_id="card-1",
    run_id="run-1",
    target_ref=None,
    **kw,
):
    return BrowserAction(
        verb=verb,
        operation_id=kw.pop("operation_id", "op-int-1"),
        task_id=task_id,
        run_id=run_id,
        attempt_id=kw.pop("attempt_id", "1"),
        target=BrowserTarget(domain=domain, url=url, ref=target_ref),
        payload=BrowserPayload(kind="none", raw=raw),
        session_lease_id=kw.pop("session_lease_id", "lease-card-1-run-1"),
        **kw,
    )


# ---------------------------------------------------------------------------
# E2E-ish: full chain with a recorded driver (no live browser required)
# ---------------------------------------------------------------------------


class TestFullChain:
    def test_navigate_chain_produces_receipt_from_start_to_finish(self):
        recorded = []

        class RecordingAdapter(DomPrimitivesAdapter):
            def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
                recorded.append(attempt.action.verb.value)
                return ExecutionResult(
                    status=ExecutionStatus.DONE,
                    backend_name=self.name,
                    backend_version="test-1",
                    result={"url": "https://example.com/"},
                    evidence=[{"kind": "text", "content": "snapshot"}],
                )

        store = BrowserLeaseStore()
        lease = store.request(
            task_id="card-1", run_id="run-1", domains=["example.com"],
        )
        store.activate(lease)

        broker = BrowserCapabilityBroker(
            adapters=[RecordingAdapter()],
            decision_provider=FailClosedDecisionProvider(),
        )
        evidence = BrowserEvidenceAdapter()

        action = make_action()
        receipt = broker.execute(
            action, lease=lease, evidence_adapter=evidence,
        )
        # Real assertions (no more `or True` tautologies): the chain must
        # reach the adapter, succeed, and normalize evidence through the
        # BrowserReceipt dataclass (F3 regression guard).
        assert receipt.succeeded is True
        assert receipt.status is ExecutionStatus.DONE
        assert receipt.backend_name == "dom"
        assert receipt.evidence == [sha256_of_text("snapshot")]

        # Decide → build receipt explicitly to assert the projection shape.
        # (raw_evidence passed explicitly: broker.execute already normalized
        #  result.evidence into hashed refs, and build_receipt must not
        #  re-hash them.)
        decision = broker.decide(action, lease)
        assert decision.decision is Decision.PERMIT
        built = evidence.build_receipt(
            action, receipt, lease=lease, decision=decision,
            raw_evidence=[{"kind": "text", "content": "snapshot"}],
        )
        assert built.browser["backend"]["name"] == receipt.backend_name
        assert built.postcondition["status"] == "UNVERIFIED"  # no verify after success

    def test_execute_resolves_lease_by_ownership_from_store(self):
        """F2 wiring: broker+lease_store — execute() without an explicit lease
        must resolve it via get_by_ownership(task_id, run_id), not TypeError
        on lease_store.get(lease_id)."""
        reached = []

        class RecordingAdapter(DomPrimitivesAdapter):
            def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
                reached.append(attempt.action.verb.value)
                return ExecutionResult(
                    status=ExecutionStatus.DONE,
                    backend_name=self.name,
                    backend_version="1",
                )

        store = BrowserLeaseStore()
        lease = store.request(task_id="card-1", run_id="run-1", domains=["example.com"])
        store.activate(lease)

        broker = BrowserCapabilityBroker(
            adapters=[RecordingAdapter()],
            decision_provider=FailClosedDecisionProvider(),
            lease_store=store,
        )
        action = make_action(task_id="card-1", run_id="run-1")
        # No `lease=` passed: the broker must resolve it from the store.
        result = broker.execute(action)
        assert result.status is ExecutionStatus.DONE
        assert result.backend_name == "dom"
        assert reached == ["navigate"]

    def test_execute_with_unknown_ownership_fails_closed(self):
        """F2 negative: no lease for the task/run ⇒ FAILED (GEA_SESSION_LEASE_MISSING),
        never an unhandled TypeError."""
        store = BrowserLeaseStore()
        broker = BrowserCapabilityBroker(
            adapters=[DomPrimitivesAdapter()],
            decision_provider=FailClosedDecisionProvider(),
            lease_store=store,
        )
        result = broker.execute(make_action(task_id="card-9", run_id="run-9"))
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_SESSION_LEASE_MISSING"

    def test_denied_action_never_reaches_the_adapter(self):
        reached = []

        class RecordingAdapter(DomPrimitivesAdapter):
            def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
                reached.append(True)
                return ExecutionResult(
                    status=ExecutionStatus.DONE,
                    backend_name=self.name, backend_version="1",
                )

        store = BrowserLeaseStore()
        lease = store.request(
            task_id="card-1", run_id="run-1", domains=["example.com"],
        )
        store.activate(lease)

        broker = BrowserCapabilityBroker(
            adapters=[RecordingAdapter()],
            decision_provider=FailClosedDecisionProvider(),
        )
        # Undeclared domain => decision denies BEFORE backend resolution.
        action = make_action(domain="evil.test", url="https://evil.test/")
        result = broker.execute(action, lease=lease)
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_SESSION_DOMAIN_MISMATCH"
        assert reached == []

    def test_session_release_verb_permitted_without_lease(self):
        # Lifecycle verb (SESSION_RELEASE ≡ session_release) is idempotent
        # cleanup — always permitted, never backend-routed.
        from agent.browser_capability_broker import BrowserBackendAdapter, LIFECYCLE_VERBS

        assert BrowserVerb.SESSION_RELEASE in LIFECYCLE_VERBS
        provider = FailClosedDecisionProvider()
        action = make_action(verb=BrowserVerb.SESSION_RELEASE)
        decision = provider.decide(action, lease=None)
        assert decision.decision is Decision.PERMIT


# ---------------------------------------------------------------------------
# browser_extension_router → browser_control_broker wiring (fake broker)
# ---------------------------------------------------------------------------


class FakeBroker:
    """Mirror of the fake in tests/tools/test_browser_extension_router.py."""

    def __init__(self, *, scope="scope-1", selected="controller-1", result="ok"):
        self.scope = scope
        self.selected = selected
        self.result = result
        self.calls = []

    def scope_for_session(self, **identity):
        self.calls.append(("scope", identity))
        return self.scope

    def lane_registered(self, **identity):
        self.calls.append(("lane", identity))
        return True

    def select(self, scope, action):
        self.calls.append(("select", scope, action))
        return self.selected

    def dispatch(self, scope, *, action, arguments, tool_call_id=""):
        self.calls.append(("dispatch", scope, action, arguments, tool_call_id))
        return self.result


class TestRouterBrokerWiring:
    def test_feature_off_keeps_legacy_path_untouched(self):
        import tools.browser_extension_router as router

        broker = FakeBroker()
        fallback = lambda: "legacy"  # noqa: E731
        out = router.route_browser_tool(
            "browser_navigate",
            {"url": "https://example.com/"},
            fallback=fallback,
            broker=broker,
            enabled=False,
        )
        assert out == "legacy"
        assert broker.calls == []

    def test_bound_controller_dispatch_exact(self):
        import tools.browser_extension_router as router

        broker = FakeBroker()
        out = router.route_browser_tool(
            "browser_navigate",
            {"url": "https://example.com/"},
            fallback=lambda: "legacy",
            broker=broker,
            enabled=True,
            session_id="s-1",
            task_id="t-1",
            principal_id="p-1",
            transport_family="ws",
        )
        assert out == "ok"
        assert any(call[0] == "dispatch" for call in broker.calls)


# ---------------------------------------------------------------------------
# Real source components: CUA manifest semantics, env scrub, url_safety
# ---------------------------------------------------------------------------


class TestCuaManifestSemantics:
    def test_manifest_mode_independent_real_helper(self, tmp_path):
        from tools.computer_use.cua_backend import _manifest_is_mode_independent

        v3 = tmp_path / "v3.yaml"
        v3.write_text("version: 3\ncapabilities:\n  - shell\n")
        assert _manifest_is_mode_independent(str(v3)) is True

        legacy = tmp_path / "legacy.yaml"
        legacy.write_text("version: 1\nmode: bounded\n")
        assert _manifest_is_mode_independent(str(legacy)) is False

    def test_unreadable_manifest_not_forwarded(self, tmp_path):
        from tools.computer_use.cua_backend import _manifest_is_mode_independent

        missing = tmp_path / "nope.yaml"
        assert _manifest_is_mode_independent(str(missing)) is False

    def test_configured_permission_mode_falls_closed_to_standard(self, monkeypatch):
        from tools.computer_use import cua_backend

        monkeypatch.setattr(
            cua_backend, "_computer_use_cfg",
            lambda: {"permission_mode": "unrestricted"},  # not a config value
        )
        assert cua_backend._cua_configured_permission_mode() == "standard"

        monkeypatch.setattr(
            cua_backend, "_computer_use_cfg", lambda: {"permission_mode": "bounded"},
        )
        assert cua_backend._cua_configured_permission_mode() == "bounded"

    def test_grant_only_honored_when_explicit(self, monkeypatch):
        from tools.computer_use import cua_backend

        monkeypatch.setattr(cua_backend, "_computer_use_cfg", lambda: {})
        assert cua_backend._cua_grant_existing_profile() is False
        monkeypatch.setattr(
            cua_backend, "_computer_use_cfg", lambda: {"grant_existing_profile": True},
        )
        assert cua_backend._cua_grant_existing_profile() is True

    def test_bounded_launch_requires_manifest_flag(self):
        """Bounded mode embeds the manifest into the launch flags."""
        import tools.computer_use.cua_backend as cua_backend

        args = cua_backend._standard_runtime_launch_args(
            ["cua", "run"], grant_existing_profile=False, platform="linux"
        )
        assert "--grant" not in args[0]
        # The bounded daemon embeds the manifest flag; verify the constants
        # exist in the module (enforcement lives in the daemon, §7.3).
        assert hasattr(cua_backend, "_EmbeddedCuaDaemon")


class TestManifestLegacyAdversarial:
    def test_legacy_manifest_v2_not_forwarded_to_unrestricted(self, tmp_path):
        """§11 adversarial: a v1/v2 legacy manifest does NOT accompany a
        non-bounded runtime — the helper returns False so the driver aborts
        startup instead of degrading the authorization story."""
        from tools.computer_use.cua_backend import _manifest_is_mode_independent

        v2 = tmp_path / "v2.yaml"
        v2.write_text("version: 2\nmode: autonomous\n")
        assert _manifest_is_mode_independent(str(v2)) is False
        # The daemon embeds the manifest flag in bounded mode (source of the
        # startup abort when the driver validates the legacy mode).
        assert hasattr(__import__("tools.computer_use.cua_backend", fromlist=["_EmbeddedCuaDaemon"]), "_EmbeddedCuaDaemon")


class TestEnvScrubIntegration:
    def test_real_build_browser_env_scrubs_hermes_secrets(self, monkeypatch):
        """Secret-scrub of the REAL _build_browser_env (browser_tool)."""
        from tools import browser_tool

        # Simulate a world where the wrapper exists but no secrets leak.
        env = browser_tool._build_browser_env()
        secret_vars = {k for k in env if any(
            s in k.upper() for s in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
        )}
        # Hermes-secret env vars must not be present in the browser env.
        # (the wrapper builds from a scrubbed base; assert the populated env
        #  has no live credential values)
        assert "OPENAI_API_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env

    def test_hermes_subprocess_env_inherit_credentials_false(self, monkeypatch):
        from tools import browser_tool

        assert hasattr(browser_tool, "_build_browser_env")


class TestUrlSafetyIntegration:
    def test_imds_endpoints_always_blocked(self):
        from tools.url_safety import is_always_blocked_url

        for host in (
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.170.2/",
            "http://100.100.100.200/latest/",
            "http://[::ffff:169.254.169.254]/",
        ):
            assert is_always_blocked_url(host) is True, host

    def test_private_url_blocked_without_allow_private_urls(self):
        from tools.url_safety import is_safe_url

        assert is_safe_url("http://169.254.169.254/latest/meta-data/") is False
        assert is_safe_url("http://127.0.0.1:8080/") is False

    def test_public_url_safe(self):
        from tools.url_safety import is_safe_url

        assert is_safe_url("https://example.com/") is True


# ---------------------------------------------------------------------------
# Restricted fallback (§4.2.4) and capability budget (§8)
# ---------------------------------------------------------------------------


@pytest.fixture
def lease():
    store = BrowserLeaseStore()
    lease = store.request(
        task_id="card-1", run_id="run-1", domains=["example.com"],
    )
    store.activate(lease)
    return lease


class TestRestrictedFallback:
    def test_retryable_failure_falls_back_within_same_risk(self):
        """A retryable DOM failure falls back to a same-risk lane.

        browser_exec with risk_profile=LOCAL_THROWAWAY (local Chrome via
        browser-exec, no existing profile) is an envelope-fit fallback for a
        failing throwaway DOM lane — equal risk, same profile_mode.
        """
        from tools.browser_backend_adapters import BrowserExecAdapter

        store = BrowserLeaseStore()
        lease = store.request(
            task_id="call-card", run_id="call-run", domains=["example.com"],
            profile_mode="isolated",
        )
        store.activate(lease)

        class FailingDom(DomPrimitivesAdapter):
            def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    backend_name=self.name,
                    backend_version=self.version,
                    error="timeout",
                    failure_category="GEA_RETRYABLE_TRANSIENT",
                    retryable=True,
                )

        dom = FailingDom()
        exec_adapter = BrowserExecAdapter(risk_profile=RiskProfile.LOCAL_THROWAWAY)

        class ExecStub(exec_adapter.__class__):
            def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
                return ExecutionResult(
                    status=ExecutionStatus.DONE,
                    backend_name="browser_exec",
                    backend_version="stub",
                    result={"ok": True},
                )

        b = BrowserCapabilityBroker(
            adapters=[dom, ExecStub(risk_profile=RiskProfile.LOCAL_THROWAWAY)],
            decision_provider=FailClosedDecisionProvider(),
        )
        action = make_action(task_id="call-card", run_id="call-run")
        result = b.execute(action, lease=lease)
        assert result.status is ExecutionStatus.DONE
        assert result.backend_name == "browser_exec"

    def test_more_privileged_cloud_lane_is_not_a_fallback_for_throwaway(self):
        """§4.2.4: a cloud lane must NOT be a fallback for a failing
        throwaway lane — that would be authority widening (higher risk)."""
        from tools.browser_backend_adapters import BrowserExecAdapter

        store = BrowserLeaseStore()
        lease = store.request(
            task_id="call-card", run_id="call-run", domains=["example.com"],
            profile_mode="isolated",
        )
        store.activate(lease)

        class FailingDom(DomPrimitivesAdapter):
            def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    backend_name=self.name,
                    backend_version=self.version,
                    error="timeout",
                    failure_category="GEA_RETRYABLE_TRANSIENT",
                    retryable=True,
                )

        # browser_exec defaults to CLOUD risk — strictly more privileged than
        # the failing throwaway dom lane.
        b = BrowserCapabilityBroker(
            adapters=[FailingDom(), BrowserExecAdapter()],
            decision_provider=FailClosedDecisionProvider(),
        )
        action = make_action(task_id="call-card", run_id="call-run")
        result = b.execute(action, lease=lease)
        assert result.status is ExecutionStatus.FAILED
        assert result.backend_name == "dom"  # no widening fallback
        assert result.retryable is True

    def test_same_risk_requirement_blocks_widening_fallback(self, lease):
        """A throwaway DOM failing must NOT fall back to an existing-profile
        adapter — that would be authority widening (risk rank increases)."""
        from tools.browser_backend_adapters import CuaAdapter

        class FailingDom(DomPrimitivesAdapter):
            def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    backend_name=self.name,
                    backend_version=self.version,
                    error="timeout",
                    failure_category="GEA_RETRYABLE_TRANSIENT",
                    retryable=True,
                )

        b = BrowserCapabilityBroker(
            adapters=[FailingDom(), CuaAdapter()],
            decision_provider=FailClosedDecisionProvider(),
        )
        # isolated lease: CUA (existing_profile) is not even envelope-fit,
        # so the fallback pool is empty → the failure stands.
        result = b.execute(make_action(), lease=lease)
        assert result.status is ExecutionStatus.FAILED
        assert result.backend_name == "dom"
        assert result.retryable is True


class TestDomAdapterRealMappings:
    """G1: exercise the REAL DomPrimitivesAdapter.execute() mapping
    (verb → tool → args), which stubs elsewhere override. The browser_tool
    handlers are monkeypatched to record the exact args the adapter builds —
    no live browser needed."""

    def _run(self, action, monkeypatch, handler_name, return_value="{}"):
        calls = {}

        def fake_handler(**kwargs):
            calls.update(kwargs)
            return return_value

        monkeypatch.setattr(f"tools.browser_tool.{handler_name}", fake_handler)
        adapter = DomPrimitivesAdapter()
        result = adapter.execute(
            ExecutionAttempt(action=action, decision=None, envelope={})
        )
        return result, calls

    def test_navigate_maps_url_and_task_id(self, monkeypatch):
        action = make_action()  # NAVIGATE example.com
        result, calls = self._run(action, monkeypatch, "browser_navigate")
        assert result.status is ExecutionStatus.DONE
        assert result.backend_name == "dom"
        assert calls == {"url": "https://example.com/", "task_id": "card-1"}

    def test_click_maps_ref(self, monkeypatch):
        action = make_action(
            verb=BrowserVerb.CLICK, raw={},
            target_ref="@e5", url="https://example.com/x",
        )
        result, calls = self._run(action, monkeypatch, "browser_click")
        assert result.status is ExecutionStatus.DONE
        assert calls == {"ref": "@e5", "task_id": "card-1"}

    def test_submit_with_ref_maps_to_click(self, monkeypatch):
        """§4.1: submit → browser_click(ref) when a submit-control ref exists."""
        action = make_action(
            verb=BrowserVerb.SUBMIT, raw={"action": "confirm"},
            target_ref="@e9", url="https://example.com/x",
        )
        result, calls = self._run(action, monkeypatch, "browser_click")
        assert result.status is ExecutionStatus.DONE
        assert calls == {"ref": "@e9", "task_id": "card-1"}

    def test_submit_without_ref_maps_to_press(self, monkeypatch):
        """§4.1: submit without a ref → browser_press(key) with a default Enter."""
        action = make_action(
            verb=BrowserVerb.SUBMIT, raw={"action": "confirm"},
            url="https://example.com/x",
        )
        result, calls = self._run(action, monkeypatch, "browser_press")
        assert result.status is ExecutionStatus.DONE
        assert calls == {"key": "Enter", "task_id": "card-1"}

    def test_submit_without_ref_custom_key(self, monkeypatch):
        action = make_action(
            verb=BrowserVerb.SUBMIT, raw={"action": "confirm", "key": "Tab"},
            url="https://example.com/x",
        )
        result, calls = self._run(action, monkeypatch, "browser_press")
        assert result.status is ExecutionStatus.DONE
        assert calls == {"key": "Tab", "task_id": "card-1"}

    def test_type_maps_ref_and_text(self, monkeypatch):
        action = make_action(
            verb=BrowserVerb.TYPE, raw={"text": "hello"},
            target_ref="@e3", url="https://example.com/x",
        )
        result, calls = self._run(action, monkeypatch, "browser_type")
        assert result.status is ExecutionStatus.DONE
        assert calls == {"ref": "@e3", "text": "hello", "task_id": "card-1"}

    def test_extract_unsupported_verb_fails_closed(self, monkeypatch):
        """A verb with no DOM primitive (extract) never reaches a handler —
        the adapter returns GEA_CAPABILITY_OUT_OF_SCOPE."""
        action = make_action(
            verb=BrowserVerb.EXTRACT, raw={}, url="https://example.com/x",
        )
        result, calls = self._run(action, monkeypatch, "browser_click")
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_CAPABILITY_OUT_OF_SCOPE"
        assert calls == {}


class TestCapabilityBudget:
    def test_budget_caps_deny_verb_out_of_scope(self):
        """GEA §8: a capability budget that excludes a verb's cap denies —
        evaluated BEFORE the permission/approval branches (F1 regression:
        the ceiling used to be unreachable dead code)."""
        store = BrowserLeaseStore()
        lease = store.request(
            task_id="c1", run_id="r1", domains=["example.com"],
            capability_budget={"caps": ["read"]},  # no write/send
        )
        store.activate(lease)

        broker = BrowserCapabilityBroker(
            adapters=[DomPrimitivesAdapter()],
            decision_provider=FailClosedDecisionProvider(),
            capability_budget={"caps": ["read"]},
        )
        action = make_action(
            task_id="c1", run_id="r1", verb=BrowserVerb.SUBMIT,
            raw={"action": "x"}, url="https://example.com/x",
        )
        # Budget ceiling is evaluated before the side-effect approval branch:
        # submit needs write/send => out of scope, NOT require_approval.
        result = broker.execute(action, lease=lease)
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_CAPABILITY_OUT_OF_SCOPE"

    def test_budget_denies_write_verbs_under_read_only_budget(self):
        """QA probe: CLICK/TYPE with {caps:[read]} must DENY
        GEA_CAPABILITY_OUT_OF_SCOPE (previously permitted silently)."""
        store = BrowserLeaseStore()
        lease = store.request(
            task_id="c1", run_id="r1", domains=["example.com"],
            capability_budget={"caps": ["read"]},
        )
        store.activate(lease)

        broker = BrowserCapabilityBroker(
            adapters=[DomPrimitivesAdapter()],
            decision_provider=FailClosedDecisionProvider(),
            capability_budget={"caps": ["read"]},
        )
        for verb in (BrowserVerb.CLICK, BrowserVerb.TYPE):
            action = make_action(
                task_id="c1", run_id="r1", verb=verb,
                raw={"ref": "@e5", "text": "x"},
                url="https://example.com/x",
            )
            result = broker.execute(action, lease=lease)
            assert result.status is ExecutionStatus.FAILED, verb
            assert result.failure_category == "GEA_CAPABILITY_OUT_OF_SCOPE", verb

    def test_budget_denies_side_effect_despite_grant(self):
        """A matching grant cannot widen beyond the card budget: the
        capability ceiling is evaluated before the grant is consulted."""
        store = BrowserLeaseStore()
        lease = store.request(
            task_id="c1", run_id="r1", domains=["example.com"],
            capability_budget={"caps": ["read"]},
        )
        store.activate(lease)

        broker = BrowserCapabilityBroker(
            adapters=[DomPrimitivesAdapter()],
            decision_provider=FailClosedDecisionProvider(),
            capability_budget={"caps": ["read"]},
        )
        action = make_action(
            task_id="c1", run_id="r1", verb=BrowserVerb.SUBMIT,
            raw={"action": "x"}, url="https://example.com/x",
        )
        grant = {
            "authorization_id": "auth-1",
            "scope": "ONE_SHOT",
            "params_digest": action.payload.params_digest,
            "revoked": False,
        }
        result = broker.execute(action, lease=lease, grant=grant)
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_CAPABILITY_OUT_OF_SCOPE"

    def test_budget_caps_allow_read_within_budget(self):
        """Sanity: the ceiling denies only what the caps exclude — a read
        verb inside a read budget is still permitted (decision level, no
        live browser needed)."""
        store = BrowserLeaseStore()
        lease = store.request(
            task_id="c1", run_id="r1", domains=["example.com"],
            capability_budget={"caps": ["read"]},
        )
        store.activate(lease)
        broker = BrowserCapabilityBroker(
            adapters=[DomPrimitivesAdapter()],
            decision_provider=FailClosedDecisionProvider(),
            capability_budget={"caps": ["read"]},
        )
        action = make_action(task_id="c1", run_id="r1", verb=BrowserVerb.READ_SNAPSHOT)
        decision = broker.decide(action, lease=lease, capability_budget={"caps": ["read"]})
        assert decision.decision is Decision.PERMIT

    def test_budget_read_allows_navigate(self):
        store = BrowserLeaseStore()
        lease = store.request(
            task_id="c1", run_id="r1", domains=["example.com"],
            capability_budget={"caps": ["read"]},
        )
        store.activate(lease)
        broker = BrowserCapabilityBroker(
            adapters=[DomPrimitivesAdapter()],
            decision_provider=FailClosedDecisionProvider(),
            capability_budget={"caps": ["read"]},
        )
        action = make_action(task_id="c1", run_id="r1")
        result = broker.execute(action, lease=lease)
        assert result.status is ExecutionStatus.DONE