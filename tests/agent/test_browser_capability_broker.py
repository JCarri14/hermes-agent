"""Unit / contract / adversarial tests for the browser capability broker.

Covers design §11 test-plan layers:

- Unit: ``BrowserAction`` serialization, stable ``params_digest``, secret
  rejection; adapter ``resolve()`` selection order and envelope; lease-gated
  decisions.
- Contract: ``SESSION ACCESS != ACTION AUTHORIZATION``; side-effect verbs
  default to ``require_approval``; existing-profile never falls back to
  throwaway; extension-lane authority is fail-closed.
- Adversarial: submit without approval → ``GEA_UNAUTHORIZED`` path (never a
  more-privileged backend); revoked lease is terminal; payload digest change
  invalidates a grant; double lease claim denies.
"""

import pytest

from agent.browser_capability_broker import (
    BrowserAction,
    BrowserCapabilityBroker,
    BrowserTarget,
    BrowserPayload,
    BrowserVerb,
    Decision,
    ExecutionResult,
    ExecutionStatus,
    FailClosedDecisionProvider,
    ProfileMode,
    RiskProfile,
    SIDE_EFFECT_VERBS,
    make_operation_id,
    payload_contains_secret_hint,
)
from agent.browser_lease_store import BrowserLeaseStore, LeaseConflictError, LeaseStatus
from tools.browser_backend_adapters import (
    BrowserExecAdapter,
    CuaAdapter,
    DomPrimitivesAdapter,
    ExtensionLaneAdapter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_action(
    verb=BrowserVerb.NAVIGATE,
    domain="example.com",
    url="https://example.com/",
    raw=None,
    **kw,
):
    return BrowserAction(
        verb=verb,
        operation_id=kw.pop("operation_id", make_operation_id(
            "card-1", "run-1", verb, BrowserTarget(domain=domain, url=url), raw
        )),
        task_id=kw.pop("task_id", "card-1"),
        run_id=kw.pop("run_id", "run-1"),
        target=BrowserTarget(domain=domain, url=url),
        payload=BrowserPayload(kind="none", raw=raw),
        session_lease_id=kw.pop("session_lease_id", "lease-card-1-run-1"),
        **kw,
    )


@pytest.fixture
def lease():
    store = BrowserLeaseStore()
    lease = store.request(
        task_id="card-1",
        run_id="run-1",
        domains=["example.com"],
        profile_mode="isolated",
    )
    store.activate(lease)
    return lease


@pytest.fixture
def dom_adapter():
    return DomPrimitivesAdapter()


@pytest.fixture
def exec_adapter():
    return BrowserExecAdapter()


@pytest.fixture
def broker(dom_adapter):
    return BrowserCapabilityBroker(
        adapters=[dom_adapter],
        decision_provider=FailClosedDecisionProvider(),
    )


# ---------------------------------------------------------------------------
# Unit — IR
# ---------------------------------------------------------------------------


class TestBrowserActionIR:
    def test_action_requires_declared_domain(self):
        with pytest.raises(ValueError):
            BrowserAction(
                verb=BrowserVerb.NAVIGATE,
                operation_id="op",
                task_id="t",
                run_id="r",
                target=BrowserTarget(domain=""),
            )

    def test_domain_normalized_lowercase(self):
        action = make_action(domain="Example.COM")
        assert action.target.domain == "example.com"

    def test_params_digest_stable_across_key_order(self):
        a1 = make_action(raw={"a": 1, "b": {"c": 3}})
        a2 = make_action(raw={"b": {"c": 3}, "a": 1})
        assert a1.payload.params_digest == a2.payload.params_digest

    def test_params_digest_changes_with_payload(self):
        a1 = make_action(raw={"a": 1})
        a2 = make_action(raw={"a": 2})
        assert a1.payload.params_digest != a2.payload.params_digest

    def test_payload_secret_hint_rejected(self):
        with pytest.raises(ValueError):
            make_action(raw={"text": "hello", "password": "hunter2"})

    def test_secret_hint_detector(self):
        assert payload_contains_secret_hint({"api_key": "x"})
        assert payload_contains_secret_hint({"Authorization": "Bearer x"})
        assert not payload_contains_secret_hint({"text": "plain"})

    def test_serialization_roundtrip_redacted(self):
        action = make_action(raw={"q": "x"})
        d = action.as_dict()
        assert d["schema"] == "browser_execution.v1.BrowserAction"
        assert d["payload"]["params_digest"] == action.payload.params_digest
        assert "raw" not in d["payload"]  # raw payload never serialized

    def test_unsupported_postcondition_rejected(self):
        from agent.browser_capability_broker import ExpectedPostcondition

        with pytest.raises(ValueError):
            make_action(expected_postcondition=ExpectedPostcondition(type="bogus"))

    def test_operation_id_stable_and_payload_sensitive(self):
        op1 = make_operation_id(
            "card-1", "run-1", BrowserVerb.SUBMIT,
            BrowserTarget(domain="example.com", url="https://example.com/x"),
            {"amount": "10"},
        )
        op2 = make_operation_id(
            "card-1", "run-1", BrowserVerb.SUBMIT,
            BrowserTarget(domain="example.com", url="https://example.com/x"),
            {"amount": "10"},
        )
        op3 = make_operation_id(
            "card-1", "run-1", BrowserVerb.SUBMIT,
            BrowserTarget(domain="example.com", url="https://example.com/x"),
            {"amount": "99"},
        )
        assert op1 == op2
        assert op1 != op3


# ---------------------------------------------------------------------------
# Contract — SESSION ACCESS != ACTION AUTHORIZATION
# ---------------------------------------------------------------------------


class TestSessionAccessVsActionAuthorization:
    def test_navigate_requires_active_lease(self, broker):
        result = broker.execute(
            make_action(),
            task_id="card-1",
            run_id="run-1",
            lease=None,
        )
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_SESSION_LEASE_MISSING"

    def test_navigate_without_lease_store_requires_explicit_lease(self):
        b = BrowserCapabilityBroker(adapters=[DomPrimitivesAdapter()])
        result = b.execute(make_action(), task_id="card-1", run_id="run-1")
        # no lease and no lease_store => the broker cannot know ownership
        assert result.status is ExecutionStatus.FAILED

    def test_navigate_active_lease_permitted_and_executes(self, lease, broker):
        result = broker.execute(make_action(), lease=lease)
        assert result.status is ExecutionStatus.DONE
        assert result.backend_name == "dom"

    def test_submit_with_matching_grant_permitted(self, lease):
        # Grant-gated side effect: the frontier must PERMIT and resolve to a
        # backend. A fake adapter stands in for the real browser surface so
        # the test exercises the frontier, not the browser.
        from agent.browser_capability_broker import ExecutionAttempt

        class FakeSubmitAdapter(DomPrimitivesAdapter):
            def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
                return ExecutionResult(
                    status=ExecutionStatus.DONE,
                    backend_name=self.name,
                    backend_version=self.version,
                    result={"ok": True},
                )

        b = BrowserCapabilityBroker(
            adapters=[FakeSubmitAdapter()],
            decision_provider=FailClosedDecisionProvider(),
        )
        action = make_action(
            verb=BrowserVerb.SUBMIT,
            raw={"action": "confirm"},
            url="https://example.com/checkout",
        )
        grant = {
            "authorization_id": "auth-1",
            "scope": "ONE_SHOT",
            "params_digest": action.payload.params_digest,
            "revoked": False,
        }
        result = b.execute(action, lease=lease, grant=grant)
        assert result.status is ExecutionStatus.DONE
        assert result.backend_name == "dom"

    def test_domain_outside_lease_denied(self, lease, broker):
        action = make_action(domain="evil.example.net", url="https://evil.example.net/")
        result = broker.execute(action, lease=lease)
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_SESSION_DOMAIN_MISMATCH"

    def test_submit_defaults_to_require_approval(self, lease, broker):
        action = make_action(
            verb=BrowserVerb.SUBMIT,
            raw={"action": "confirm"},
            url="https://example.com/checkout",
        )
        result = broker.execute(action, lease=lease)
        assert result.status is ExecutionStatus.INTENT
        assert result.failure_category == "GEA_HUMAN_APPROVAL_REQUIRED"

    def test_submit_grant_payload_mismatch_denied(self, lease, broker):
        action = make_action(
            verb=BrowserVerb.SUBMIT,
            raw={"action": "confirm"},
            url="https://example.com/checkout",
        )
        grant = {
            "authorization_id": "auth-1",
            "scope": "ONE_SHOT",
            "params_digest": "deadbeef" * 8,  # different payload => new authz
            "revoked": False,
        }
        result = broker.execute(action, lease=lease, grant=grant)
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_PAYLOAD_HASH_MISMATCH"

    def test_revoked_grant_denied(self, lease, broker):
        action = make_action(
            verb=BrowserVerb.SUBMIT,
            raw={"action": "confirm"},
            url="https://example.com/checkout",
        )
        grant = {
            "authorization_id": "auth-1",
            "scope": "ONE_SHOT",
            "params_digest": action.payload.params_digest,
            "revoked": True,
        }
        result = broker.execute(action, lease=lease, grant=grant)
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_AUTHORIZATION_REVOKED"

    def test_side_effect_verbs_all_require_approval_by_default(self, lease, broker):
        for verb in SIDE_EFFECT_VERBS:
            action = make_action(verb=verb, raw={"x": "y"})
            result = broker.execute(action, lease=lease)
            assert result.status is ExecutionStatus.INTENT, verb
            assert result.failure_category == "GEA_HUMAN_APPROVAL_REQUIRED", verb


# ---------------------------------------------------------------------------
# Contract — envelope, risk profiles, restricted fallback
# ---------------------------------------------------------------------------


class TestEnvelopeAndFallback:
    def test_existing_profile_never_falls_back_to_throwaway(self):
        # Consent for existing_profile: the fallback pool must NOT contain a
        # throwaway adapter — that would be a silent consent downgrade.
        dom_throwaway = DomPrimitivesAdapter()
        cua = CuaAdapter()  # risk_profile = existing_profile

        # Broker with existing_profile lease: cua lane is envelope-fit; dom
        # (throwaway) is not even selectable, let alone a fallback.
        b = BrowserCapabilityBroker(
            adapters=[dom_throwaway, cua],
            decision_provider=FailClosedDecisionProvider(),
        )
        store = BrowserLeaseStore()
        lease = store.request(
            task_id="c1", run_id="r1", domains=["example.com"],
            profile_mode="existing_profile", consent_ref="interaction-42",
        )
        store.activate(lease)
        action = make_action(task_id="c1", run_id="r1")

        # envelope says existing_profile => throwaway dom adapter is not
        # selectable; resolution must fail closed rather than downgrade.
        selected = b.resolve_backend(action, lease, b.decide(action, lease))
        # cua requires profile_mode existing_profile/cdp_override — it applies
        assert selected is None or selected.name in ("cua",)

    def test_isolated_envelope_rejects_privileged_adapters(self):
        b = BrowserCapabilityBroker(
            adapters=[DomPrimitivesAdapter(), CuaAdapter()],
            decision_provider=FailClosedDecisionProvider(),
        )
        store = BrowserLeaseStore()
        lease = store.request(
            task_id="c1", run_id="r1", domains=["example.com"], profile_mode="isolated",
        )
        store.activate(lease)
        action = make_action(task_id="c1", run_id="r1")

        selected = b.resolve_backend(action, lease, b.decide(action, lease))
        assert selected is not None
        assert selected.name == "dom"  # cua (existing_profile) not envelope-fit

    def test_extension_lane_authoritative_no_fallback(self, lease):
        ext = ExtensionLaneAdapter()

        def selector(action):
            return ext  # a controller is bound for everything

        # A bound controller corresponds to a cdp_override / existing_profile
        # session (the human is operating the browser), not an isolated one.
        store = BrowserLeaseStore()
        cdp_lease = store.request(
            task_id="c1", run_id="r1", domains=["example.com"],
            profile_mode="cdp_override", consent_ref="interaction-7",
        )
        store.activate(cdp_lease)
        action = make_action(task_id="c1", run_id="r1")

        b = BrowserCapabilityBroker(
            adapters=[DomPrimitivesAdapter(), ext],
            decision_provider=FailClosedDecisionProvider(),
            extension_selector=selector,
        )
        selected = b.resolve_backend(action, cdp_lease, b.decide(action, cdp_lease))
        assert selected is not None and selected.name == "extension"

    def test_extension_selector_mismatch_fails_closed(self, lease):
        ext = ExtensionLaneAdapter(capabilities=[BrowserVerb.READ_SNAPSHOT])

        def selector(action):
            return ext

        store = BrowserLeaseStore()
        cdp_lease = store.request(
            task_id="c1", run_id="r1", domains=["example.com"],
            profile_mode="cdp_override", consent_ref="interaction-7",
        )
        store.activate(cdp_lease)

        b = BrowserCapabilityBroker(
            adapters=[DomPrimitivesAdapter(), ext],
            decision_provider=FailClosedDecisionProvider(),
            extension_selector=selector,
        )
        action = make_action(task_id="c1", run_id="r1")  # navigate not in ext caps
        result = b.execute(action, lease=cdp_lease)
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_BACKEND_UNAVAILABLE"

    def test_preferred_backend_respected_when_envelope_fit(self, lease, broker):
        action = make_action(preferred_backend="dom")
        result = broker.execute(action, lease=lease)
        assert result.status is ExecutionStatus.DONE
        assert result.backend_name == "dom"


# ---------------------------------------------------------------------------
# Adversarial
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_js_redirect_attempt_to_imds_stays_domain_gated(self, lease, broker):
        # The adversarial site tries to navigate to the IMDS endpoint; the
        # broker can only act on declared lease domains, so the target domain
        # is not in the lease => denied before any backend.
        action = make_action(
            domain="169.254.169.254", url="http://169.254.169.254/latest/meta-data/"
        )
        result = broker.execute(action, lease=lease)
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_SESSION_DOMAIN_MISMATCH"

    def test_retry_after_ambiguous_effect_is_not_automatic(self, lease, broker):
        # A crashed adapter (unknown state) must yield UNKNOWN +
        # RECONCILIATION_REQUIRED, never a blind retry.
        class FlakyAdapter(DomPrimitivesAdapter):
            # name stays "dom" so the lane resolver can select it
            version = "flaky"

            def execute(self, attempt):
                raise RuntimeError("connection lost mid-effect")

        b = BrowserCapabilityBroker(
            adapters=[FlakyAdapter()],
            decision_provider=FailClosedDecisionProvider(),
        )
        result = b.execute(make_action(), lease=lease)
        assert result.status is ExecutionStatus.UNKNOWN
        assert result.failure_category == "GEA_RECONCILIATION_REQUIRED"
        assert result.retryable is False

    def test_double_lease_claim_denied(self):
        store = BrowserLeaseStore()
        store.request(task_id="c1", run_id="r1", domains=["example.com"])
        with pytest.raises(LeaseConflictError):
            store.request(task_id="c1", run_id="r1", domains=["example.com"])

    def test_console_eval_cookie_read_still_permitted_only_under_lease(self, lease, broker):
        # SESSION ACCESS: console_eval on a declared domain is a read; the
        # *denylist* enforcement lives in browser_tool's restrict_evaluate.
        # The broker's job is the lease gate, and it must not widen to any
        # undeclared domain.
        action = make_action(verb=BrowserVerb.CONSOLE_EVAL, raw={"expression": "document.cookie"})
        result = broker.execute(action, lease=lease)
        assert result.status is ExecutionStatus.DONE  # lease-gated read executed
        # Undeclared domain: same expression is denied at the frontier.
        action2 = make_action(
            verb=BrowserVerb.CONSOLE_EVAL,
            raw={"expression": "document.cookie"},
            domain="evil.example.net",
            url="https://evil.example.net/",
        )
        result2 = broker.execute(action2, lease=lease)
        assert result2.status is ExecutionStatus.FAILED
        assert result2.failure_category == "GEA_SESSION_DOMAIN_MISMATCH"

    def test_revoked_lease_terminal_for_side_effects(self, lease, broker):
        store = BrowserLeaseStore()
        lease2 = store.request(task_id="c9", run_id="r9", domains=["example.com"])
        store.activate(lease2)
        store.revoke(lease2, revocation_ref="rev-1")
        action = make_action(task_id="c9", run_id="r9")
        result = broker.execute(action, lease=lease2)
        assert result.status is ExecutionStatus.FAILED
        assert result.failure_category == "GEA_SESSION_LEASE_MISSING"