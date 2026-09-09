"""Contract / redaction tests for the browser evidence adapter.

Covers design §3.4 (BrowserReceipt = GEA ActionReceipt projection), §9
(receipt lifecycle: success without verified postcondition is never
``verified``), §5.4 (no secrets in receipts) and §11 contract rows.
"""

import re

import pytest

from agent.browser_capability_broker import (
    BrowserAction,
    BrowserTarget,
    BrowserPayload,
    BrowserVerb,
    ExecutionResult,
    ExecutionStatus,
)
from agent.browser_lease_store import BrowserLeaseStore
from tools.browser_evidence import (
    BrowserEvidenceAdapter,
    sha256_of_text,
    evidence_hashes,
    redact_text,
    redact_url,
)


def make_action(
    verb=BrowserVerb.NAVIGATE,
    domain="example.com",
    url="https://example.com/",
    raw=None,
    task_id="card-1",
    run_id="run-1",
    **kw,
):
    return BrowserAction(
        verb=verb,
        operation_id=kw.pop("operation_id", "op-1"),
        task_id=task_id,
        run_id=run_id,
        attempt_id=kw.pop("attempt_id", "1"),
        target=BrowserTarget(domain=domain, url=url),
        payload=BrowserPayload(kind="none", raw=raw),
        session_lease_id=kw.pop("session_lease_id", "lease-card-1-run-1"),
        **kw,
    )


@pytest.fixture
def adapter():
    return BrowserEvidenceAdapter()


@pytest.fixture
def lease():
    store = BrowserLeaseStore()
    l = store.request(
        task_id="card-1", run_id="run-1", domains=["example.com"],
        profile_mode="isolated",
    )
    store.activate(l)
    return l


# ---------------------------------------------------------------------------
# Contract — receipt shape (§3.4, GEA §5.2)
# ---------------------------------------------------------------------------


class TestReceiptContract:
    def test_receipt_is_gea_action_receipt_projection(self, adapter, lease):
        action = make_action()
        result = ExecutionResult(
            status=ExecutionStatus.DONE,
            backend_name="dom",
            backend_version="1.0",
            result={"ok": True},
        )
        receipt = adapter.build_receipt(action, result, lease=lease)
        d = receipt.as_dict()
        assert d["schema"] == "governed_external_actions.v1.ActionReceipt"
        assert d["version"] == "1.0"
        assert d["operation_id"] == "op-1"
        assert d["task_id"] == "card-1"
        assert d["run_id"] == "run-1"
        assert d["execution"]["execution_status"] == "done"
        assert d["browser"]["backend"] == {"name": "dom", "version": "1.0"}
        assert d["browser"]["session_lease_id"] == "lease-card-1-run-1"
        assert d["browser"]["profile_mode"] == "isolated"

    def test_success_without_postcondition_is_unverified_never_verified(self, adapter, lease):
        action = make_action()
        result = ExecutionResult(
            status=ExecutionStatus.DONE, backend_name="dom", backend_version="1.0"
        )
        receipt = adapter.build_receipt(action, result, lease=lease)
        assert receipt.postcondition["status"] == "UNVERIFIED"
        assert receipt.postcondition["status"] != "PASS"

    def test_unknown_status_maps_to_unknown_postcondition(self, adapter, lease):
        action = make_action()
        result = ExecutionResult(
            status=ExecutionStatus.UNKNOWN,
            backend_name="dom",
            backend_version="1.0",
            error="GEA_RECONCILIATION_REQUIRED",
            failure_category="GEA_RECONCILIATION_REQUIRED",
        )
        receipt = adapter.build_receipt(action, result, lease=lease)
        assert receipt.postcondition["status"] == "UNKNOWN"
        assert receipt.failure["category"] == "GEA_RECONCILIATION_REQUIRED"
        assert receipt.failure["retryable"] is False

    def test_expected_postcondition_type_from_action(self, adapter, lease):
        from agent.browser_capability_broker import ExpectedPostcondition

        action = make_action(
            verb=BrowserVerb.SUBMIT,
            expected_postcondition=ExpectedPostcondition(type="url_equals", predicate="/done"),
        )
        result = ExecutionResult(
            status=ExecutionStatus.ACCEPTED, backend_name="dom", backend_version="1.0"
        )
        receipt = adapter.build_receipt(action, result, lease=lease)
        assert receipt.postcondition["expected_ref"] is None
        assert receipt.postcondition["expected_type"] == "url_equals"

    def test_authorization_block_carries_decision_and_approval_refs(self, adapter, lease):
        from agent.browser_capability_broker import AuthorizationDecision, Decision

        action = make_action()
        result = ExecutionResult(
            status=ExecutionStatus.DONE, backend_name="dom", backend_version="1.0"
        )
        decision = AuthorizationDecision(Decision.PERMIT, "GEA_OK")
        receipt = adapter.build_receipt(action, result, lease=lease, decision=decision)
        assert receipt.authorization["decision"] == "permit"
        assert isinstance(receipt.authorization["approval_refs"], list)

    def test_backend_identity_recorded_not_inferred(self, adapter, lease):
        action = make_action()
        result = ExecutionResult(
            status=ExecutionStatus.DONE,
            backend_name="browser_exec",
            backend_version="9.9.9",
        )
        receipt = adapter.build_receipt(action, result, lease=lease)
        assert receipt.browser["backend"] == {"name": "browser_exec", "version": "9.9.9"}


# ---------------------------------------------------------------------------
# Contract — redaction invariants (§5.4: receipts never contain secrets)
# ---------------------------------------------------------------------------

_COOKIE_TOKEN_RE = re.compile(r"(?i)(cookie|token|api[_-]?key|password|secret|bearer)\s*[=:]\s*\S+")


class TestRedaction:
    def test_receipt_never_contains_secrets(self, adapter, lease):
        action = make_action(
            url="https://example.com/?token=abc123&session=xyz",
            raw={"text": "my password is hunter2 and api_key=deadbeef"},
        )
        result = ExecutionResult(
            status=ExecutionStatus.DONE,
            backend_name="dom",
            backend_version="1.0",
            evidence=[{"kind": "snapshot", "content": "Session token: abc123"}],
        )
        receipt = adapter.build_receipt(action, result, lease=lease, raw_evidence=[
            {"kind": "snapshot", "content": "Session token: abc123"},
        ])
        blob = str(receipt.as_dict())
        assert not _COOKIE_TOKEN_RE.search(blob)
        assert "abc123" not in blob
        assert "hunter2" not in blob
        assert "deadbeef" not in blob
        assert "xyz" not in blob

    def test_evidence_refs_are_hashes_not_content(self, adapter, lease):
        action = make_action()
        result = ExecutionResult(
            status=ExecutionStatus.DONE, backend_name="dom", backend_version="1.0"
        )
        raw = [{"kind": "snapshot", "content": "secret-page-content-12345"}]
        receipt = adapter.build_receipt(action, result, lease=lease, raw_evidence=raw)
        refs = receipt.browser["evidence_refs"]
        assert len(refs) == 1
        assert refs[0] == sha256_of_text("secret-page-content-12345")
        assert "secret-page-content-12345" not in str(receipt.as_dict())

    def test_evidence_hashes_multi(self):
        refs = evidence_hashes([
            {"kind": "dom", "content": "hello"},
            {"kind": "img", "content": b"\x89PNG"},
            {"kind": "empty", "content": None},
        ])
        assert len(refs) == 2

    def test_redact_url_userinfo_and_query_credentials(self):
        out = redact_url("https://user:pass@example.com/path?token=abc&q=hello")
        assert out is not None
        assert "pass" not in out
        assert "abc" not in out
        assert "user:" not in out
        assert "q=hello" in out

    def test_redact_url_plain_url_unchanged(self):
        plain = "https://example.com/path?q=hello"
        assert redact_url(plain) == plain

    def test_redact_text_secret_fragments(self):
        out = redact_text("token=abc123 and password=hunter2")
        assert "abc123" not in out
        assert "hunter2" not in out

    def test_digest_hosts_secret_values_but_not_payload_travel(self, adapter, lease):
        # The raw payload must never travel in the receipt serialization.
        action = make_action(raw={"q": "needle-value"})
        result = ExecutionResult(
            status=ExecutionStatus.DONE, backend_name="dom", backend_version="1.0"
        )
        receipt = adapter.build_receipt(action, result, lease=lease)
        assert "needle-value" not in str(receipt.as_dict())
        assert receipt.browser["tab_info_redacted"] == "https://example.com/"