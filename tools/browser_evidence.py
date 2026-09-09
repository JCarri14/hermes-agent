"""Browser evidence adapter — ExecutionResult → BrowserReceipt.

Implements the receipt projection of ``BROWSER_EXECUTION_CAPABILITY_V1.md``
§3.4: the browser receipt IS the GEA ``ActionReceipt`` with browser
extensions, projected onto existing stores (``task_events`` + ``task_runs``
+ ``attachments``) — no new table.

Contract rules enforced here:

- **No secrets ever**: receipts carry only hashed evidence refs and
  redacted tab info. Cookie values, tokens, query params with credentials,
  and raw screenshots are rejected/redacted before projection.
- **``execution_status=success`` without a verified postcondition**
  ⇒ receipt ``partial|unknown``, never ``verified`` (GEA §9).
- **Postcondition mapping** per verb: each verb maps to a default
  postcondition type, overridable by ``action.expected_postcondition``.
- **Backend identity + version** are recorded from the adapter
  (``result.backend_name`` / ``result.backend_version``), never inferred.
- Evidence refs are sha256 hashes of the raw evidence content — the raw
  bytes stay out of the active context (``AUDIT != ACTIVE CONTEXT``).

Thread-safe: pure functions, immutable dicts out.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.browser_capability_broker import (
    BrowserAction,
    BrowserVerb,
    ExecutionResult,
    ExecutionStatus,
    PostconditionStatus,
)

#: Secrets that must never survive into a receipt, even inside evidence text.
#: ``[^\s&;]+`` bounds each match so one redaction cannot swallow a whole URL
#: query string ("token=abc&q=hello" → only the token value is replaced).
_SECRET_PATTERNS = (
    re.compile(r"(?i)(cookie|token|password|passwd|api[_-]?key|secret|authorization|bearer)\s*[=:]\s*[^\s&;]+"),
    # userinfo w/ credentials in URL: keep the scheme, drop the credentials
    re.compile(r"(?i)(https?://)[^/\s:@]+:[^/\s:@]+@"),
)

#: Query param name hints that carry credentials and must be redacted by key.
_CREDENTIAL_QUERY_HINTS = ("token", "key", "secret", "password", "auth", "sig", "signature", "session")

#: Default expected postcondition type per verb (design §3.1 table + §4.1).
_VERB_DEFAULT_POSTCONDITION = {
    BrowserVerb.NAVIGATE: "page_load",
    BrowserVerb.READ_SNAPSHOT: "provider_id",
    BrowserVerb.EXTRACT: "dom_contains",
    BrowserVerb.CLICK: "state_fingerprint",
    BrowserVerb.TYPE: "dom_contains",
    BrowserVerb.SCROLL: "state_fingerprint",
    BrowserVerb.BACK: "url_equals",
    BrowserVerb.PRESS: "state_fingerprint",
    BrowserVerb.CONSOLE_EVAL: "state_fingerprint",
    BrowserVerb.DIALOG: "state_fingerprint",
    BrowserVerb.GET_IMAGES: "provider_id",
    BrowserVerb.VISION: "provider_id",
    BrowserVerb.UPLOAD: "provider_id",
    BrowserVerb.DOWNLOAD: "provider_id",
    BrowserVerb.SUBMIT: "page_load",
    BrowserVerb.VERIFY: "dom_contains",
    BrowserVerb.SESSION_ATTACH: "provider_id",
    BrowserVerb.SESSION_RELEASE: "provider_id",
}


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_text(text: str) -> str:
    return sha256_of_bytes(text.encode("utf-8", errors="replace"))


def redact_text(value: str) -> str:
    """Redact credential-shaped fragments from arbitrary evidence text."""
    out = value
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("[REDACTED]", out)
    return out


def redact_url(url: Optional[str]) -> Optional[str]:
    """Redact query params that carry credentials and userinfo credentials.

    Parses the (valid) URL first, then redacts the userinfo and credential
    query params component-wise — never injecting markers into the netloc in
    a way that would break re-parsing (bracketed-host ambiguity).
    """
    if not url:
        return url
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    try:
        parts = urlsplit(url)
    except ValueError:
        # Unparseable URL: aggressive text redaction is the safe fallback.
        return redact_text(url)
    scheme, netloc, path, query, fragment = parts

    # Userinfo credentials → replace the whole userinfo with a marker,
    # keep the host.
    if "@" in netloc:
        userinfo, _, host = netloc.rpartition("@")
        if userinfo:
            netloc = f"[REDACTED]@{host}"

    # Query params whose names look like credentials → the WHOLE pair is
    # redacted (key and value), so the receipt does not even reveal that a
    # credential-shaped parameter existed in the URL.
    if query:
        kept = []
        hidden = 0
        for key, val in parse_qsl(query, keep_blank_values=True):
            low = key.lower()
            if any(h in low for h in _CREDENTIAL_QUERY_HINTS):
                hidden += 1
            else:
                kept.append((key, val))
        if hidden:
            kept.append(("[redacted_params]", str(hidden)))
        query = urlencode(kept)

    return urlunsplit((scheme, netloc, path, query, fragment))


def evidence_hashes(raw_evidence: List[Dict[str, Any]]) -> List[str]:
    """Reduce raw evidence entries to a list of content hashes.

    Each entry is ``{"kind": ..., "content": ...}`` where content is a
    string (DOM text, snapshot, base64 image, etc.). Only sha256 hashes
    return; raw content never re-enters the active context.
    """
    refs: List[str] = []
    for item in raw_evidence or []:
        content = item.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            refs.append(sha256_of_text(content))
        elif isinstance(content, bytes):
            refs.append(sha256_of_bytes(content))
    return refs


@dataclass(frozen=True)
class BrowserReceipt:
    """Projection of GEA ActionReceipt + browser extensions (§3.4).

    Immutable; serializable via :meth:`as_dict`. Never contains secrets.
    """

    receipt_id: str
    operation_id: str
    task_id: str
    run_id: str
    attempt_id: str
    step_key: Optional[str]
    authorization: Dict[str, Any]
    execution: Dict[str, Any]
    browser: Dict[str, Any]
    postcondition: Dict[str, Any]
    failure: Optional[Dict[str, Any]]
    created_at: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": "governed_external_actions.v1.ActionReceipt",
            "version": "1.0",
            "receipt_id": self.receipt_id,
            "operation_id": self.operation_id,
            "step_key": self.step_key,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "authorization": self.authorization,
            "execution": self.execution,
            "browser": self.browser,
            "postcondition": self.postcondition,
            "failure": self.failure,
            "created_at": self.created_at,
        }


class BrowserEvidenceAdapter:
    """Normalizes an :class:`ExecutionResult` into a :class:`BrowserReceipt`.

    Usage::

        adapter = BrowserEvidenceAdapter()
        receipt = adapter.build_receipt(action, result, lease=lease, decision=decision)

    Pure normalization — no I/O, no storage. Persistence is the caller's
    job (project onto ``task_events`` / attachments).
    """

    def build_receipt(
        self,
        action: BrowserAction,
        result: ExecutionResult,
        *,
        lease: Optional[Any] = None,
        decision: Optional[Any] = None,
        raw_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> BrowserReceipt:
        # Postcondition: success without verification is partial/unknown.
        postcondition = self._postcondition(action, result)

        browser_block = {
            "backend": {
                "name": result.backend_name,
                "version": result.backend_version,
            },
            "profile_mode": (
                lease.profile_mode if lease is not None else action.profile_mode.value
            ),
            "session_lease_id": action.session_lease_id or (
                lease.lease_id if lease is not None else None
            ),
            "tab_info_redacted": redact_url(
                getattr(action.target, "url", None)
            ),
            "evidence_refs": evidence_hashes(raw_evidence if raw_evidence is not None else result.evidence),
        }

        authorization_block: Dict[str, Any] = (
            decision.as_dict()
            if decision is not None and hasattr(decision, "as_dict")
            else {"decision": "unknown", "reason_code": "GEA_UNKNOWN"}
        )
        authorization_block["approval_refs"] = [
            ref for ref in [
                getattr(lease, "consent_ref", None) if lease is not None else None,
                getattr(lease, "grant_ref", None) if lease is not None else None,
            ]
            if ref
        ]

        execution_status = result.status.value
        execution_block = {
            "request_hash": self._request_hash(action),
            "execution_status": execution_status,
            "started_at": None,
            "finished_at": time.time(),
        }

        failure_block = None
        if result.status in (ExecutionStatus.FAILED, ExecutionStatus.UNKNOWN):
            failure_block = {
                "category": result.failure_category or result.error or "UNKNOWN",
                "retryable": bool(result.retryable),
            }

        return BrowserReceipt(
            receipt_id=self._receipt_id(action, result),
            operation_id=action.operation_id,
            task_id=action.task_id,
            run_id=action.run_id,
            attempt_id=action.attempt_id or "1",
            step_key=action.step_key,
            authorization=authorization_block,
            execution=execution_block,
            browser=browser_block,
            postcondition=postcondition,
            failure=failure_block,
            created_at=time.time(),
        )

    # -- internal helpers -----------------------------------------------------

    def _postcondition(self, action: BrowserAction, result: ExecutionResult) -> Dict[str, Any]:
        expected_type = (
            action.expected_postcondition.type
            if action.expected_postcondition is not None
            else _VERB_DEFAULT_POSTCONDITION.get(action.verb, "provider_id")
        )
        expected_ref = (
            action.expected_postcondition.ref
            if action.expected_postcondition is not None
            else None
        )
        if result.status in (ExecutionStatus.DONE, ExecutionStatus.ACCEPTED):
            # success without a verified postcondition => partial, never verified
            status = PostconditionStatus.UNVERIFIED.value
            verifier = "broker"
        elif result.status is ExecutionStatus.INTENT:
            status = PostconditionStatus.UNVERIFIED.value
            verifier = "broker"
        else:
            status = PostconditionStatus.UNKNOWN.value
            verifier = "broker"
        return {
            "status": status,
            "expected_ref": expected_ref,
            "observed_ref": None,
            "verified_at": time.time(),
            "verifier": verifier,
            "expected_type": expected_type,
        }

    def _request_hash(self, action: BrowserAction) -> str:
        return action.payload.params_digest or sha256_of_text(json.dumps(action.as_dict(), sort_keys=True))

    def _receipt_id(self, action: BrowserAction, result: ExecutionResult) -> str:
        raw = f"{action.operation_id}|{action.attempt_id or '1'}|{result.backend_name}"
        return sha256_of_text(raw)