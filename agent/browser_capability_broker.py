"""Browser capability broker — typed agent-side frontier (BROWSER_EXECUTION_CAPABILITY_V1).

This module is the agent-side decision layer of the single browser frontier
described by the approved design contract
``BROWSER_EXECUTION_CAPABILITY_V1.md`` (§3-§4, §8): a provider-neutral
``BrowserAction`` IR, a ``BrowserBackendAdapter`` interface, and a broker
that resolves a backend lane and executes inside an authority envelope that
it can never widen.

Contract invariants implemented here (fail-closed by construction):

- **Decision before effect.** An ``AuthorizationDecision`` (GEA §3.3) is
  computed BEFORE any backend resolution. ``deny`` short-circuits; only
  ``permit`` proceeds to backend selection (``require_approval`` surfaces a
  human gate).
- **SESSION ACCESS != ACTION AUTHORIZATION.** An ACTIVE session lease
  enables *reads* on the domains it declares; side-effect verbs
  (``submit``, ``session_attach``, ``upload``, ``download``) go through the
  gate independently and default to ``require_approval``.
- **Envelope without widening.** ``envelope = min(lease.profile_mode,
  capability_budget, manifest ceiling, data_sensitivity_ceiling)``. The
  resolver selects the first adapter whose ``risk_profile`` fits the
  envelope; the adapter cannot alter ``params_digest``, ``target.domain`` or
  ``profile_mode``.
- **Restricted fallback.** On failure of the selected backend, the broker
  only falls back to a backend whose ``risk_profile`` is no more
  privileged than the first, with the SAME ``profile_mode``. A user
  consented for existing-profile is never silently downgraded to a
  throwaway lane — missing/ambiguous scope fails closed
  (``GEA_SESSION_LEASE_MISSING`` / ``GEA_SESSION_DOMAIN_MISMATCH``),
  mirroring the verified precedent in ``tools/browser_tool.py``
  ("a consented user must never be silently downgraded to a throwaway").
- **Extension lane is authoritative.** When an authenticated browser
  controller is bound (``gateway.browser_control_broker``), the extension
  lane is the only candidate for that action; a capability mismatch or
  missing/ambiguous scope fails closed and never jumps to another local or
  cloud backend.

Secrets never appear in broker objects: payloads are reduced to
``params_digest`` (sha256), URLs are domain/reference level, and receipts
carry only hashed evidence refs (see ``tools.browser_evidence``).

The decision provider is injectable (``decision_provider=``) so the future
``hermes_cli.action_gate`` (GEA §3.5 component) can plug in; the default is
a fail-closed provider documented below.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical enums (GEA §5 + browser contract §3.1.1)
# ---------------------------------------------------------------------------


class BrowserVerb(str, Enum):
    """Provider-neutral IR verbs (§3.1.1)."""

    NAVIGATE = "navigate"
    READ_SNAPSHOT = "read_snapshot"
    EXTRACT = "extract"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    BACK = "back"
    PRESS = "press"
    CONSOLE_EVAL = "console_eval"
    DIALOG = "dialog"
    GET_IMAGES = "get_images"
    VISION = "vision"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    SUBMIT = "submit"
    VERIFY = "verify"
    SESSION_ATTACH = "session_attach"
    SESSION_RELEASE = "session_release"


#: Verbs that are pure reads — gated only by an ACTIVE lease on declared domains.
READ_VERBS = frozenset(
    {
        BrowserVerb.READ_SNAPSHOT,
        BrowserVerb.EXTRACT,
        BrowserVerb.GET_IMAGES,
        BrowserVerb.VISION,
        BrowserVerb.VERIFY,
        BrowserVerb.CONSOLE_EVAL,
    }
)

#: Side-effect verbs — never implicitly authorized; require_approval by default.
#: The contract names ``submit`` and ``session_attach`` explicitly; V1 keeps
#: ``upload``/``download`` fail-closed as well (file payloads are side effects).
SIDE_EFFECT_VERBS = frozenset(
    {
        BrowserVerb.SUBMIT,
        BrowserVerb.SESSION_ATTACH,
        BrowserVerb.UPLOAD,
        BrowserVerb.DOWNLOAD,
    }
)

#: Special verbs that never map to an execution backend (lease lifecycle).
LIFECYCLE_VERBS = frozenset({BrowserVerb.SESSION_RELEASE})


class ProfileMode(str, Enum):
    """Lease / execution profile mode (§3.1 context.profile_mode)."""

    ISOLATED = "isolated"
    EXISTING_PROFILE = "existing_profile"
    CLOUD = "cloud"
    CDP_OVERRIDE = "cdp_override"


class RiskProfile(str, Enum):
    """Adapter risk profile (§3.2). Ordered from least to most privileged."""

    LOCAL_THROWAWAY = "local_throwaway"
    CLOUD = "cloud"
    EXISTING_PROFILE = "existing_profile"
    CONTROLLER_BOUND = "controller_bound"


class ActionClass(str, Enum):
    """GEA §5.1 — canonical action class."""

    READ_ONLY = "READ_ONLY"
    EXTERNAL_REVERSIBLE = "EXTERNAL_REVERSIBLE"
    EXTERNAL_CONSEQUENTIAL = "EXTERNAL_CONSEQUENTIAL"
    EXTERNAL_DESTRUCTIVE = "EXTERNAL_DESTRUCTIVE"


class PostconditionStatus(str, Enum):
    """GEA §5.7 — receipt postcondition status."""

    UNVERIFIED = "UNVERIFIED"
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    STALE = "STALE"


class ExecutionStatus(str, Enum):
    """GEA §5.8 — execution status."""

    INTENT = "intent"
    ACCEPTED = "accepted"
    DONE = "done"
    FAILED = "failed"
    UNKNOWN = "unknown"


class Decision(str, Enum):
    """GEA §3.3 — authorization decision."""

    PERMIT = "permit"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


# Priority order for backend lanes inside BROWSER_UI (contract §4.2).
# Extension is handled separately because it is authoritative when bound.
BACKEND_LANE_ORDER: Tuple[str, ...] = ("dom", "browser_exec", "semantic", "cua")

#: Max rank for fallback: a fallback backend may be at most *equally*
#: privileged, and never a different profile_mode.
_RISK_RANK: Dict[RiskProfile, int] = {
    RiskProfile.LOCAL_THROWAWAY: 0,
    RiskProfile.CLOUD: 1,
    RiskProfile.EXISTING_PROFILE: 2,
    RiskProfile.CONTROLLER_BOUND: 3,
}


# ---------------------------------------------------------------------------
# Authorization decision (GEA §3.3 — local minimal, injectable provider)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthorizationDecision:
    """GEA §3.3 compatible decision object.

    ``decision_provider`` may be swapped later by ``hermes_cli.action_gate``;
    the default provider in this module is fail-closed.
    """

    decision: Decision
    reason_code: str
    matched_policy: Optional[Dict[str, str]] = None
    matched_grant: Optional[Dict[str, Any]] = None
    message_human: Optional[str] = None

    @property
    def is_permit(self) -> bool:
        return self.decision is Decision.PERMIT

    @property
    def is_require_approval(self) -> bool:
        return self.decision is Decision.REQUIRE_APPROVAL

    @property
    def is_deny(self) -> bool:
        return self.decision is Decision.DENY

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason_code": self.reason_code,
            "matched_policy": self.matched_policy,
            "matched_grant": self.matched_grant,
            "message_human": self.message_human,
        }


# GEA §6 reason codes used by this broker (subset).
REASON_SESSION_LEASE_MISSING = "GEA_SESSION_LEASE_MISSING"
REASON_SESSION_LEASE_EXPIRED = "GEA_SESSION_LEASE_EXPIRED"
REASON_SESSION_DOMAIN_MISMATCH = "GEA_SESSION_DOMAIN_MISMATCH"
REASON_SESSION_ACCOUNT_MISMATCH = "GEA_SESSION_ACCOUNT_MISMATCH"
REASON_SESSION_RECONSENT_REQUIRED = "GEA_SESSION_RECONSENT_REQUIRED"
REASON_AUTHORIZATION_MISSING = "GEA_AUTHORIZATION_MISSING"
REASON_AUTHORIZATION_EXPIRED = "GEA_AUTHORIZATION_EXPIRED"
REASON_AUTHORIZATION_REVOKED = "GEA_AUTHORIZATION_REVOKED"
REASON_AUTHORIZATION_SCOPE_MISMATCH = "GEA_AUTHORIZATION_SCOPE_MISMATCH"
REASON_AUTHORIZATION_TARGET_MISMATCH = "GEA_AUTHORIZATION_TARGET_MISMATCH"
REASON_PAYLOAD_HASH_MISMATCH = "GEA_PAYLOAD_HASH_MISMATCH"
REASON_HUMAN_APPROVAL_REQUIRED = "GEA_HUMAN_APPROVAL_REQUIRED"
REASON_CAPABILITY_OUT_OF_SCOPE = "GEA_CAPABILITY_OUT_OF_SCOPE"
REASON_TOOLSET_NOT_ALLOWED_FOR_CARD = "GEA_TOOLSET_NOT_ALLOWED_FOR_CARD"
REASON_UNTRUSTED_CONTENT_AUTHORITY = "GEA_UNTRUSTED_CONTENT_AUTHORITY_ATTEMPT"
REASON_BLIND_RETRY_FORBIDDEN = "GEA_BLIND_RETRY_FORBIDDEN"
REASON_RETRY_IDENTITY_MISSING = "GEA_RETRY_IDENTITY_MISSING"
REASON_RECONCILIATION_REQUIRED = "GEA_RECONCILIATION_REQUIRED"
REASON_ACK_MISSING = "GEA_ACK_MISSING"
REASON_POSTCONDITION_UNVERIFIED = "GEA_POSTCONDITION_UNVERIFIED"
REASON_BACKEND_UNAVAILABLE = "GEA_BACKEND_UNAVAILABLE"
REASON_UI_DRIFT = "GEA_UI_DRIFT"
REASON_STALE_SELECTOR = "GEA_STALE_SELECTOR"
REASON_RETRYABLE_TRANSIENT = "GEA_RETRYABLE_TRANSIENT"

#: Payload keys that are never allowed into the IR payload (secrets stay out).
_SECRET_KEY_HINTS = (
    "password", "passwd", "token", "secret", "cookie", "api_key", "apikey",
    "authorization", "credentials", "credential", "bearer", "private_key",
    "session_key", "auth",
)


def payload_contains_secret_hint(payload: Optional[Dict[str, Any]]) -> bool:
    """True if ``payload`` carries a key that looks like a secret hint.

    The IR contract forbids secrets in the payload — they would leak into
    logs, digests and receipts. Fail-closed: any suspicious key rejects the
    action at IR construction time.
    """
    if not payload:
        return False
    lowered = {str(k).lower() for k in payload.keys()}
    return any(hint in key for key in lowered for hint in _SECRET_KEY_HINTS)


# ---------------------------------------------------------------------------
# BrowserAction IR (contract §3.1)
# ---------------------------------------------------------------------------

_POSTCONDITION_TYPES = (
    "page_load", "url_equals", "dom_contains", "provider_id",
    "state_fingerprint", "element_absent",
)


@dataclass(frozen=True)
class BrowserTarget:
    """Target of a browser action (§3.1 target)."""

    url: Optional[str] = None
    domain: str = ""
    ref: Optional[str] = None
    selector: Optional[str] = None
    iframe: Optional[str] = None
    tab: Optional[str] = None
    account_hint: Optional[str] = None


@dataclass(frozen=True)
class BrowserPayload:
    """Action payload — never contains secrets; only a digest travels."""

    kind: str = "none"  # none | text | file_upload | file_download | js_expression | form
    params_digest: str = ""
    raw: Optional[Dict[str, Any]] = None  # broker-local only; not serialized


@dataclass(frozen=True)
class ExpectedPostcondition:
    """Desired postcondition (§3.1 expected_postcondition → §5.7)."""

    type: str = "page_load"  # one of _POSTCONDITION_TYPES
    predicate: str = ""
    ref: Optional[str] = None


@dataclass(frozen=True)
class BrowserAction:
    """Provider-neutral IR (schema ``browser_execution.v1.BrowserAction``).

    Validation performed at construction:

    - payload must not contain secret-hint keys (fail-closed);
    - ``params_digest`` is recomputed when omitted (stable sha256 over the
      canonical JSON of the raw payload, keys sorted);
    - target.domain is required and normalized lowercase;
    - postcondition type must be one of the supported set.
    """

    verb: BrowserVerb
    operation_id: str
    task_id: str
    run_id: str
    step_key: Optional[str] = None
    attempt_id: str = ""
    target: BrowserTarget = field(default_factory=BrowserTarget)
    payload: BrowserPayload = field(default_factory=BrowserPayload)
    session_lease_id: Optional[str] = None
    profile_mode: ProfileMode = ProfileMode.ISOLATED
    preferred_backend: Optional[str] = None
    expected_postcondition: Optional[ExpectedPostcondition] = None
    strategy_hint: Optional[str] = None
    allowed_data_sensitivity: Optional[str] = None
    schema: str = "browser_execution.v1.BrowserAction"
    version: str = "1.0"

    def __post_init__(self) -> None:
        if payload_contains_secret_hint((self.payload.raw if self.payload else None) or {}):
            raise ValueError("BrowserAction payload must not contain secret-hint keys")
        raw = (self.payload.raw if self.payload else None) or {}
        if not self.payload.params_digest:
            digest = _stable_payload_digest(raw)
            object.__setattr__(self.payload, "params_digest", digest)
        domain = (self.target.domain or "").strip().lower()
        if not domain:
            raise ValueError("BrowserAction requires a declared target.domain")
        object.__setattr__(self.target, "domain", domain)
        if self.expected_postcondition is not None:
            if self.expected_postcondition.type not in _POSTCONDITION_TYPES:
                raise ValueError(
                    f"unsupported postcondition type {self.expected_postcondition.type!r}"
                )

    def domain_matches(self, domains: Sequence[str]) -> bool:
        """True if the action's target domain is declared in ``domains``.

        Exact match only (no wildcard widening); subdomains are NOT
        implicitly allowed — the lease declares what it covers. This is the
        cookie-jar boundary: it never travels to undeclared hosts.
        """
        return self.target.domain in {d.strip().lower() for d in domains if d}

    @property
    def is_side_effect(self) -> bool:
        return self.verb in SIDE_EFFECT_VERBS

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "operation_id": self.operation_id,
            "step_key": self.step_key,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "verb": self.verb.value,
            "target": {
                "url": self.target.url,
                "domain": self.target.domain,
                "ref": self.target.ref,
                "selector": self.target.selector,
                "iframe": self.target.iframe,
                "tab": self.target.tab,
                "account_hint": self.target.account_hint,
            },
            "payload": {"kind": self.payload.kind, "params_digest": self.payload.params_digest},
            "session_lease_id": self.session_lease_id,
            "profile_mode": self.profile_mode.value,
            "preferred_backend": self.preferred_backend,
            "expected_postcondition": (
                {
                    "type": self.expected_postcondition.type,
                    "predicate": self.expected_postcondition.predicate,
                    "ref": self.expected_postcondition.ref,
                }
                if self.expected_postcondition
                else None
            ),
            "strategy_hint": self.strategy_hint,
        }


def _stable_payload_digest(payload: Optional[Dict[str, Any]]) -> str:
    """Stable sha256 of ``payload`` — key order independent, secrets absent."""
    canonical = json.dumps(payload or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_operation_id(
    card_id: str, run_id: str, verb: BrowserVerb, target: BrowserTarget, payload: Optional[Dict[str, Any]]
) -> str:
    """Candidate operation_id composition (GEA §10.1 / H8 candidate).

    ``sha256(card + run + verb + domain + url + payload_digest)`` — stable
    between retries; a payload change produces a different id and therefore
    a different authorization. Ratification of the final composition is a
    human gate (H8); this is the documented V1 candidate.
    """
    digest = _stable_payload_digest(payload)
    raw = "|".join(
        [
            card_id,
            run_id,
            verb.value,
            (target.domain or "").lower(),
            target.url or "",
            digest,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# BrowserBackendAdapter interface (contract §3.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionAttempt:
    """A normalized execution attempt handed to an adapter.

    ``decision`` is filled by the broker before ``execute()``; adapters
    built via ``resolve()`` may leave it None (the attempt is then
    re-created by the broker with the real decision).
    """

    action: BrowserAction
    decision: Optional[AuthorizationDecision] = None
    envelope: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Normalized result from an adapter (not a receipt — §3.2 execute())."""

    status: ExecutionStatus
    backend_name: str
    backend_version: str
    result: Any = None
    error: Optional[str] = None
    failure_category: Optional[str] = None
    retryable: bool = False
    provider_key: Optional[str] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)  # hashed refs
    received_at: Optional[str] = None
    raw_hash: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.status in (ExecutionStatus.ACCEPTED, ExecutionStatus.DONE)


class BackendExecutionError(Exception):
    """Raised by adapters on deterministic execution failure."""

    def __init__(
        self,
        message: str,
        *,
        category: str = REASON_RETRYABLE_TRANSIENT,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable


class BrowserBackendAdapter:
    """Interface every backend lane implements (§3.2).

    Contract rules:

    - ``resolve()`` is PURE (no side effects) and returns an
      ``ExecutionAttempt`` or None (None => this backend does not apply;
      try the next one).
    - ``execute()`` receives an intent that was ALREADY authorized (the GEA
      decision is resolved before backend resolution).
    - The adapter cannot alter ``params_digest``, ``target.domain`` or
      ``profile_mode`` — doing so would be authority widening.
    - ``authority_envelope`` is ``"leafless"``: the adapter never grants
      authorization; it only executes what was authorized.
    """

    name: str = ""
    version: str = ""
    capabilities: Set[BrowserVerb] = set()
    risk_profile: RiskProfile = RiskProfile.LOCAL_THROWAWAY
    idempotency_support: str = "none"  # none | client_side | provider_key
    authority_envelope: str = "leafless"

    def resolve(self, action: BrowserAction) -> Optional[ExecutionAttempt]:
        raise NotImplementedError

    def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
        raise NotImplementedError

    def postcondition_verifier(self, action: BrowserAction, result: ExecutionResult):
        """Return a verdict or None (no self-verification support)."""
        return None

    def describes_capability(self, verb: BrowserVerb) -> bool:
        return verb in self.capabilities


# ---------------------------------------------------------------------------
# Fail-closed decision provider
# ---------------------------------------------------------------------------


class FailClosedDecisionProvider:
    """Default GEA decision provider (fail-closed, no authority widening).

    Decision order (GEA §3.3): hard-deny floor → capability/toolset ceiling
    → session lease → action approval. Anything ambiguous denies.

    Rules:

    - no ACTIVE lease for the declared domain ⇒ deny
      (SESSION ACCESS requires the lease; side effects always do);
    - side-effect verbs (submit/session_attach/upload/download) ⇒
      ``require_approval`` unless an explicit matching grant is supplied by
      the caller (``grant: {authorization_id, scope, expires_at}``);
    - read verbs ⇒ ``permit`` ONLY when the lease is ACTIVE and the target
      domain is declared in the lease;
    - a revoked lease is terminal: deny with
      ``GEA_AUTHORIZATION_REVOKED`` (no re-activation);
    - lifecycle verbs (session_release) are always permitted (idempotent);
    - a payload digest change vs the grant's recorded digest ⇒ deny
      ``GEA_PAYLOAD_HASH_MISMATCH`` (new payload ⇒ new authorization).
    """

    def decide(self, action: BrowserAction, lease: Optional[Any], **context: Any) -> AuthorizationDecision:
        grant = context.get("grant") or {}
        budget = context.get("capability_budget")

        # Hard floor: toolsets_capability budget absent ⇒ deny for everything
        # except read-only navigation under an ACTIVE lease (GEA §8:
        # absence of budget → deny by default; V1 brokers still allow lease
        # reads because SESSION ACCESS is first-class).
        if action.verb in LIFECYCLE_VERBS:
            return AuthorizationDecision(Decision.PERMIT, "GEA_OK")

        # Budget ceiling (GEA §8): evaluated BEFORE any permission branch so
        # a card capability budget can never be bypassed by the lease / read /
        # UI / grant paths below (authority widening). A verb whose capability
        # class is not listed is out of scope, period. A falsy budget (None or
        # an empty dict) means no ceiling is configured; an explicit
        # ``{"caps": [...]}`` (even with an empty caps list) is a real ceiling.
        if budget and not _verb_allowed_by_budget(action.verb, budget):
            return AuthorizationDecision(
                Decision.DENY, REASON_CAPABILITY_OUT_OF_SCOPE,
                message_human="verb not allowed by card capability budget",
            )

        if action.verb is BrowserVerb.NAVIGATE and not action.is_side_effect:
            if lease is None or not getattr(lease, "is_active", False):
                return AuthorizationDecision(
                    Decision.DENY, REASON_SESSION_LEASE_MISSING,
                    message_human="no active session lease for this task/run",
                )
            if not action.domain_matches(getattr(lease, "domains", [])):
                return AuthorizationDecision(
                    Decision.DENY, REASON_SESSION_DOMAIN_MISMATCH,
                    message_human=f"domain {action.target.domain!r} not declared in the lease",
                )
            return AuthorizationDecision(Decision.PERMIT, "GEA_OK")

        # All remaining verbs require an ACTIVE lease.
        if lease is None or not getattr(lease, "is_active", False):
            return AuthorizationDecision(
                Decision.DENY, REASON_SESSION_LEASE_MISSING,
                message_human="no active session lease for this task/run",
            )
        if not action.domain_matches(getattr(lease, "domains", [])):
            return AuthorizationDecision(
                Decision.DENY, REASON_SESSION_DOMAIN_MISMATCH,
                message_human=f"domain {action.target.domain!r} not declared in the lease",
            )

        # Side-effect verbs: never implicit — require approval or a grant.
        if action.is_side_effect:
            if grant:
                if grant.get("revoked"):
                    return AuthorizationDecision(
                        Decision.DENY, REASON_AUTHORIZATION_REVOKED,
                        matched_grant=grant,
                    )
                digest = grant.get("params_digest")
                if digest and digest != action.payload.params_digest:
                    return AuthorizationDecision(
                        Decision.DENY, REASON_PAYLOAD_HASH_MISMATCH,
                        matched_grant=grant,
                        message_human="payload changed since the grant — new authorization required",
                    )
                scope = grant.get("scope", "ONE_SHOT")
                if scope in ("ONE_SHOT", "CARD_SCOPE", "SESSION_SCOPE"):
                    return AuthorizationDecision(
                        Decision.PERMIT, "GEA_GRANT_MATCHED", matched_grant=grant,
                    )
                return AuthorizationDecision(
                    Decision.DENY, REASON_AUTHORIZATION_SCOPE_MISMATCH, matched_grant=grant,
                )
            return AuthorizationDecision(
                Decision.REQUIRE_APPROVAL, REASON_HUMAN_APPROVAL_REQUIRED,
                message_human=f"verb {action.verb.value!r} is a side effect — human approval required",
            )

        # Read/UI verbs inside an ACTIVE lease on a declared domain: SESSION
        # ACCESS permits them (they are not external side effects).
        if action.verb in READ_VERBS or action.verb in (
            BrowserVerb.CLICK, BrowserVerb.TYPE, BrowserVerb.SCROLL,
            BrowserVerb.BACK, BrowserVerb.PRESS, BrowserVerb.DIALOG,
        ):
            return AuthorizationDecision(Decision.PERMIT, "GEA_OK")

        return AuthorizationDecision(
            Decision.DENY, REASON_AUTHORIZATION_MISSING,
            message_human="no authorization path for this action",
        )


def _verb_allowed_by_budget(verb: BrowserVerb, budget: Dict[str, Any]) -> bool:
    """Map an IR verb to the budget capability vocabulary (GEA §8 caps)."""
    caps = set(budget.get("caps") or [])
    mapping = {
        BrowserVerb.NAVIGATE: {"read"},
        BrowserVerb.READ_SNAPSHOT: {"read"},
        BrowserVerb.EXTRACT: {"read", "extract"},
        BrowserVerb.CLICK: {"write"},
        BrowserVerb.TYPE: {"write"},
        BrowserVerb.SCROLL: {"read"},
        BrowserVerb.BACK: {"read"},
        BrowserVerb.PRESS: {"write"},
        BrowserVerb.CONSOLE_EVAL: {"read", "extract"},
        BrowserVerb.DIALOG: {"write"},
        BrowserVerb.GET_IMAGES: {"read"},
        BrowserVerb.VISION: {"read"},
        BrowserVerb.UPLOAD: {"write"},
        BrowserVerb.DOWNLOAD: {"read"},
        BrowserVerb.SUBMIT: {"write", "send"},
        BrowserVerb.VERIFY: {"read"},
        BrowserVerb.SESSION_ATTACH: {"auth", "session_attach"},
        BrowserVerb.SESSION_RELEASE: set(),
    }
    return bool((mapping.get(verb) or set()) & caps) or not mapping.get(verb)


# ---------------------------------------------------------------------------
# Capability broker
# ---------------------------------------------------------------------------


class BrowserCapabilityBroker:
    """Typed broker: decision → envelope → backend resolution → execution.

    Usage (agent-side, one instance per process):

        broker = BrowserCapabilityBroker(adapters=[dom_adapter, exec_adapter, ...])
        result = broker.execute(action, lease=lease, task_id=..., run_id=...)

    The broker is deliberately NOT wired into ``run_agent`` or the tool
    registry in V1: it is the typed decision+resolution layer that the
    existing seams (``tools.browser_extension_router`` for the extension
    lane, ``tools.browser_tool`` for DOM primitives, ``tools.browser_use_cli``
    for browser_exec) plug into. The design keeps the existing seams as
    adapters rather than replacing them in this phase.
    """

    def __init__(
        self,
        *,
        adapters: Optional[Sequence[BrowserBackendAdapter]] = None,
        decision_provider: Optional[Any] = None,
        lease_store: Optional[Any] = None,
        manifest_ceiling: Optional[Dict[str, Any]] = None,
        data_sensitivity_ceiling: str = "CONFIDENTIAL",
        capability_budget: Optional[Dict[str, Any]] = None,
        extension_selector: Optional[Callable[[BrowserAction], Optional[BrowserBackendAdapter]]] = None,
    ) -> None:
        self._adapters: List[BrowserBackendAdapter] = list(adapters or [])
        self._decision_provider = decision_provider or FailClosedDecisionProvider()
        self._lease_store = lease_store
        self._manifest_ceiling = manifest_ceiling
        self._data_sensitivity_ceiling = data_sensitivity_ceiling
        self._capability_budget = capability_budget
        self._extension_selector = extension_selector

    # -- registration -------------------------------------------------------

    def register_adapter(self, adapter: BrowserBackendAdapter) -> None:
        if not adapter.name:
            raise ValueError("adapter must declare a name")
        self._adapters.append(adapter)

    def adapters(self) -> List[BrowserBackendAdapter]:
        return list(self._adapters)

    # -- decision -----------------------------------------------------------

    def decide(self, action: BrowserAction, lease: Optional[Any] = None, **context: Any) -> AuthorizationDecision:
        return self._decision_provider.decide(action, lease, **context)

    # -- envelope -----------------------------------------------------------

    def envelope(self, action: BrowserAction, lease: Optional[Any]) -> Dict[str, Any]:
        """Compute the authority envelope (never widened by resolution)."""
        profile_mode = (lease.profile_mode if lease is not None else None) or action.profile_mode
        budget = self._effective_budget(lease)
        manifest = dict(self._manifest_ceiling or {})
        return {
            "profile_mode": profile_mode.value if isinstance(profile_mode, ProfileMode) else str(profile_mode),
            "capability_budget": budget,
            "manifest_ceiling": manifest,
            "data_sensitivity_ceiling": self._data_sensitivity_ceiling,
        }

    def _effective_budget(self, lease: Optional[Any]) -> Dict[str, Any]:
        """Merged capability budget: a lease-level budget (session lease) is
        more specific than the broker-level one and wins when present."""
        budget = dict(self._capability_budget or {})
        lease_budget = getattr(lease, "capability_budget", None)
        if isinstance(lease_budget, dict):
            budget = lease_budget or budget
        return budget

    def _risk_allowed_in_envelope(self, risk: RiskProfile, envelope: Dict[str, Any]) -> bool:
        profile_mode = envelope.get("profile_mode", "isolated")
        allowed: Dict[str, Set[RiskProfile]] = {
            "isolated": {RiskProfile.LOCAL_THROWAWAY},
            "cloud": {RiskProfile.CLOUD, RiskProfile.LOCAL_THROWAWAY},
            "existing_profile": {
                RiskProfile.EXISTING_PROFILE, RiskProfile.CONTROLLER_BOUND,
            },
            "cdp_override": {RiskProfile.CONTROLLER_BOUND, RiskProfile.LOCAL_THROWAWAY},
        }
        return risk in allowed.get(profile_mode, {RiskProfile.LOCAL_THROWAWAY})

    def _risk_fallback_allowed(
        self, candidate: RiskProfile, selected: RiskProfile, profile_mode: str
    ) -> bool:
        """Restricted fallback: never more privileged, same profile_mode.

        existing_profile leases never silently downgrade to throwaway: with
        ``profile_mode == existing_profile`` the fallback candidate must be
        exactly as privileged (existing_profile/controller_bound) — a
        throwaway would be a silent consent downgrade, so it fails closed
        instead (mirrors browser_tool's "consented user must never be
        silently downgraded to a throwaway").
        """
        if profile_mode == "existing_profile":
            return candidate in (RiskProfile.EXISTING_PROFILE, RiskProfile.CONTROLLER_BOUND)
        return _RISK_RANK.get(candidate, 0) <= _RISK_RANK.get(selected, 0)

    # -- backend resolution -------------------------------------------------

    def resolve_backend(
        self,
        action: BrowserAction,
        lease: Optional[Any],
        decision: AuthorizationDecision,
    ) -> Optional[BrowserBackendAdapter]:
        """Select the winning backend for an already-authorized action.

        Order: extension lane (authoritative when bound) → declared lane
        order (dom → browser_exec → semantic → cua), restricted to adapters
        whose risk_profile fits the envelope.
        """
        if decision.is_deny:
            return None

        envelope = self.envelope(action, lease)
        profile_mode = envelope.get("profile_mode", "isolated")

        # Extension lane is authoritative when an authenticated controller is
        # bound for this action; missing/ambiguous scope fails closed.
        if self._extension_selector is not None:
            ext = self._extension_selector(action)
            if ext is not None:
                if ext.describes_capability(action.verb) and self._risk_allowed_in_envelope(
                    ext.risk_profile, envelope
                ):
                    return ext
                return None  # bound but mismatch => fail closed, no fallback

        # Respect preferred_backend if it is a real, capable, envelope-fit lane.
        if action.preferred_backend:
            preferred = next(
                (a for a in self._adapters if a.name == action.preferred_backend),
                None,
            )
            if preferred is not None and preferred.describes_capability(action.verb) and (
                self._risk_allowed_in_envelope(preferred.risk_profile, envelope)
            ):
                return preferred

        for lane in BACKEND_LANE_ORDER:
            for adapter in self._adapters:
                if adapter.name != lane:
                    continue
                if not adapter.describes_capability(action.verb):
                    continue
                if not self._risk_allowed_in_envelope(adapter.risk_profile, envelope):
                    continue
                return adapter
        return None

    def _fallback_candidates(
        self,
        action: BrowserAction,
        lease: Optional[Any],
        selected: BrowserBackendAdapter,
    ) -> List[BrowserBackendAdapter]:
        """Restricted fallback pool — subset of the envelope and same profile_mode."""
        envelope = self.envelope(action, lease)
        profile_mode = envelope.get("profile_mode", "isolated")
        pool = []
        for adapter in self._adapters:
            if adapter.name == selected.name:
                continue
            if not adapter.describes_capability(action.verb):
                continue
            if not self._risk_allowed_in_envelope(adapter.risk_profile, envelope):
                continue
            if not self._risk_fallback_allowed(
                adapter.risk_profile, selected.risk_profile, profile_mode
            ):
                continue
            pool.append(adapter)
        return pool

    # -- execution ----------------------------------------------------------

    def execute(
        self,
        action: BrowserAction,
        *,
        task_id: Optional[str] = None,
        run_id: Optional[str] = None,
        lease: Optional[Any] = None,
        grant: Optional[Dict[str, Any]] = None,
        allow_fallback: bool = True,
        evidence_adapter: Optional[Any] = None,
    ) -> ExecutionResult:
        """Full pipeline: decision → envelope → resolve → execute → fallback.

        Never falls back across profile modes; never falls back to a more
        privileged risk profile; never retries blindly after an ambiguous
        effect (caller must VERIFY BEFORE RETRY).
        """
        # 1. Resolve the lease if not provided. The kwargs default to the
        # action's own identity — the action IS the task/run context.
        if lease is None and self._lease_store is not None:
            lease = self._lease_store.get_by_ownership(
                task_id=task_id or action.task_id,
                run_id=run_id or action.run_id,
            )
        if lease is None and self._lease_store is not None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                backend_name="__broker__",
                backend_version="1.0",
                error=REASON_SESSION_LEASE_MISSING,
                failure_category=REASON_SESSION_LEASE_MISSING,
                retryable=False,
            )

        # 2. Decision before effect.
        decision = self.decide(
            action, lease=lease, grant=grant,
            capability_budget=self._effective_budget(lease),
        )
        if decision.is_deny:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                backend_name="__broker__",
                backend_version="1.0",
                error=decision.reason_code,
                failure_category=decision.reason_code,
                retryable=False,
            )
        if decision.is_require_approval:
            return ExecutionResult(
                status=ExecutionStatus.INTENT,
                backend_name="__broker__",
                backend_version="1.0",
                result={"decision": "require_approval", "reason_code": decision.reason_code},
                failure_category=REASON_HUMAN_APPROVAL_REQUIRED,
                retryable=False,
            )

        # 3. Resolve backend.
        selected = self.resolve_backend(action, lease, decision)
        if selected is None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                backend_name="__broker__",
                backend_version="1.0",
                error=REASON_BACKEND_UNAVAILABLE,
                failure_category=REASON_BACKEND_UNAVAILABLE,
                retryable=True,
            )

        # 4. Execute (attempt) with restricted fallback.
        envelope = self.envelope(action, lease)
        attempt = ExecutionAttempt(action=action, decision=decision, envelope=envelope)
        try:
            result = selected.execute(attempt)
        except BackendExecutionError as exc:
            result = ExecutionResult(
                status=ExecutionStatus.FAILED,
                backend_name=selected.name,
                backend_version=selected.version,
                error=str(exc),
                failure_category=exc.category,
                retryable=exc.retryable,
            )
        except Exception as exc:  # unknown adapter failure
            logger.warning("browser adapter %s raised: %s", selected.name, exc, exc_info=True)
            result = ExecutionResult(
                status=ExecutionStatus.UNKNOWN,
                backend_name=selected.name,
                backend_version=selected.version,
                error=REASON_RECONCILIATION_REQUIRED,
                failure_category=REASON_RECONCILIATION_REQUIRED,
                retryable=False,
            )

        # 5. Restricted fallback on retryable failure only.
        if allow_fallback and not result.succeeded and result.retryable:
            for candidate in self._fallback_candidates(action, lease, selected):
                attempt = ExecutionAttempt(action=action, decision=decision, envelope=envelope)
                try:
                    result = candidate.execute(attempt)
                    result.backend_name = candidate.name
                    result.backend_version = candidate.version
                    logger.info(
                        "browser broker: fell back %s -> %s for %s",
                        selected.name, candidate.name, action.verb.value,
                    )
                    break
                except BackendExecutionError as exc:
                    result = ExecutionResult(
                        status=ExecutionStatus.FAILED,
                        backend_name=candidate.name,
                        backend_version=candidate.version,
                        error=str(exc),
                        failure_category=exc.category,
                        retryable=exc.retryable,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("browser fallback %s raised: %s", candidate.name, exc, exc_info=True)
                    result = ExecutionResult(
                        status=ExecutionStatus.UNKNOWN,
                        backend_name=candidate.name,
                        backend_version=candidate.version,
                        error=REASON_RECONCILIATION_REQUIRED,
                        failure_category=REASON_RECONCILIATION_REQUIRED,
                        retryable=False,
                    )

        # 6. Normalize through the evidence adapter when provided.
        if evidence_adapter is not None and result.status not in (ExecutionStatus.INTENT,):
            try:
                receipt = evidence_adapter.build_receipt(action, result, lease=lease, decision=decision)
                # ``receipt`` is a BrowserReceipt dataclass (tools.browser_evidence),
                # NOT a plain dict — read the browser block attribute directly.
                browser_block = receipt.browser if hasattr(receipt, "browser") else receipt.get("browser", {})
                result.evidence = browser_block.get("evidence_refs", []) or []
            except Exception as exc:  # evidence must never break execution
                logger.warning("evidence adapter failed: %s", exc, exc_info=True)
        return result