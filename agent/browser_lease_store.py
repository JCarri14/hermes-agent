"""AUTHENTICATED_SESSION_BOUNDARY — browser session lease store.

Implements the session lease contract
(``BROWSER_EXECUTION_CAPABILITY_V1.md`` §3.3, §5):

- **Lease per (task, run)** — single-owner with CAS semantics
  (unique ``(task_id, run_id)``; a second claim denies).
- **State machine** ``REQUESTED → ACTIVE → AUTH_REQUIRED/EXPIRED/REVOKED/
  RELEASED`` with validated transitions (§5.2).
- **SESSION ACCESS != ACTION AUTHORIZATION**: an ACTIVE lease enables reads
  on the domains it declares; it never authorizes side effects (that is the
  broker's GEA decision's job).
- **No inheritance**: subagents never see a parent lease directly; a child
  card needs its own lease with an explicit grant.
- **Projection, no new table**: transitions are emitted as additive
  ``session`` events (kanban ``task_events`` appendix schema, §5.2) through
  an injectable ``event_sink``. The store itself holds the live registry in
  memory; durability comes from the append-only sink, matching the GEA
  "receipt = projection" rule.
- **Cleanup**: expiry sweep (TTL + inactivity), RELEASED on task complete /
  consent-off, REVOKED is terminal (no re-activation).

Secrets never appear here: ``consent_ref``/``grant_ref`` are interaction
refs, not credentials; domains are plain hostnames.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class LeaseStatus(str, Enum):
    """BrowserSessionLease status (§3.3)."""

    REQUESTED = "requested"
    ACTIVE = "active"
    AUTH_REQUIRED = "auth_required"
    EXPIRED = "expired"
    REVOKED = "revoked"
    RELEASED = "released"


#: Statuses that no longer permit execution.
TERMINAL_STATUSES = frozenset({LeaseStatus.EXPIRED, LeaseStatus.REVOKED, LeaseStatus.RELEASED})

#: Non-terminal statuses.
NON_TERMINAL_STATUSES = frozenset(
    {LeaseStatus.REQUESTED, LeaseStatus.ACTIVE, LeaseStatus.AUTH_REQUIRED}
)

#: Valid transition map (§5.2). Missing transitions are rejected.
_ALLOWED_TRANSITIONS: Dict[LeaseStatus, frozenset] = {
    LeaseStatus.REQUESTED: frozenset(
        {LeaseStatus.ACTIVE, LeaseStatus.RELEASED}
    ),
    LeaseStatus.ACTIVE: frozenset(
        {
            LeaseStatus.RELEASED,
            LeaseStatus.EXPIRED,
            LeaseStatus.AUTH_REQUIRED,
            LeaseStatus.REVOKED,
        }
    ),
    LeaseStatus.AUTH_REQUIRED: frozenset(
        {LeaseStatus.ACTIVE, LeaseStatus.REVOKED, LeaseStatus.RELEASED}
    ),
    LeaseStatus.EXPIRED: frozenset(),  # terminal (re-grant via new lease)
    LeaseStatus.REVOKED: frozenset(),  # terminal; no re-activation
    LeaseStatus.RELEASED: frozenset(),
}

#: Default inactivity timeout (matches browser_tool.py:1946-1967).
DEFAULT_INACTIVITY_TIMEOUT_S = 120

#: Default TTL for a lease when expired_at is not provided (8h).
DEFAULT_LEASE_TTL_S = 8 * 60 * 60


class LeaseTransitionError(Exception):
    """Raised when a lease transition violates the state machine."""


class LeaseConflictError(Exception):
    """Raised when a second (task, run) claims an existing lease (CAS deny)."""


# Event sink signature: ``sink(lease, from_status, to_status, reason, consent_ref)``
# Returns nothing; failures must never break the store (logged, not raised).
EventSink = Callable[..., None]


@dataclass
class BrowserSessionLease:
    """Mutable lease record (§3.3). Ownership is (task_id, run_id)."""

    lease_id: str
    task_id: str
    run_id: str
    profile_mode: str = "isolated"  # isolated | existing_profile | cloud | cdp_override
    backend_lane: str = "dom"       # dom | browser_exec | cua | extension
    domains: List[str] = field(default_factory=list)
    accounts: Optional[List[str]] = None
    principal: str = "HUMAN"        # the session belongs to the human
    status: LeaseStatus = LeaseStatus.REQUESTED
    consent_ref: Optional[str] = None          # HUMAN_AUTHORIZATION_PROVENANCE
    grant_ref: Optional[str] = None            # GEA authorization_id (session_attach)
    revocation_ref: Optional[str] = None
    capability_budget: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    activated_at: Optional[float] = None
    expires_at: Optional[float] = None
    last_activity_at: Optional[float] = None
    inactivity_timeout_s: int = DEFAULT_INACTIVITY_TIMEOUT_S
    reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- status helpers -----------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.status is LeaseStatus.ACTIVE

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_revoked(self) -> bool:
        return self.status is LeaseStatus.REVOKED

    def touch(self) -> None:
        self.last_activity_at = time.time()

    def domain_declared(self, domain: str) -> bool:
        return (domain or "").strip().lower() in {d.strip().lower() for d in self.domains}

    def is_expired_wallclock(self, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.time()
        if self.status is LeaseStatus.ACTIVE and self.expires_at is not None:
            if now >= self.expires_at:
                return True
        if self.status is LeaseStatus.ACTIVE and self.last_activity_at is not None:
            if now - self.last_activity_at > self.inactivity_timeout_s:
                return True
        return False

    def as_dict(self) -> Dict[str, Any]:
        """Redacted projection — never includes secrets, only refs."""
        return {
            "lease_id": self.lease_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "profile_mode": self.profile_mode,
            "backend_lane": self.backend_lane,
            "domains": list(self.domains),
            "accounts": list(self.accounts) if self.accounts else None,
            "principal": self.principal,
            "status": self.status.value,
            "consent_ref": self.consent_ref,
            "grant_ref": self.grant_ref,
            "revocation_ref": self.revocation_ref,
            "created_at": self.created_at,
            "activated_at": self.activated_at,
            "expires_at": self.expires_at,
            "last_activity_at": self.last_activity_at,
            "inactivity_timeout_s": self.inactivity_timeout_s,
            "reason": self.reason,
        }


class BrowserLeaseStore:
    """In-memory single-owner lease registry with CAS semantics.

    Concurrency: a module-level ``threading.RLock`` guards every mutation.
    CAS is implemented as compare-and-set on the ``status`` field: a claim
    only succeeds when the current status is unchanged.

    The ``event_sink`` (optional) receives every transition as an additive
    ``session`` event — the design's projection onto ``task_events``
    without a new table. Sink failures are logged and never raise.
    """

    def __init__(
        self,
        *,
        event_sink: Optional[EventSink] = None,
        inactivity_timeout_s: Optional[int] = None,
        lease_ttl_s: Optional[int] = None,
        now_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        self._leases: Dict[str, BrowserSessionLease] = {}
        self._lock = threading.RLock()
        self._event_sink = event_sink
        self._inactivity_timeout_s = (
            inactivity_timeout_s if inactivity_timeout_s is not None else DEFAULT_INACTIVITY_TIMEOUT_S
        )
        self._lease_ttl_s = lease_ttl_s if lease_ttl_s is not None else DEFAULT_LEASE_TTL_S
        self._now = now_provider or time.time

    # -- low-level helpers ----------------------------------------------------

    def _emit(self, lease: BrowserSessionLease, from_status: LeaseStatus, to_status: LeaseStatus, reason: Optional[str]) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(
                lease=lease.as_dict(),  # JSON-serializable, redacted projection
                from_status=from_status.value,
                to_status=to_status.value,
                reason=reason,
                consent_ref=lease.consent_ref,
            )
        except Exception:  # projection must never break the lease lifecycle
            logger.warning("lease event_sink failed (ignored): %s", reason, exc_info=True)

    def _transition(
        self,
        lease: BrowserSessionLease,
        to_status: LeaseStatus,
        *,
        reason: Optional[str] = None,
        set_fields: Optional[Dict[str, Any]] = None,
    ) -> BrowserSessionLease:
        with self._lock:
            from_status = lease.status
            if to_status not in _ALLOWED_TRANSITIONS.get(from_status, frozenset()):
                raise LeaseTransitionError(
                    f"invalid lease transition {from_status.value} -> {to_status.value} "
                    f"for lease {lease.lease_id}"
                )
            if set_fields:
                for key, value in set_fields.items():
                    setattr(lease, key, value)
            lease.status = to_status
            lease.reason = reason
            self._emit(lease, from_status, to_status, reason)
            return lease

    # -- public API -----------------------------------------------------------

    def request(
        self,
        *,
        task_id: str,
        run_id: str,
        domains: Sequence[str],
        profile_mode: str = "isolated",
        backend_lane: str = "dom",
        accounts: Optional[Sequence[str]] = None,
        consent_ref: Optional[str] = None,
        grant_ref: Optional[str] = None,
        capability_budget: Optional[Dict[str, Any]] = None,
        lease_id: Optional[str] = None,
        expires_at: Optional[float] = None,
    ) -> BrowserSessionLease:
        """Create a REQUESTED lease. Single-owner: a second claim denies.

        Raises :class:`LeaseConflictError` when ``(task_id, run_id)`` is
        already present and not terminal-released. Terminal leases are
        replaced by a NEW lease with a NEW lease_id (a fresh authorization,
        never a re-activation of the old one).
        """
        with self._lock:
            existing = self._find_by_ownership(task_id, run_id)
            if existing is not None and not existing.is_terminal:
                raise LeaseConflictError(
                    f"lease already exists for ({task_id}, {run_id}) "
                    f"[{existing.status.value}]; single-owner CAS denies the second claim"
                )
            if not domains:
                raise ValueError("a lease must declare at least one domain")
            default_id = lease_id or f"lease-{task_id}-{run_id}"
            if default_id in self._leases:
                # A terminal lease with the same task/run was replaced: mint a
                # fresh id so the new authorization is never confused with the
                # old one in receipts/logs.
                default_id = f"{default_id}-{int(self._now() * 1000)}"
            lease = BrowserSessionLease(
                lease_id=default_id,
                task_id=task_id,
                run_id=run_id,
                profile_mode=profile_mode,
                backend_lane=backend_lane,
                domains=[d.strip().lower() for d in domains],
                accounts=list(accounts) if accounts else None,
                consent_ref=consent_ref,
                grant_ref=grant_ref,
                capability_budget=capability_budget,
                expires_at=expires_at if expires_at is not None else self._now() + self._lease_ttl_s,
                last_activity_at=self._now(),
                inactivity_timeout_s=self._inactivity_timeout_s,
            )
            self._leases[lease.lease_id] = lease
            self._emit(lease, LeaseStatus.REQUESTED, LeaseStatus.REQUESTED, "requested")
            return lease

    def activate(
        self,
        lease: BrowserSessionLease,
        *,
        grant: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> BrowserSessionLease:
        """REQUESTED -> ACTIVE. For ``existing_profile`` a grant is required.

        Fail-closed: an existing_profile lease without a human consent_ref
        or GEA grant_ref cannot activate (design §5.3: existing-profile only
        with host-side grant + approval).
        """
        if lease.profile_mode == "existing_profile":
            # Both of these represent the same provenance chain; at least one
            # must be present. consent_ref = interaction_ref (human approval);
            # grant_ref = GEA authorization_id (session_attach).
            approval = lease.consent_ref or lease.grant_ref
            if not approval:
                raise LeaseTransitionError(
                    "existing_profile lease requires consent_ref or grant_ref "
                    "(host-side grant + human approval) before activation"
                )
            if grant is not None and grant.get("revoked"):
                raise LeaseTransitionError("grant is revoked — cannot activate lease")
        return self._transition(
            lease,
            LeaseStatus.ACTIVE,
            reason=reason or "grant",
            set_fields={"activated_at": self._now(), "last_activity_at": self._now()},
        )

    def require_auth(
        self, lease: BrowserSessionLease, *, reason: str = "session_lost"
    ) -> BrowserSessionLease:
        """ACTIVE -> AUTH_REQUIRED (401/redirect-to-login/lost session).

        The agent never auto-re-logs-in with its own credentials; a human
        re-consent (GEA ``SESSION_RECONSENT_REQUIRED``) is required to
        return to ACTIVE.
        """
        return self._transition(lease, LeaseStatus.AUTH_REQUIRED, reason=reason)

    def reconsent(
        self, lease: BrowserSessionLease, *, consent_ref: str, reason: Optional[str] = None
    ) -> BrowserSessionLease:
        """AUTH_REQUIRED -> ACTIVE after human re-consent (explicit, not automatic)."""
        if not consent_ref:
            raise LeaseTransitionError("re-consent requires a consent_ref (interaction_ref)")
        with self._lock:
            lease.consent_ref = consent_ref
            return self._transition(
                lease,
                LeaseStatus.ACTIVE,
                reason=reason or "human_reconsent",
                set_fields={"last_activity_at": self._now()},
            )

    def revoke(
        self,
        lease: BrowserSessionLease,
        *,
        revocation_ref: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> BrowserSessionLease:
        """Any non-terminal -> REVOKED (terminal; no re-activation)."""
        with self._lock:
            lease.revocation_ref = revocation_ref
            from_status = lease.status
            if from_status is LeaseStatus.REVOKED:
                return lease
            if from_status is LeaseStatus.RELEASED:
                raise LeaseTransitionError("cannot revoke a released lease")
            if to_ok := _ALLOWED_TRANSITIONS.get(from_status):
                if LeaseStatus.REVOKED not in to_ok:
                    raise LeaseTransitionError(
                        f"invalid transition {from_status.value} -> revoked"
                    )
            return self._transition(
                lease, LeaseStatus.REVOKED, reason=reason or "revoked"
            )

    def release(
        self, lease: BrowserSessionLease, *, reason: Optional[str] = None
    ) -> BrowserSessionLease:
        """ACTIVE/AUTH_REQUIRED/REQUESTED -> RELEASED (task complete / consent-off)."""
        return self._transition(lease, LeaseStatus.RELEASED, reason=reason or "task_complete")

    def expire(self, lease: BrowserSessionLease, *, reason: str = "ttl") -> BrowserSessionLease:
        """ACTIVE -> EXPIRED (TTL / inactivity, wall-clock enforced)."""
        return self._transition(lease, LeaseStatus.EXPIRED, reason=reason)

    # -- read / sweep ----------------------------------------------------------

    def get(self, lease_id: str) -> Optional[BrowserSessionLease]:
        with self._lock:
            return self._leases.get(lease_id)

    def get_by_ownership(self, task_id: str, run_id: str) -> Optional[BrowserSessionLease]:
        with self._lock:
            return self._find_by_ownership(task_id, run_id)

    def _find_by_ownership(self, task_id: str, run_id: str) -> Optional[BrowserSessionLease]:
        for lease in self._leases.values():
            if lease.task_id == task_id and lease.run_id == run_id:
                return lease
        return None

    def list_active(self) -> List[BrowserSessionLease]:
        with self._lock:
            return [l for l in self._leases.values() if l.is_active]

    def sweep(self, now: Optional[float] = None) -> List[str]:
        """Expire ACTIVE leases whose TTL or inactivity timeout lapsed.

        Returns the lease_ids expired. Idempotent; safe to call from the
        cleanup thread / orphan reaper.
        """
        now = now if now is not None else self._now()
        expired_ids: List[str] = []
        with self._lock:
            for lease in list(self._leases.values()):
                if lease.status is LeaseStatus.ACTIVE and lease.is_expired_wallclock(now):
                    lease.status = LeaseStatus.EXPIRED
                    lease.reason = "expired_wallclock"
                    self._emit(lease, LeaseStatus.ACTIVE, LeaseStatus.EXPIRED, "expired_wallclock")
                    expired_ids.append(lease.lease_id)
        return expired_ids

    def drop(self, lease: BrowserSessionLease) -> None:
        """Remove a lease from the live registry (post-release cleanup)."""
        with self._lock:
            self._leases.pop(lease.lease_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._leases)