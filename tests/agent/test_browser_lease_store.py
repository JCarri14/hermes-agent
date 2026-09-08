"""Unit / contract / concurrency tests for the browser session lease store.

Covers design §5.2 (state machine), §8.1 (single-owner CAS, expiry race),
§5.3 (existing-profile activation gate) and §11 concurrency/recovery rows.
"""

import threading
import time

import pytest

from agent.browser_lease_store import (
    BrowserLeaseStore,
    LeaseConflictError,
    LeaseStatus,
    LeaseTransitionError,
)


@pytest.fixture
def store():
    return BrowserLeaseStore()


@pytest.fixture
def events():
    collected = []
    return collected


def recording_sink(collected):
    def sink(**payload):
        collected.append(payload)
    return sink


def make_lease(store, task_id="c1", run_id="r1", **kw):
    defaults = dict(task_id=task_id, run_id=run_id, domains=["example.com"])
    defaults.update(kw)
    return store.request(**defaults)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_requested_to_active(self, store):
        lease = make_lease(store)
        assert lease.status is LeaseStatus.REQUESTED
        store.activate(lease)
        assert lease.status is LeaseStatus.ACTIVE
        assert lease.activated_at is not None

    def test_requested_deny_fails_closed_released(self, store):
        lease = make_lease(store)
        store.release(lease, reason="deny")
        assert lease.status is LeaseStatus.RELEASED

    def test_active_to_allowed_transitions(self, store):
        lease = make_lease(store)
        store.activate(lease)
        for target in (LeaseStatus.RELEASED, LeaseStatus.EXPIRED, LeaseStatus.REVOKED):
            l2 = make_lease(store, task_id="x", run_id=f"r-{target.value}")
            store.activate(l2)
            if target is LeaseStatus.REVOKED:
                store.revoke(l2)
            elif target is LeaseStatus.EXPIRED:
                store.expire(l2)
            else:
                store.release(l2)
            assert l2.status is target

    def test_active_to_auth_required_and_reconsent(self, store):
        lease = make_lease(store)
        store.activate(lease)
        store.require_auth(lease, reason="redirect_login")
        assert lease.status is LeaseStatus.AUTH_REQUIRED
        # no auto re-login: reconsent requires an explicit interaction_ref
        with pytest.raises(LeaseTransitionError):
            store.reconsent(lease, consent_ref="")
        store.reconsent(lease, consent_ref="interaction-77")
        assert lease.status is LeaseStatus.ACTIVE

    def test_revoked_terminal_no_reactivation(self, store):
        lease = make_lease(store)
        store.activate(lease)
        store.revoke(lease, revocation_ref="rev-1")
        assert lease.is_revoked
        assert lease.is_terminal
        with pytest.raises(LeaseTransitionError):
            store.activate(lease)
        with pytest.raises(LeaseTransitionError):
            store.reconsent(lease, consent_ref="anything")

    def test_expired_terminal(self, store):
        lease = make_lease(store)
        store.activate(lease)
        store.expire(lease)
        assert lease.status is LeaseStatus.EXPIRED
        with pytest.raises(LeaseTransitionError):
            store.activate(lease)

    def test_invalid_transition_rejected(self, store):
        lease = make_lease(store)  # REQUESTED
        with pytest.raises(LeaseTransitionError):
            store.revoke(lease)  # REQUESTED -> REVOKED is not in the map
        # REQUESTED can go RELEASED (fail-closed) though
        store.release(lease)


# ---------------------------------------------------------------------------
# Single-owner CAS
# ---------------------------------------------------------------------------


class TestSingleOwnerCAS:
    def test_second_claim_denied(self, store):
        make_lease(store, task_id="c1", run_id="r1")
        with pytest.raises(LeaseConflictError):
            make_lease(store, task_id="c1", run_id="r1")

    def test_second_claim_denied_even_while_active(self, store):
        lease = make_lease(store, task_id="c1", run_id="r1")
        store.activate(lease)
        with pytest.raises(LeaseConflictError):
            make_lease(store, task_id="c1", run_id="r1")

    def test_terminal_lease_replaced_by_new_authorization(self, store):
        lease = make_lease(store, task_id="c1", run_id="r1")
        store.release(lease)  # terminal
        lease2 = make_lease(store, task_id="c1", run_id="r1")
        assert lease2.lease_id != lease.lease_id

    def test_different_runs_same_task_are_independent(self, store):
        l1 = make_lease(store, task_id="c1", run_id="r1")
        l2 = make_lease(store, task_id="c1", run_id="r2")
        assert l1.lease_id != l2.lease_id


# ---------------------------------------------------------------------------
# Existing-profile activation gate (§5.3)
# ---------------------------------------------------------------------------


class TestExistingProfileGate:
    def test_existing_profile_requires_consent_to_activate(self, store):
        lease = make_lease(store, profile_mode="existing_profile")
        with pytest.raises(LeaseTransitionError):
            store.activate(lease)

    def test_existing_profile_activates_with_consent_ref(self, store):
        lease = make_lease(
            store, profile_mode="existing_profile", consent_ref="interaction-42"
        )
        store.activate(lease)
        assert lease.is_active

    def test_existing_profile_activates_with_grant_ref(self, store):
        lease = make_lease(
            store, profile_mode="existing_profile", grant_ref="auth-9"
        )
        store.activate(lease)
        assert lease.is_active

    def test_existing_profile_activation_denied_when_grant_revoked(self, store):
        lease = make_lease(
            store, profile_mode="existing_profile", grant_ref="auth-9"
        )
        with pytest.raises(LeaseTransitionError):
            store.activate(lease, grant={"revoked": True})


# ---------------------------------------------------------------------------
# Expiry / inactivity sweep
# ---------------------------------------------------------------------------


class TestExpirySweep:
    def test_wallclock_ttl_expiry(self):
        now = [1000.0]
        store = BrowserLeaseStore(now_provider=lambda: now[0])
        lease = make_lease(store, expires_at=1100.0)
        store.activate(lease)
        assert store.sweep(now=now[0]) == []
        now[0] = 1200.0
        expired = store.sweep(now=now[0])
        assert lease.lease_id in expired
        assert lease.status is LeaseStatus.EXPIRED

    def test_inactivity_timeout_expiry(self):
        now = [1000.0]
        store = BrowserLeaseStore(now_provider=lambda: now[0], inactivity_timeout_s=120)
        lease = make_lease(store)
        store.activate(lease)
        now[0] = 1300.0  # 300s > 120s inactivity
        store.sweep(now=now[0])
        assert lease.status is LeaseStatus.EXPIRED

    def test_activity_keeps_lease_alive(self):
        now = [1000.0]
        store = BrowserLeaseStore(now_provider=lambda: now[0], inactivity_timeout_s=120)
        lease = make_lease(store, expires_at=now[0] + 100000)
        store.activate(lease)
        now[0] = 1100.0
        lease.touch()  # activity
        now[0] = 1110.0
        store.sweep(now=now[0])
        assert lease.status is LeaseStatus.ACTIVE


# ---------------------------------------------------------------------------
# Events (projection onto kanban task_events — additive, no new table)
# ---------------------------------------------------------------------------


class TestEventProjection:
    def test_sink_receives_additive_session_events(self):
        collected = []
        store = BrowserLeaseStore(event_sink=recording_sink(collected))
        lease = make_lease(store)
        store.activate(lease)
        store.revoke(lease, revocation_ref="r-1")
        assert len(collected) == 3  # requested, active, revoked
        assert collected[0]["from_status"] == "requested"
        assert collected[1]["to_status"] == "active"
        assert collected[2]["to_status"] == "revoked"
        assert collected[2]["lease"]["lease_id"] == lease.lease_id
        # no secrets in the payload
        for event in collected:
            assert "cookie" not in str(event).lower()
            assert "token" not in str(event).lower()

    def test_sink_failure_never_breaks_lifecycle(self):
        store = BrowserLeaseStore(
            event_sink=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        lease = make_lease(store)
        store.activate(lease)  # must not raise
        assert lease.is_active


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_parallel_claims_only_one_wins(self):
        store = BrowserLeaseStore()
        results = []
        lock = threading.Lock()

        def claim(i):
            try:
                l = make_lease(store, task_id="race", run_id="r1")
                with lock:
                    results.append(("ok", i, l.lease_id))
            except LeaseConflictError:
                with lock:
                    results.append(("conflict", i, None))

        threads = [threading.Thread(target=claim, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        oks = [r for r in results if r[0] == "ok"]
        conflicts = [r for r in results if r[0] == "conflict"]
        assert len(oks) == 1
        assert len(conflicts) == 7

    def test_expiry_race_never_reactivates(self):
        now = [1000.0]
        store = BrowserLeaseStore(now_provider=lambda: now[0])
        lease = make_lease(store, expires_at=1100.0)
        store.activate(lease)
        now[0] = 2000.0
        # two sweepers race; both must agree the lease is EXPIRED
        store.sweep(now=now[0])
        store.sweep(now=now[0])
        assert lease.status is LeaseStatus.EXPIRED
        with pytest.raises(LeaseTransitionError):
            store.activate(lease)