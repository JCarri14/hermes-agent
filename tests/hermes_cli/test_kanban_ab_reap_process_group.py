# D3 A+B regression tests — block/reclaim process-group reaping + forensic identity.
# Covers the demonstrated production failure (DOUBLE_OWNER via block-no-reap) and
# the D4 correction set: killpg (parent+children), self-vs-external ordering,
# identity snapshot before state mutation, SIGTERM→SIGKILL escalation.
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hermes_cli.kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (mirrors repo convention)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _is_reapable(pid):
    return kb._worker_pgid(pid) is not None


def _spawn_leader_with_child(marker):
    """Session-leader 'worker' that forks a child, mirrors start_new_session=True."""
    code = (
        "import subprocess,time,sys\n"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        f"open({marker!r}+'.childpid','w').write(str(p.pid))\n"
        "time.sleep(120)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    childpid = None
    for _ in range(50):
        try:
            childpid = int(Path(marker + ".childpid").read_text().strip())
            break
        except (OSError, ValueError):
            time.sleep(0.1)
    return proc.pid, childpid, proc


def _cleanup(leader, childpid, proc):
    if childpid and kb._pid_alive(childpid):
        try:
            os.kill(childpid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    pgid = kb._worker_pgid(leader) if leader else None
    if pgid and kb._pid_alive(leader):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def test_worker_pgid_identity(tmp_path):
    """A start_new_session worker IS its PGID leader; a normal child is not."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(5)"],
        start_new_session=True,
    )
    try:
        assert _is_reapable(proc.pid) is True
    finally:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(5)"])
    try:
        assert kb._worker_pgid(child.pid) is None  # not a group leader
    finally:
        child.kill()
        child.wait()


def test_terminate_reclaimed_reaps_parent_and_child(tmp_path):
    """A: _terminate_reclaimed_worker kills the process GROUP (parent+child),
    not just the leader — the exact double-writer failure (leader kill alone
    leaves the descendant writer alive)."""
    marker = str(tmp_path / "w")
    leader, childpid, proc = _spawn_leader_with_child(marker)
    try:
        assert leader and childpid
        assert _is_reapable(leader)
        assert os.path.exists(marker + ".childpid")
        host_prefix = kb._claimer_id().split(":", 1)[0]
        info = kb._terminate_reclaimed_worker(leader, f"{host_prefix}:t")
        assert info.get("terminated") is True, info
        time.sleep(0.3)
        assert not kb._pid_alive(leader)
        assert not kb._pid_alive(childpid), "descendant survived leader reap → double-writer"
    finally:
        _cleanup(leader, childpid, proc)


def test_sigterm_graceful_then_sigkill_escalation():
    """A cooperative group dies on SIGTERM (no SIGKILL); escalation only if needed."""
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import signal,time,os\n"
         "signal.signal(signal.SIGTERM, lambda *a: (time.sleep(0.05), os._exit(0)))\n"
         "time.sleep(60)"],
        start_new_session=True,
    )
    host_prefix = kb._claimer_id().split(":", 1)[0]
    try:
        info = kb._terminate_reclaimed_worker(proc.pid, f"{host_prefix}:t")
        assert info.get("terminated") is True, info
        assert info.get("sigkill") in (False, None), info
    finally:
        _cleanup(proc.pid, None, proc)


def test_identity_needs_hostlocal_claim():
    """_terminate_reclaimed_worker returns un-terminated host_local=False for a
    foreign claim (never signals an unrelated host's worker)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(5)"], start_new_session=True,
    )
    try:
        info = kb._terminate_reclaimed_worker(proc.pid, "otherhost:xxx")
        assert info["host_local"] is False
        assert info["terminated"] is False
        assert kb._pid_alive(proc.pid), "foreign claim must not be signalled"
    finally:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()


def test_end_run_preserves_worker_identity_in_metadata(tmp_path, kanban_home):
    """B/D4 C4: _end_run snapshots worker_pid/claim_lock into run metadata
    before the UPDATE nulls them — forensic handle survives the transition."""
    conn = kb.connect()  # isolated HERMES_HOME DB from fixture
    task_id = kb.create_task(
        conn, title="identity", assignee="default", workspace_kind="scratch",
    )
    # claim the task (records a run with worker_pid + claim_lock on tasks)
    kb.claim_task(conn, task_id, ttl_seconds=300)
    # simulate the run having a real worker pid/claim (mimic external spawn)
    host_prefix = kb._claimer_id().split(":", 1)[0]
    conn.execute(
        "UPDATE tasks SET worker_pid=?, claim_lock=? WHERE id=?",
        (424242, f"{host_prefix}:fk", task_id),
    )
    conn.execute(
        "UPDATE task_runs SET worker_pid=?, claim_lock=? "
        "WHERE task_id=? AND ended_at IS NULL",
        (424242, f"{host_prefix}:fk", task_id),
    )
    # end the run via _end_run (as a block/complete/abort would)
    run_id = kb._current_run_id(conn, task_id)
    assert run_id is not None
    kb._end_run(conn, task_id, outcome="blocked", status="blocked", summary="test")
    # run row: worker_pid is now NULL but metadata preserves prev identity
    run = conn.execute(
        "SELECT worker_pid, metadata FROM task_runs WHERE id=?", (run_id,)
    ).fetchone()
    assert run["worker_pid"] is None
    import json
    meta = json.loads(run["metadata"] or "{}")
    assert meta.get("prev_worker_pid") == 424242, meta
    assert meta.get("prev_claim_lock") == f"{host_prefix}:fk", meta
    conn.close()


def test_external_block_reaps_obsolete_owner(tmp_path, kanban_home):
    """A+B/D4 C2: externally-invoked block reaps the obsolete worker's process
    group — no other live executor may survive.

    This is the RED-side regression fixture for the production incident
    (LIVE_INCIDENT_DOUBLE_OWNER_BLOCK_NO_REAP)."""
    conn = kb.connect()
    task_id = kb.create_task(
        conn, title="extblock", assignee="default", workspace_kind="scratch",
    )
    kb.claim_task(conn, task_id, ttl_seconds=300)
    host_prefix = kb._claimer_id().split(":", 1)[0]
    marker = str(tmp_path / "ext")
    leader, childpid, proc = _spawn_leader_with_child(marker)
    try:
        conn.execute(
            "UPDATE tasks SET worker_pid=?, claim_lock=? WHERE id=?",
            (leader, f"{host_prefix}:t", task_id),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid=?, claim_lock=? "
            "WHERE task_id=? AND ended_at IS NULL",
            (leader, f"{host_prefix}:t", task_id),
        )
        conn.commit()
        assert leader != os.getpid()
        ok = kb.block_task(conn, task_id, reason="ext block")
        assert ok
        time.sleep(0.3)
        assert not kb._pid_alive(leader), "obsolete owner survived external block"
        assert not kb._pid_alive(childpid), "descendant writer survived external block"
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
    finally:
        _cleanup(leader, childpid, proc)
        conn.close()


def test_external_block_stale_run_id_does_not_kill_current_owner(tmp_path, kanban_home):
    """D4 QA correction 1: a stale/duplicate caller whose `expected_run_id` does
    NOT match the live run must NEVER reap the current legitimate owner.

    Regression for the reverse double-owner: old worker A crashed; new worker B
    owns run2; a stale caller (remembering run1) calls block → the guarded
    UPDATE fails on the run_id mismatch → block_task returns False and must NOT
    have killed B's process group (the SNAPSHOTTED pid is B, but the reap is
    gated on the transition applying, so it must not fire)."""
    conn = kb.connect()
    task_id = kb.create_task(
        conn, title="staleblock", assignee="default", workspace_kind="scratch",
    )
    kb.claim_task(conn, task_id, ttl_seconds=300)
    run_id = kb._current_run_id(conn, task_id)
    assert run_id is not None
    host_prefix = kb._claimer_id().split(":", 1)[0]
    # B = a NEW legitimate worker, a live session leader with a child writer.
    marker = str(tmp_path / "B")
    leaderB, childB, procB = _spawn_leader_with_child(marker)
    try:
        conn.execute(
            "UPDATE tasks SET worker_pid=?, claim_lock=? WHERE id=?",
            (leaderB, f"{host_prefix}:t", task_id),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid=?, claim_lock=? "
            "WHERE id=?",
            (leaderB, f"{host_prefix}:t", run_id),
        )
        conn.commit()
        assert leaderB != os.getpid()
        # A stale caller calls block with expected_run_id = run1 (a non-existent
        # / stale run id that does NOT match the live run). The guarded UPDATE
        # (ADD current_run_id = expected) must fail → return False.
        stale_run = run_id + 999
        ok = kb.block_task(
            conn, task_id, reason="stale", expected_run_id=stale_run,
        )
        assert ok is False, "stale block must not apply"
        time.sleep(0.3)
        # B and its child must STILL be alive — the reap must not have fired.
        assert kb._pid_alive(leaderB), "stale block killed current owner B"
        assert kb._pid_alive(childB), "stale block killed B's descendant"
        # board still running / owned by B
        task = kb.get_task(conn, task_id)
        assert task.status in ("running", "ready")
    finally:
        _cleanup(leaderB, childB, procB)
        conn.close()