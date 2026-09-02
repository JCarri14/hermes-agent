"""Board→project scoping in kanban_db.

A kanban board can be scoped to a first-class Hermes project so every task on
it anchors to that project (deterministic worktree + branch). Covers the
metadata round-trip and the create-time inheritance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


def test_board_metadata_project_id_roundtrip(fresh_home):
    assert kb.read_board_metadata("default").get("project_id") is None

    kb.write_board_metadata("default", project_id="p_abc123")
    assert kb.read_board_metadata("default")["project_id"] == "p_abc123"

    # None leaves unchanged; "" clears.
    kb.write_board_metadata("default", name="Still Here")
    assert kb.read_board_metadata("default")["project_id"] == "p_abc123"
    kb.write_board_metadata("default", project_id="")
    assert kb.read_board_metadata("default")["project_id"] is None


def test_create_board_accepts_project_id(fresh_home):
    meta = kb.create_board("proj-board", name="Proj Board", project_id="p_xyz")
    assert meta["project_id"] == "p_xyz"
    assert kb.read_board_metadata("proj-board")["project_id"] == "p_xyz"


def test_create_task_inherits_board_project(fresh_home, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with pdb.connect_closing() as pconn:
        proj_id = pdb.create_project(pconn, name="Widget", primary_path=str(repo))

    kb.create_board("scoped", name="Scoped", project_id=proj_id)
    conn = kb.connect(board="scoped")
    try:
        tid = kb.create_task(conn, title="inherit me", board="scoped")
        assert kb.get_task(conn, tid).project_id == proj_id
    finally:
        conn.close()


def test_create_task_project_mismatch_hard_error(fresh_home, tmp_path):
    """(D) board bound to project A + task project B => HARD ERROR, no task."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    with pdb.connect_closing() as pconn:
        board_proj = pdb.create_project(pconn, name="BoardProj", primary_path=str(tmp_path / "a"))
        task_proj = pdb.create_project(pconn, name="TaskProj", primary_path=str(tmp_path / "b"))

    kb.create_board("scoped2", name="Scoped2", project_id=board_proj)
    conn = kb.connect(board="scoped2")
    try:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="is project-bound"):
            kb.create_task(conn, title="mismatch", board="scoped2", project_id=task_proj)
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert before == after  # no task artifact created
    finally:
        conn.close()


def test_create_task_board_does_not_exist_hard_error(fresh_home, tmp_path):
    """(E) project valid but declared board does not exist => HARD ERROR."""
    (tmp_path / "a").mkdir()
    with pdb.connect_closing() as pconn:
        proj_id = pdb.create_project(pconn, name="P", primary_path=str(tmp_path / "a"))
    conn = kb.connect(board="ghost-board")
    try:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="does not exist"):
            kb.create_task(conn, title="ghost board", board="ghost-board", project_id=proj_id)
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert before == after
    finally:
        conn.close()


def test_create_task_project_not_in_registry_hard_error(fresh_home, tmp_path):
    """(C) requested project id/slug does not resolve => HARD ERROR, no graceful degrade."""
    (tmp_path / "a").mkdir()
    with pdb.connect_closing() as pconn:
        proj_id = pdb.create_project(pconn, name="Real", primary_path=str(tmp_path / "a"))
    kb.create_board("legacy", name="Legacy")  # non-project-bound board
    conn = kb.connect(board="legacy")
    try:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="does not exist in the projects registry"):
            kb.create_task(conn, title="ghost proj", board="legacy", project_id="does-not-exist-xyz")
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert before == after
    finally:
        conn.close()


def test_create_task_valid_project_on_legacy_board(fresh_home, tmp_path):
    """(A) valid project + non-project-bound board => works, project-linked."""
    repo = tmp_path / "repo"
    repo.mkdir()
    with pdb.connect_closing() as pconn:
        proj_id = pdb.create_project(pconn, name="ERP", primary_path=str(repo))
    kb.create_board("erp", name="ERP", project_id=proj_id)
    conn = kb.connect(board="erp")
    try:
        tid = kb.create_task(conn, title="valid", board="erp", project_id=proj_id)
        t = kb.get_task(conn, tid)
        assert t is not None
        assert t.project_id == proj_id
        assert t.workspace_kind == "worktree"
        assert str(t.workspace_path).endswith(f".worktrees/{tid}")
        assert t.branch_name and t.branch_name.startswith("erp/")
    finally:
        conn.close()


def test_create_task_board_bound_project_inherited(fresh_home, tmp_path):
    """(B) board project-bound + project omitted => derived from board.project_id."""
    repo = tmp_path / "repo"
    repo.mkdir()
    with pdb.connect_closing() as pconn:
        proj_id = pdb.create_project(pconn, name="Scoped", primary_path=str(repo))
    kb.create_board("scoped", name="Scoped", project_id=proj_id)
    conn = kb.connect(board="scoped")
    try:
        tid = kb.create_task(conn, title="inherit", board="scoped")
        assert kb.get_task(conn, tid) is not None
        assert kb.get_task(conn, tid).project_id == proj_id
    finally:
        conn.close()


def test_create_task_legacy_scratch_still_works(fresh_home, tmp_path):
    """Legacy: no project context on a non-project-bound board => scratch OK."""
    kb.create_board("lab", name="Lab")  # board has no project_id
    conn = kb.connect(board="lab")
    try:
        tid = kb.create_task(conn, title="scratch", board="lab")
        t = kb.get_task(conn, tid)
        assert t is not None
        assert t.project_id is None
        assert t.workspace_kind in ("scratch",)
    finally:
        conn.close()
