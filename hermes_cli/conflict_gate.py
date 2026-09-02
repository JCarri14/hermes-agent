"""conflict_gate — deterministic, read-only conflict-prediction gate.

Decides whether a candidate Kanban task (Developer work on a worktree
branch) may run in parallel with an already-running Developer task on the
same base project without raising merge conflicts.

Classification:
  PARALLEL_SAFE  - mergeable + no policy overlay fires.
  SERIALIZE      - do not start the candidate now; retry after the active
                   Developer merges and branches are refreshed.
  UNKNOWN        - cannot determine safely; MUST be treated as SERIALIZE.

The gate is read-only and deterministic: it only reads git
(git merge-tree, git diff --name-only, git rev-parse --count) and never
mutates branches, the working tree, worktrees, or Kanban state. It performs
no LLM calls and no network calls.

UNKNOWN → SERIALIZE (never guess). This is a deliberate safety bias:
a conservative (over-serializing) gate can only cost throughput, never
correctness or destruction.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

# Sentinel classification strings.
PARALLEL_SAFE = "PARALLEL_SAFE"
SERIALIZE = "SERIALIZE"
UNKNOWN = "UNKNOWN"

# Read-only git timeout per call.
GIT_TIMEOUT_SECONDS = 5.0

# Deterministic policy overlays. Any hit => SERIALIZE.
MIGRATION_PREFIXES = (
    "supabase/migrations/",
    "supabase/migrations",
    "migrations/",
    "migrations",
)
# Auth / security / RLS / multi-tenancy sensitive-path markers.
AUTH_SECURITY_MARKERS = (
    "auth/",
    "rls",
    "row_level_security",
    "multi_tenant",
    "multi-tenant",
    "tenant",
    "permission",
    "security/",
    "gateway/",
    "policies/",
)
# Generated / shared artifacts that races commonly corrupt.
GENERATED_SHARED_MARKERS = (
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "go.sum",
    "Cargo.lock",
    "graphify/",
    "package.json",
    "tsconfig.json",
)


class ConflictGateError(Exception):
    """Raised when the gate cannot compute safely; callers downgrade to UNKNOWN."""


@dataclass
class GateInput:
    """Everything the gate needs. The dispatcher supplies these from card data
    and cheap DB reads; the gate itself stays pure and side-effect free."""

    repo_root: str
    base_ref: str  # e.g. "origin/dev"
    active_branches: List[str]  # branch names of running, same-base Developer cards
    candidate_branch: str  # the candidate's branch (will-be)
    candidate_manifest_paths: Optional[List[str]] = None  # declared target_paths from card
    same_domain_running: List[str] = field(default_factory=list)  # running card ids in same sensitive domain


@dataclass
class Verdict:
    cls: str
    reasons: List[str] = field(default_factory=list)


def _run_git(repo_root: str, args: List[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["git", "-C", repo_root] + args,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - timeout path
        raise ConflictGateError(f"git timed out: {args[:3]}")
    except FileNotFoundError as exc:  # pragma: no cover - git missing
        raise ConflictGateError("git binary not found")


def _branch_has_commits(repo_root: str, base_ref: str, branch: str) -> bool:
    p = _run_git(repo_root, ["rev-list", "--count", f"{base_ref}..{branch}"])
    if p.returncode != 0:
        raise ConflictGateError(f"rev-list failed for {branch}: {p.stderr.strip()[:200]}")
    try:
        return int((p.stdout or "0").strip() or "0") > 0
    except ValueError as exc:  # pragma: no cover
        raise ConflictGateError(f"rev-list non-numeric for {branch}: {exc}")


def _touched_paths(repo_root: str, base_ref: str, branch: str) -> List[str]:
    p = _run_git(repo_root, ["diff", "--name-only", f"{base_ref}...{branch}"])
    if p.returncode != 0:
        raise ConflictGateError(f"diff failed for {branch}: {p.stderr.strip()[:200]}")
    return [ln for ln in (p.stdout or "").splitlines() if ln.strip()]


def _merge_tree_has_conflicts(repo_root: str, a_branch: str, b_branch: str) -> bool:
    """True if merging a_branch and b_branch yields conflicts.

    git merge-tree --write-tree <a> <b>: exit 0 = clean, non-zero = conflicts.
    Verified against git 2.53 with a real content conflict (exit 1 + message)
    and a disjoint set (exit 0).
    """
    p = _run_git(repo_root, ["merge-tree", "--write-tree", a_branch, b_branch])
    if p.returncode == 0:
        return False
    # Non-zero exit => conflict (or a hard merge error). Both should serialize.
    return True


def _paths_overlap(a: List[str], b: List[str]) -> List[str]:
    return sorted(set(a) & set(b))


def _any_prefix(paths: List[str], prefixes) -> bool:
    return any(p.startswith(prefix) for p in paths for prefix in prefixes)


def _any_marker(paths: List[str], markers) -> bool:
    low = [p.lower() for p in paths]
    return any(any(m in p for m in markers) for p in low)


def evaluate(inp: GateInput) -> Verdict:
    reasons: List[str] = []

    # 0) No active same-base Developer -> nothing to conflict with.
    if not inp.active_branches:
        return Verdict(PARALLEL_SAFE, ["no active same-base Developer"])

    # Guard of commit sufficiency. merge-tree cannot predict a branch that has
    # no commits beyond base; if we have no declared manifest either we must
    # NOT guess => UNKNOWN (serialize). This is the false-positive guard for
    # two branches that still fork straight from the same dev.
    try:
        cand_has_commits = _branch_has_commits(inp.repo_root, inp.base_ref, inp.candidate_branch)
    except ConflictGateError as exc:
        return Verdict(UNKNOWN, [f"candidate commit check: {exc}"])

    declared = list(inp.candidate_manifest_paths or [])
    if cand_has_commits:
        try:
            diff_paths = _touched_paths(inp.repo_root, inp.base_ref, inp.candidate_branch)
        except ConflictGateError as exc:
            return Verdict(UNKNOWN, [f"candidate diff: {exc}"])
    else:
        diff_paths = []
        if not declared:
            reasons.append("candidate has no commits yet and no target_paths manifest")
            return Verdict(UNKNOWN, reasons)
    # Combine actual diff with the card's declared target_paths so policy overlays
    # (migrations/auth/generated) honour declared intent even before commits land.
    candidate_paths = sorted(set(diff_paths) | set(declared))

    active_infos = []
    for abranch in inp.active_branches:
        try:
            a_has = _branch_has_commits(inp.repo_root, inp.base_ref, abranch)
        except ConflictGateError as exc:
            return Verdict(UNKNOWN, [f"active commit check ({abranch}): {exc}"])
        if a_has:
            try:
                a_paths = _touched_paths(inp.repo_root, inp.base_ref, abranch)
            except ConflictGateError as exc:
                return Verdict(UNKNOWN, [f"active diff ({abranch}): {exc}"])
        else:
            a_paths = []  # active branch also has no commits yet
        active_infos.append((abranch, a_has, a_paths))

    # merge-tree conflicts (only when both sides have commits)
    for abranch, a_has, a_paths in active_infos:
        if cand_has_commits and a_has:
            try:
                _mt_conflict = _merge_tree_has_conflicts(inp.repo_root, abranch, inp.candidate_branch)
            except ConflictGateError as exc:
                # Fail-safe: a merge-tree error/timeout must degrade to UNKNOWN
                # (-> SERIALIZE), never propagate and risk crashing the dispatcher.
                return Verdict(UNKNOWN, [f"merge-tree ({abranch} vs candidate): {exc}"])
            if _mt_conflict:
                reasons.append(f"merge-tree conflict vs {abranch}")
                return Verdict(SERIALIZE, reasons)
        # Path overlap: commits (diff) or declared manifests.
        overlap = _paths_overlap(a_paths, candidate_paths)
        if overlap:
            reasons.append(f"overlapping paths with {abranch}: {overlap[:8]}")
            return Verdict(SERIALIZE, reasons)

    # Deterministic policy overlays (candidate's touched + declared paths).
    if _any_prefix(candidate_paths, MIGRATION_PREFIXES):
        reasons.append("candidate touches migrations")
        return Verdict(SERIALIZE, reasons)
    if _any_marker(candidate_paths, AUTH_SECURITY_MARKERS):
        reasons.append("candidate touches auth/security/RLS")
        return Verdict(SERIALIZE, reasons)
    if _any_prefix(candidate_paths, GENERATED_SHARED_MARKERS):
        reasons.append("candidate touches generated/shared artifacts")
        return Verdict(SERIALIZE, reasons)
    if inp.same_domain_running:
        reasons.append(f"same sensitive domain as running: {inp.same_domain_running}")
        return Verdict(SERIALIZE, reasons)

    return Verdict(PARALLEL_SAFE, reasons)


# ─────────────────────────── dispatcher-side helpers ───────────────────────────
# These are the only two functions that touch Kanban/DB concerns. They are
# read-only (they never create directories or mutate rows). The core
# ``evaluate()`` above is pure and side-effect free.
#
# Same-base resolution: a card is "on the same base" as another when they share
# the same ``project_id`` (same repo/worktree base). The gate serializes the
# SECOND Developer on a base; the first never passes through ``evaluate``.

def _parse_target_paths_manifest(conn, task_id) -> list:
    """Best-effort parse of a card's body for a ``target_paths`` JSON list.
    Returns [] when absent/unparseable (the gate then relies on git diff or
    UNKNOWN->SERIALIZE). Never raises."""
    try:
        row = conn.execute("SELECT body FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None or not row["body"]:
            return []
        body = row["body"]
        # Accept a JSON array on its own line keyed `"target_paths": [...]`.
        import json, re
        m = re.search(r'"target_paths"\s*:\s*(\[[^\]]*\])', body)
        if not m:
            return []
        parsed = json.loads(m.group(1))
        if isinstance(parsed, list):
            return [str(p) for p in parsed if isinstance(p, str)]
        return []
    except Exception:
        return []


def _resolve_gate_repo_root(conn, task_id, cand, *, board=None) -> str:
    """Resolve the git repo root the gate will run against, WITHOUT creating
    anything. Prefers the card's already-set workspace_path; else None (the
    gate then returns UNKNOWN -> SERIALIZE)."""
    try:
        wp = cand["workspace_path"] if cand and cand["workspace_path"] else None
        if wp and os.path.isdir(wp):
            # Only accept the path if it plausibly is a git tree (root has .git,
            # or is a linked worktree). We do not create a worktree here.
            if os.path.isdir(os.path.join(wp, ".git")):
                return wp
            import subprocess as _sp
            try:
                p = _sp.run(
                    ["git", "-C", wp, "rev-parse", "--show-toplevel"],
                    capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS,
                    check=False,
                )
                if p.returncode == 0 and p.stdout.strip():
                    return p.stdout.strip()
            except Exception:
                return ""
        return ""
    except Exception:
        return ""
