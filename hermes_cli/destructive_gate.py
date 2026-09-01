"""destructive_gate — deterministic, read-only pre-action destructive gate.

Decides whether a ready Kanban card that appears to perform a destructive /
irreversible action on a LIVE resource may be claimed + executed (spawned).

The ordering this gate enforces is the lab's correct policy for live /
destructive work:

    pre-verification -> human GO -> destructive action -> postcondition verification

It deliberately *blocks* the destructive action from even starting until a
human GO has been recorded on the card (a canonical comment). The gate runs on
the dispatcher, before ``claim_task``, so a destructive-live card that lacks a
recorded human GO stays ``ready`` (never claimed, never spawned) — the action
cannot execute without prior human consent.

Classification:
  DESTRUCTIVE_LIVE - the card signals a destructive/irreversible action on a
                     live resource => requires a registered human GO.
  SAFE            - not destructive-live => normal dispatch (no GO needed).

The gate is read-only and deterministic: it only reads the card (title/body)
and its comments from the board DB. It performs no LLM calls, no network, and
never mutates tasks/task_runs rows (the dispatcher's only write is a
diagnostic event, guarded by ``not dry_run``).

UNKNOWN/ambiguity => DESTRUCTIVE_LIVE (require GO, never guess). This mirrors
the conflict-gate fail-safe bias: a conservative gate can only ask a human to
confirm, never cause unintended destruction.

The GO comment is canonical and shared by all destructively-gated boards:

    @go destructive <task_id>      (one line, anywhere in the comment body)

The author of the GO comment must differ from the card's assignee (a worker
cannot self-authorize its own destructive action).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Pattern

# Sentinel classification strings.
DESTRUCTIVE_LIVE = "DESTRUCTIVE_LIVE"
SAFE = "SAFE"

# ──────────────────────────── classifier vocabulary ────────────────────────────
# Destructive / irreversible action verbs. Matched as whole (word-boundary)
# tokens so that e.g. "remove unused import" in normal dev cards is NOT hit,
# but "drop table", "delete bucket", "teardown tenant" ARE.
DESTRUCTIVE_VERBS = (
    "delete", "deleting", "deleted",
    "drop", "dropping", "dropped",
    "teardown", "tear-down", "tearing down",
    "remove", "removing", "removed",
    "purge", "purging", "purged",
    "wipe", "wiping", "wiped",
    "cleanup", "clean-up", "cleaning up",
    "decommission", "decommissioning", "decommissioned",
    "destroy", "destroying", "destroyed",
    "reset", "resetting", "reset the",
)

# Live / deployed resource markers. The gate requires a destructive verb AND a
# live-resource marker (or an explicit ``destructive_action`` directive) before
# it treats a card as destructive-live. This keeps normal dev cards safe while
# still catching bucket/tenant/db/prod cleanup.
LIVE_RESOURCE_MARKERS = (
    # object storage / infra
    "bucket", "r2", "s3", "object storage",
    # tenants / multi-tenancy
    "tenant", "tenants",
    # environment
    "live", " prod", "(prod", " production", " production)",
    # databases / schemas
    "database", "schema", " table", " index", " collection",
    "postgres", "supabase", "migration", "migrations",
    # deployment / service / infra
    "deployment", "service", "infra", "infrastructure",
    "cloudflare", "dns record", "route53",
)

# Explicit directive that overrides heuristics (any hit => DESTRUCTIVE_LIVE).
EXPLICIT_DIRECTIVES = (
    "destructive_action",
    "destructive action",
    "live_action: destructive",
    "requires_human_go",
    "human go required",
    "delete live",
    "remove live",
    "teardown live",
)

# Canonical human-GO comment marker. Readable, greppable, version-stable.
# Accepts:  "@go destructive <task_id>"   /   "GO_DESTRUCTIVE <task_id>"
# Group 1 captures the task_id; the caller must validate it against the
# card being evaluated to enforce per-card preauthorization (fail-closed).
GO_LINE_RE = re.compile(r"^\s*(?:@go\s+destructive|go_destructive|go\s+destructive)\s+(\S+)", re.IGNORECASE | re.MULTILINE)


@dataclass
class GateInput:
    """Everything the gate needs. The dispatcher supplies these from the card
    and cheap DB reads; the gate itself stays pure and side-effect free."""

    task_id: str
    title: str
    body: str
    human_go_comments: List[str] = field(default_factory=list)  # GO comment bodies, if any
    strict: bool = False
    allowlist: List[tuple[Pattern[str], str]] = field(default_factory=list)


@dataclass
class Verdict:
    cls: str
    reasons: List[str] = field(default_factory=list)


def _token_regex(verbs) -> "re.Pattern[str]":
    return re.compile(r"\b(" + "|".join(re.escape(v) for v in verbs) + r")\b", re.IGNORECASE)


_VERB_RE = _token_regex(DESTRUCTIVE_VERBS)


def _has_explicit_directive(title: str, body: str) -> bool:
    low = (title + "\n" + body).lower()
    return any(d in low for d in EXPLICIT_DIRECTIVES)


def _has_live_marker(title: str, body: str) -> bool:
    low = (title + "\n" + body).lower()
    return any(m in low for m in LIVE_RESOURCE_MARKERS)


def compile_allowlist(entries: Any) -> List[tuple[Pattern[str], str]]:
    """Compile explicit benign-operation patterns from config.

    Invalid entries are ignored rather than widened into a match: a malformed
    allowlist must never silently permit productive work.
    """
    if not isinstance(entries, list):
        return []
    compiled: List[tuple[Pattern[str], str]] = []
    for entry in entries:
        if (
            isinstance(entry, tuple)
            and len(entry) == 2
            and hasattr(entry[0], "search")
            and isinstance(entry[1], str)
        ):
            compiled.append(entry)
            continue
        if isinstance(entry, str):
            pattern, reason = entry, "explicit allowlist"
        elif isinstance(entry, dict):
            pattern = entry.get("pattern")
            reason = entry.get("reason") or "explicit allowlist"
        else:
            continue
        if not isinstance(pattern, str) or not pattern.strip() or not isinstance(reason, str):
            continue
        try:
            compiled.append((re.compile(pattern, re.IGNORECASE), reason))
        except re.error:
            continue
    return compiled


def _classify(title: str, body: str, *, strict: bool = False) -> Verdict:
    """Classify a card as destructive-live or safe. Pure + deterministic."""
    reasons: List[str] = []
    text = title + "\n" + body

    if _has_explicit_directive(title, body):
        reasons.append("explicit destructive_action directive")
        return Verdict(DESTRUCTIVE_LIVE, reasons)

    if _VERB_RE.search(text):
        if _has_live_marker(title, body):
            reasons.append("destructive verb + live-resource marker")
            return Verdict(DESTRUCTIVE_LIVE, reasons)
        if strict:
            reasons.append("destructive verb on strict productive board")
            return Verdict(DESTRUCTIVE_LIVE, reasons)
        reasons.append("destructive verb present but no live-resource marker")

    # No destructive verb, or a verb without a live target => not gated.
    if not reasons:
        return Verdict(SAFE, ["no destructive-live signal"])
    return Verdict(SAFE, reasons + ["not destructive-live"])


def _is_human_go(body: str, task_id: str) -> bool:
    """True when a comment body contains the canonical GO line for ``task_id``.

    Extracts the task_id from the GO line and validates it against the
    expected card id. A GO intended for a different card (e.g.
    ``@go destructive t_OTHER``) does NOT authorize this card.
    """
    m = GO_LINE_RE.search(body)
    if not m:
        return False
    return m.group(1) == task_id


def evaluate(inp: GateInput) -> Verdict:
    """Full gate verdict.

    DESTRUCTIVE_LIVE  -> the card performs a destructive/irreversible action
                         on a live resource; it still requires a human GO.
    SAFE              -> normal dispatch.

    This is the pure classifier. Whether a GO is present is surfaced via the
    reasons list ("human GO recorded" when present).
    """
    text = inp.title + "\n" + inp.body
    for pattern, reason in inp.allowlist:
        if pattern.search(text):
            return Verdict(SAFE, [f"allowlist: {reason}"])
    v = _classify(inp.title, inp.body, strict=inp.strict)
    if v.cls == SAFE:
        return v
    if any(_is_human_go(c, inp.task_id) for c in inp.human_go_comments):
        return Verdict(DESTRUCTIVE_LIVE, ["destructive-live but human GO recorded"])
    return Verdict(DESTRUCTIVE_LIVE, v.reasons + ["NO human GO recorded -> block"])


# ─────────────────────────── dispatcher-side helpers ───────────────────────────
# Read-only: reads the card + comments; never writes.

def _human_go_comment_bodies(conn, task_id, *, assignee: Optional[str] = None) -> List[str]:
    """Return comment bodies that can serve as a human GO for ``task_id``.

    A GO comment must:
      - contain the canonical GO line (``@go destructive ...``), and
      - NOT be authored by the card's assignee (a worker cannot self-authorize).
    """
    try:
        rows = conn.execute(
            "SELECT author, body FROM task_comments WHERE task_id=? ORDER BY id ASC",
            (task_id,),
        ).fetchall()
    except Exception:
        return []
    out: List[str] = []
    for r in rows:
        author = (r["author"] or "").strip()
        body = (r["body"] or "").strip()
        if not _is_human_go(body, task_id):
            continue
        if assignee and author and author.lower() == str(assignee).lower():
            continue  # self-authored GO by the executor is not a human GO
        out.append(body)
    return out


def destructive_gate_requires_go(conn, task_id, *, board=None, strict: bool = False, allowlist=None):
    """Admission decision for a ready card. Mirrors the conflict-gate contract:

      None  -> allow (card is not destructive-live, OR a human GO is recorded).
      tuple -> block (DESTRUCTIVE_LIVE with no human GO recorded) — the
               descriptor leaves the card ``ready`` (no claim/spawn) until a
               human records the canonical GO comment.

    Deterministic + read-only; called only when ``kanban.destructive_gate`` is
    enabled. This is the PRE-ACTION gate: it blocks execution of the
    destructive action BEFORE it starts, enforcing:

        pre-verification -> human GO -> destructive action -> postcondition.
    """
    try:
        row = conn.execute(
            "SELECT title, body, assignee FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
    except Exception as exc:
        return ("DESTRUCTIVE_LIVE", f"destructive_gate card lookup failed: {exc}")
    if row is None:
        return ("DESTRUCTIVE_LIVE", "destructive_gate card not found")
    title = row["title"] or ""
    body = row["body"] or ""
    assignee = row["assignee"] or None

    go_comments = _human_go_comment_bodies(conn, task_id, assignee=assignee)
    try:
        v = evaluate(GateInput(
            task_id=task_id,
            title=title,
            body=body,
            human_go_comments=go_comments,
            strict=strict,
            allowlist=compile_allowlist(allowlist),
        ))
    except Exception as exc:
        # Fail-safe: any unexpected error degrades to block (require GO), never
        # lets a possibly-destructive action through without a human GO.
        return ("DESTRUCTIVE_LIVE", f"destructive_gate evaluate error: {exc}")
    if v.cls == SAFE:
        return None  # not destructive-live -> allow
    if go_comments:
        return None  # destructive-live but a human GO is recorded -> allow
    return ("DESTRUCTIVE_LIVE", "; ".join(v.reasons))