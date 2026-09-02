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

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Pattern

# Sentinel classification strings.
DESTRUCTIVE_LIVE = "DESTRUCTIVE_LIVE"
SAFE = "SAFE"

# Canonical GO line format (v1.1):
#
#     @go destructive <task_id> <action_id>
#
# `<action_id>` binds the human GO to a concrete destructive action on the
# card (digest over tenant|verb|resource|executor — see
# ``compute_destructive_action_id``). The legacy form without action_id is
# still accepted for backward compatibility on cards that do NOT declare a
# ``destructive_action_id`` line (see section "Archivo" of the v1.1 policy).
GO_GUIDANCE = (
    "Destructive gate (pre-action): {reason}\n"
    "Fix: (1) run `hermes kanban preverify-destructive <task_id> <action_id>` "
    "for a verified pre-check, then (2) have a human (NOT the executor, NOT "
    "the dashboard) record `@go destructive <task_id> <action_id>` on the card, "
    "then (3) the dispatcher will admit the claim. A GO recorded after the "
    "destructive action does not satisfy the pre-action gate."
)

POSTCONDITION_GUIDANCE = (
    "Destructive gate (pre-action): {reason}\n"
    "Fix: record the postcondition verification with "
    "`hermes kanban approve-destructive --postcondition <task_id> <action_id>` "
    "(evidence: e.g. the resource read returned the expected, non-destructive state)."
)

# Authors that can never serve as the human author of a destructive GO /
# pre-verification: executor-adjacent identities and system surfaces. The
# canonical ``destructive_authorized`` event is only recorded when the
# comment author is NOT in this list and NOT the card's assignee.
_AUTHOR_DENYLIST_DEFAULT: tuple[str, ...] = (
    "dashboard",
    "worker",
    "hermes-system",
    "system",
    "specifier",
    "decomposer",
    "auto-decomposer",
)

# Event kinds of the deterministic human-GO mechanism (task_events rows).
DESTRUCTIVE_EVENT_KINDS: frozenset[str] = frozenset({
    "destructive_preverified",
    "destructive_authorized",
    "destructive_postcondition_posted",
})

# Default staleness window for a recorded `destructive_authorized` event.
DEFAULT_AUTHORIZED_TTL_SECONDS = 604800  # 7 days

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
    # v1.1: destructive verbs over live resources that the gate must not miss.
    "truncate", "truncating", "truncated",
    "revoke", "revoking", "revoked",
    "replace", "replacing", "replaced",
    "rename", "renaming", "renamed",
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
    # credentials / identity (only meaningful as a destructive target when a
    # destructive verb is present — the classifier requires both)
    "credential", "credentials", "secret", "secrets", "api key", "api keys",
    " user", " users",
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
# Accepts (v1.1 superset of the legacy regex, group 2 is the OPTIONAL
# action_id):
#   "@go destructive <task_id> [<action_id>]"
#   "GO_DESTRUCTIVE <task_id> [<action_id>]"
#   "go destructive <task_id> [<action_id>]"
# Group 1 captures the task_id; the caller must validate it against the
# card being evaluated to enforce per-card preauthorization (fail-closed).
# The legacy form WITHOUT action_id still matches (backward compatibility);
# the v1.1 form binds the GO to a concrete action.
GO_LINE_RE = re.compile(
    r"^\s*(?:@go\s+destructive|go_destructive|go\s+destructive)\s+(\S+)(?:\s+(\S+))?",
    re.IGNORECASE | re.MULTILINE,
)


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

def _human_go_comment_bodies(conn, task_id, *, assignee: Optional[str] = None, denylist: Optional[Any] = None) -> List[str]:
    """Return comment bodies that can serve as a human GO for ``task_id``.

    A GO comment must:
      - contain the canonical GO line (``@go destructive ...``),
      - NOT be authored by the card's assignee (a worker cannot
        self-authorize), and
      - NOT be authored by a denylisted system/dashboard/executor identity
        (``denylist``; defaults to :data:`_AUTHOR_DENYLIST_DEFAULT`).
    """
    denied = {
        str(a).strip().lower()
        for a in (denylist if denylist is not None else _AUTHOR_DENYLIST_DEFAULT)
        if str(a).strip()
    }
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
        if author.lower() in denied:
            continue  # system/dashboard/worker-authored GO is not a human GO
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

# ═══════════════════════════════════════════════════════════════════════════════
# v1.1 deterministic human-GO mechanism
#
# 1. The card declares its live action with canonical body lines (read by
#    ``_parse_destructive_declaration``):
#
#        destructive_action_id: sha256:<hex16>
#        destructive_verb: delete
#        destructive_resource: r2://erp-client-a-docs
#        destructive_tenant: CLIENT_A
#
# 2. ``compute_destructive_action_id`` derives the digest when the card does
#    NOT declare one: sha256(tenant|verb|resource|executor) — the `executor`
#    component ties the GO to the actual assignee (ONE CARD -> ONE EXECUTOR).
# 3. The canonical human GO is `@go destructive <task_id> <action_id>`
#    (comment + `destructive_authorized` event). The claim gate (claim_task)
#    verifies, in order: pre-verification event, canonical authorized event
#    matching the current action_id, staleness (TTL / edited-after-GO).
# 4. complete_task additionally demands a `destructive_postcondition_posted`
#    event recorded after the GO before the card may close.
# ═══════════════════════════════════════════════════════════════════════════════


def author_is_denied(author: Optional[str], *, assignee: Optional[str] = None,
                     denylist: Optional[Any] = None) -> bool:
    """True when ``author`` cannot record a human GO / pre-verification.

    Denied: empty authors, any identity in the denylist (default
    :data:`_AUTHOR_DENYLIST_DEFAULT`), and the card's assignee (executor).
    Case-insensitive on every comparison.
    """
    a = (author or "").strip().lower()
    if not a:
        return True
    denied = {
        str(x).strip().lower()
        for x in (denylist if denylist is not None else _AUTHOR_DENYLIST_DEFAULT)
        if str(x).strip()
    }
    if a in denied:
        return True
    if assignee and a == str(assignee).strip().lower():
        return True
    return False


def _normalize_token(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _first_verb_token(text: str) -> Optional[str]:
    """First (leftmost) destructive verb token in ``text`` (deterministic)."""
    low = text.lower()
    best: Optional[str] = None
    best_pos = 10**9
    for v in DESTRUCTIVE_VERBS:
        for m in re.finditer(r"\b" + re.escape(v) + r"\b", low):
            if m.start() < best_pos:
                best, best_pos = v, m.start()
    return best


def _first_live_marker(text: str) -> Optional[str]:
    """First (leftmost) live-resource marker in ``text`` (deterministic)."""
    low = text.lower()
    best: Optional[str] = None
    best_pos = 10**9
    for marker in LIVE_RESOURCE_MARKERS:
        pos = low.find(marker)
        if pos != -1 and pos < best_pos:
            best, best_pos = marker.strip(), pos
    return best


# Card body lines that declare the live destructive action (v1.1). Any line,
# anywhere in title/body; regex per-line so whitespace/casing are forgiving.
_DECL_ID_RE = re.compile(r"^\s*destructive_action_id\s*:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)
_DECL_VERB_RE = re.compile(r"^\s*destructive_verb\s*:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)
_DECL_RESOURCE_RE = re.compile(r"^\s*destructive_resource\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_DECL_TENANT_RE = re.compile(r"^\s*destructive_tenant\s*:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)


def _parse_destructive_declaration(title: str, body: str) -> dict:
    """Extract the card's declared live-action fields (v1.1 contract).

    Returns ``{"action_id", "verb", "resource", "tenant"}`` — every key
    present, with the raw (untrimmed-whitespace, original-case) value or ""
    when the card does not declare it.
    """
    text = (title or "") + "\n" + (body or "")
    out = {
        "action_id": "",
        "verb": "",
        "resource": "",
        "tenant": "",
    }
    m = _DECL_ID_RE.search(text)
    if m:
        out["action_id"] = m.group(1).strip()
    m = _DECL_VERB_RE.search(text)
    if m:
        out["verb"] = m.group(1).strip()
    m = _DECL_RESOURCE_RE.search(text)
    if m:
        out["resource"] = m.group(1).strip()
    m = _DECL_TENANT_RE.search(text)
    if m:
        out["tenant"] = m.group(1).strip()
    return out


def compute_destructive_action_id(
    title: str, body: str, *, tenant: Optional[str] = None,
    assignee: Optional[str] = None,
) -> str:
    """Deterministic digest that binds a card to ONE concrete live action.

    * If the card declares ``destructive_action_id:``, that declared value is
      authoritative (it is what the operator bound the GO to).
    * Otherwise the digest is ``sha256:<hex16>`` over
      ``tenant|verb|resource|executor`` — the classifier vocabulary supplies
      verb/resource when the card does not declare them, and ``executor`` is
      the card's assignee so the GO is not portable across workers.

    Same input -> same digest (pure function, no DB, no clock).
    """
    decl = _parse_destructive_declaration(title, body)
    declared = decl.get("action_id") or ""
    if declared:
        return declared

    tenant_n = _normalize_token(decl.get("tenant") or tenant)
    verb = _normalize_token(decl.get("verb")) or _normalize_token(
        _first_verb_token((title or "") + "\n" + (body or ""))
    ) or "unknown"
    resource = _normalize_token(decl.get("resource")) or _normalize_token(
        _first_live_marker((title or "") + "\n" + (body or ""))
    ) or "unknown"
    executor = _normalize_token(assignee)
    raw = f"{tenant_n}|{verb}|{resource}|{executor}"
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def action_id_from_go_line(body: str) -> Optional[str]:
    """Return the action_id of a v1.1 GO line, or None (legacy form)."""
    m = GO_LINE_RE.search(body or "")
    if not m:
        return None
    return m.group(2)


def latest_destructive_event(conn, task_id: str, kind: str) -> Optional[dict]:
    """Latest (max-id) ``task_events`` row of ``kind`` for ``task_id``.

    Returns ``{"id", "created_at", "payload"}`` or None. Read-only; the
    event payload's ``action_id`` field is the binding the gates enforce.
    """
    if kind not in DESTRUCTIVE_EVENT_KINDS:
        return None
    try:
        row = conn.execute(
            "SELECT id, created_at, payload FROM task_events "
            "WHERE task_id = ? AND kind = ? ORDER BY id DESC LIMIT 1",
            (task_id, kind),
        ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    payload: dict = {}
    if row["payload"]:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            payload = {}
    return {
        "id": int(row["id"]),
        "created_at": int(row["created_at"] or 0),
        "payload": payload,
    }


def _is_stale_authorized(conn, task_id: str, auth_event: dict, *, ttl_seconds: int) -> tuple[bool, str]:
    """Staleness of a canonical ``destructive_authorized`` event.

    Stale when either:
      * the event is older than ``ttl_seconds`` (default 7 days); or
      * the card was edited (title/body) AFTER the GO was recorded
        (any ``edited`` event with payload fields including title/body and
        ``id > auth id``) — an edited card may describe a DIFFERENT action,
        so the old GO must not cover it.
    """
    created = auth_event.get("created_at") or 0
    if ttl_seconds and ttl_seconds > 0 and (int(time.time()) - created) > ttl_seconds:
        return True, f"authorized event older than ttl {ttl_seconds}s"
    try:
        rows = conn.execute(
            "SELECT id, payload FROM task_events "
            "WHERE task_id = ? AND kind = 'edited' AND id > ? ORDER BY id ASC",
            (task_id, int(auth_event.get("id") or 0)),
        ).fetchall()
    except Exception:
        rows = []
    for r in rows:
        fields: list = []
        if r["payload"]:
            try:
                pl = json.loads(r["payload"])
            except (TypeError, ValueError):
                pl = {}
            fields = pl.get("fields") or pl.get("changed_fields") or []
        if any(f in ("title", "body") for f in fields):
            return True, f"card edited (event {r['id']}) after GO"
    return False, ""


def _block(cls: str, reason: str, *, guidance: Optional[str] = None) -> dict:
    return {"cls": cls, "reason": reason, "guidance": guidance or GO_GUIDANCE.format(reason=reason)}


def _resolve_scope(conn, task_id: str, *, strict: bool, allowlist: Optional[Any],
                   tenant_scope: Any) -> Optional[dict]:
    """Shared card/scope resolution for the two verdicts.

    Returns ``None`` when the card is NOT destructive-live in scope (allow);
    otherwise a dict with ``title/body/tenant/assignee/text/verdict/action_id/
    in_tenant_scope/declared``.
    """
    try:
        row = conn.execute(
            "SELECT title, body, tenant, assignee FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    except Exception as exc:
        return {"error": _block("DESTRUCTIVE_LIVE", f"destructive_gate card lookup failed: {exc}")}
    if row is None:
        return {"error": _block("DESTRUCTIVE_LIVE", "destructive_gate card not found")}
    title = row["title"] or ""
    body = row["body"] or ""
    tenant = (row["tenant"] or "").strip() or None
    assignee = row["assignee"] or None
    text = title + "\n" + body
    for pattern, reason in compile_allowlist(allowlist):
        if pattern.search(text):
            return None  # explicitly benign operation -> SAFE
    verdict = _classify(title, body, strict=strict)
    in_tenant_scope = bool(tenant and tenant in set(tenant_scope or ()))
    if verdict.cls == SAFE and not in_tenant_scope:
        return None  # not destructive-live in scope -> allow
    decl = _parse_destructive_declaration(title, body)
    action_id = compute_destructive_action_id(title, body, tenant=tenant, assignee=assignee)
    return {
        "task_id": task_id,
        "title": title,
        "body": body,
        "tenant": tenant,
        "assignee": assignee,
        "text": text,
        "verdict": verdict,
        "action_id": action_id,
        "declared": bool((decl.get("action_id") or "").strip()),
        "in_tenant_scope": in_tenant_scope,
    }


def claim_gate_verdict(
    conn, task_id: str, *, board: Optional[str] = None, strict: bool = False,
    allowlist: Optional[Any] = None, tenant_scope: Any = (),
    require_preverify: bool = False, authorized_ttl_seconds: Optional[int] = None,
    denylist_authors: Optional[Any] = None,
) -> Optional[dict]:
    """Admission decision for CLAIMING a destructive-live card.

    ``None`` -> allow the claim. ``{"cls","reason","guidance"}`` -> block.

    Fail-closed ordering (each rule independent):
      1. If the card is destructive-live in scope and the mechanism is active:
      2. ``require_preverify=True``  -> a ``destructive_preverified`` event for
         the CURRENT action_id must exist BEFORE any ``destructive_authorized``
         event (``id_authorized > id_preverified``), else the pre-action
         ordering is violated.
      3. A canonical ``destructive_authorized`` event for the current
         action_id is the ONLY source of truth; a legacy GO comment without
         the event does not count when an action binding exists or the
         mechanism is strict. With ``require_preverify=False`` and no declared
         action_id, a legacy GO comment (author != executor, not denied) is
         still honored for backward compatibility.
      4. A stale GO (TTL or edited-after-GO) blocks; a GO for a DIFFERENT
         action_id (card edited post-GO) blocks with ``mismatched GO``.
    """
    scope = _resolve_scope(conn, task_id, strict=strict, allowlist=allowlist, tenant_scope=tenant_scope)
    if scope is None:
        return None  # allow (not destructive-live in scope, or allowlisted)
    if "error" in scope:
        return scope["error"]

    action_id = scope["action_id"]
    auth = latest_destructive_event(conn, task_id, "destructive_authorized")
    denylist = list(denylist_authors) if denylist_authors else list(_AUTHOR_DENYLIST_DEFAULT)

    if require_preverify:
        pre = latest_destructive_event(conn, task_id, "destructive_preverified")
        if pre is None or pre["payload"].get("action_id") != action_id:
            return _block(
                "DESTRUCTIVE_LIVE",
                "GO ordering requires pre-verification first (no "
                f"destructive_preverified event for action_id {action_id})",
            )
        if auth is None or auth["id"] <= pre["id"]:
            if auth is None:
                return _block("DESTRUCTIVE_LIVE", "NO recorded human GO")
            return _block(
                "DESTRUCTIVE_LIVE",
                "GO ordering requires pre-verification first "
                f"(destructive_authorized event {auth['id']} recorded before "
                f"destructive_preverified event {pre['id']})",
            )
        return _claim_authorized_gate(conn, scope, auth, ttl=authorized_ttl_seconds)

    # Legacy-compatible path (require_preverify=False).
    if auth is not None:
        # A canonical event (the operator used approve-destructive) is the
        # source of truth: it must match the current action and not be stale.
        return _claim_authorized_gate(conn, scope, auth, ttl=authorized_ttl_seconds)
    if scope["declared"]:
        # Card declares an action binding; a legacy GO without action_id
        # cannot authorize it — the operator must use the v1.1 flow.
        return _block(
            "DESTRUCTIVE_LIVE",
            f"stale or mismatched GO for action (card declares "
            f"destructive_action_id {scope['action_id']!r} but no canonical "
            "destructive_authorized event exists)",
        )
    go_comments = _human_go_comment_bodies(conn, task_id, assignee=scope["assignee"], denylist=denylist)
    valid, mismatched = _partition_go_comments(go_comments, scope["action_id"])
    if valid:
        return None  # legacy human GO comment (author != executor) honored
    if mismatched:
        return _block(
            "DESTRUCTIVE_LIVE",
            f"stale or mismatched GO for action (recorded GO binds "
            f"action_id {mismatched!r}, current action is "
            f"{scope['action_id']!r})",
        )
    return _block("DESTRUCTIVE_LIVE", "NO recorded human GO")


def _partition_go_comments(go_comments: List[str], action_id: str) -> tuple[list, Optional[str]]:
    """Split author-valid GO comment bodies into (valid, first_mismatch).

    * Legacy form (no action_id in the GO line) is valid when no action
      binding exists (the caller only reaches here for undeclared cards).
    * v1.1 form (action_id present) is valid only when it equals the
      card's current action_id; a wrong action_id is surfaced as a
      mismatch (the GO does not authorize the requested action).
    """
    valid: list = []
    mismatch: Optional[str] = None
    for body in go_comments:
        bound = action_id_from_go_line(body)
        if bound is None:
            valid.append(body)
        elif bound.lower() == str(action_id).lower():
            valid.append(body)
        elif mismatch is None:
            mismatch = bound
    return valid, mismatch


def _claim_authorized_gate(conn, scope: dict, auth: dict, *, ttl: Optional[int]) -> Optional[dict]:
    """Validate an existing canonical authorized event against the card.

    Returns ``None`` when the GO is valid for the current action, or a
    ``{"cls","reason","guidance"}`` block dict when it is mismatched or stale.
    """
    action_id = scope["action_id"]
    if auth["payload"].get("action_id") != action_id:
        return _block(
            "DESTRUCTIVE_LIVE",
            f"stale or mismatched GO for action (authorized action_id "
            f"{auth['payload'].get('action_id')!r} != current "
            f"{action_id!r})",
        )
    ttl_seconds = ttl if ttl is not None else DEFAULT_AUTHORIZED_TTL_SECONDS
    stale, why = _is_stale_authorized(conn, scope["task_id"], auth, ttl_seconds=ttl_seconds)
    if stale:
        return _block("DESTRUCTIVE_LIVE", f"stale GO for action ({why})")
    return None


def completion_gate_verdict(
    conn, task_id: str, *, board: Optional[str] = None, strict: bool = False,
    allowlist: Optional[Any] = None, tenant_scope: Any = (),
    require_preverify: bool = False, authorized_ttl_seconds: Optional[int] = None,
    denylist_authors: Optional[Any] = None,
) -> Optional[dict]:
    """Admission decision for COMPLETING a destructive-live card.

    ``None`` -> allowed (no tracing metadata needed). Otherwise a dict:

      * ``{"allowed": True, "meta": {...}}`` -> allowed; ``meta`` carries
        ``approved_action_id`` / ``authorized_event_id`` /
        ``postverified_event_id`` for the closing run's metadata.
      * ``{"allowed": False, "cls", "reason", "guidance"}`` -> block.

    Requirements (fail-closed): a canonical human GO for the current
    action_id (stale/mismatched GO still blocks), then a
    ``destructive_postcondition_posted`` event recorded AFTER the GO.
    """
    scope = _resolve_scope(conn, task_id, strict=strict, allowlist=allowlist, tenant_scope=tenant_scope)
    if scope is None:
        return None
    if "error" in scope:
        return {"allowed": False, **scope["error"]}

    action_id = scope["action_id"]
    auth = latest_destructive_event(conn, task_id, "destructive_authorized")
    denylist = list(denylist_authors) if denylist_authors else list(_AUTHOR_DENYLIST_DEFAULT)

    if auth is None:
        if not require_preverify and not scope["declared"]:
            # Legacy-compat completion: a legacy GO comment (author !=
            # executor, not denied) suffices when no action binding exists;
            # a v1.1 GO comment bound to a DIFFERENT action does not.
            go_comments = _human_go_comment_bodies(conn, task_id, assignee=scope["assignee"], denylist=denylist)
            valid, mismatched = _partition_go_comments(go_comments, scope["action_id"])
            if valid:
                return {"allowed": True, "meta": None}
            if mismatched:
                return {
                    "allowed": False,
                    **_block(
                        "DESTRUCTIVE_LIVE",
                        f"stale or mismatched GO for action (recorded GO binds "
                        f"action_id {mismatched!r}, current action is "
                        f"{scope['action_id']!r})",
                    ),
                }
        return {"allowed": False, **_block("DESTRUCTIVE_LIVE", "no recorded human GO")}

    blocked = _claim_authorized_gate(conn, scope, auth, ttl=authorized_ttl_seconds)
    if blocked is not None:
        return {"allowed": False, **blocked}

    post = latest_destructive_event(conn, task_id, "destructive_postcondition_posted")
    if (
        post is None
        or post["payload"].get("action_id") != action_id
        or post["id"] <= auth["id"]
    ):
        return {
            "allowed": False,
            **_block(
                "DESTRUCTIVE_LIVE",
                "postcondition verification missing",
                guidance=POSTCONDITION_GUIDANCE.format(reason="postcondition verification missing"),
            ),
        }
    return {
        "allowed": True,
        "meta": {
            "approved_action_id": action_id,
            "authorized_event_id": auth["id"],
            "postverified_event_id": post["id"],
        },
    }
