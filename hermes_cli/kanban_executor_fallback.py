"""Kanban executor fallback: run one card attempt through ``claude -p``.

The dispatcher spawns this module (``python -m hermes_cli.kanban_executor_fallback
<task_id> <workspace>``) when a card's provider-quota failure queued a
``claude-p`` fallback attempt. It is a *thin executor bridge*: it feeds the SAME
card context (title + body, the bounded task contract) to the real Claude Code
CLI, and terminates the card only through the canonical board protocol
(``complete_task`` / ``block_task``).

Exit-code contract with the dispatcher's reap classifier:

* ``0`` — the attempt reached a durable terminal transition
  (``done`` via complete_task, or ``blocked`` via block_task). The dispatcher
  treats a clean exit with a non-running task as a normal completion.
* ``KANBAN_RATE_LIMIT_EXIT_CODE`` (75) — the attempt failed for a QUALIFYING
  provider/credential/transport reason (claude rate-limit, billing wall, missing
  binary, missing OAuth). The reap classifier records ``rate_limited`` and the
  dispatcher releases the card back to ``ready`` so the next tick queues the
  lower-priority fallback. The card is NOT mutated here.

Non-qualifying outcomes (``error_max_turns``, unparseable output, missing
completion marker at the end of a successful run) block the card — task-side
contract failures never reroute, and UNKNOWN never blind-reroutes.

``complete_task``/``block_task`` carry the attempt's executor identity so the
run chain exposes REQUESTED vs ACTUAL executor per attempt.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any, Optional

from hermes_cli import kanban_db as kb

_COMPLETION_MARKER = "EXECUTOR_FALLBACK_COMPLETE:"
_MAX_TURNS = "15"
_ALLOWED_TOOLS = (
    "Read,Edit,Write,Grep,"
    "Bash(git *),Bash(uv *),Bash(pytest *),Bash(python3 *),Bash(rg *),Bash(ls *)"
)
_MAX_PROMPT_BODY_CHARS = 8000
_TIMEOUT_SECONDS = 1800

# Claude Code JSON error codes that mean the *executor backend* is unusable
# (qualifying for reroute), as opposed to the task being too complex / broken.
_QUALIFYING_ERROR_CODES = {
    "error_budget",       # billing/cost limit reached
    "error_rate_limit",   # account rate-limited / overloaded
    "error_quota",        # quota exhausted
}
_QUALIFYING_TEXT_PATTERNS = (
    re.compile(r"\b(rate[\s_-]?limit|quota|429|billing)\b", re.IGNORECASE),
    re.compile(r"not\s+(authenticated|logged\s*in)", re.IGNORECASE),
)


def _marker_summary(text: str) -> Optional[str]:
    """Return the one-line summary that follows the completion marker."""
    match = re.search(
        re.escape(_COMPLETION_MARKER) + r"\s*(.+)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    summary = match.group(1).strip().splitlines()[0].strip()
    return summary[:500] or None


def _is_qualifying_failure(error_payload: Optional[dict], raw_stderr: str, returncode: int) -> bool:
    """True only for provider/credential/transport unavailability."""
    if isinstance(error_payload, dict):
        code = str(error_payload.get("code") or "")
        if code in _QUALIFYING_ERROR_CODES:
            return True
        message = str(error_payload.get("message") or error_payload.get("msg") or "")
        if any(p.search(message) for p in _QUALIFYING_TEXT_PATTERNS):
            return True
    combined = (raw_stderr or "") + (json.dumps(error_payload) if isinstance(error_payload, dict) else "")
    if any(p.search(combined) for p in _QUALIFYING_TEXT_PATTERNS):
        return True
    # Non-zero exits without a recognised error payload are unknown → never
    # blind-reroute. returncode alone is not a qualifying signal.
    return False


def _actual_model_from(result_payload: dict) -> Optional[str]:
    usage = result_payload.get("modelUsage")
    if isinstance(usage, dict):
        for key in usage:
            if key:
                return str(key)
    return None


def _run_claude(prompt: str, workspace: str) -> tuple[int, dict, str]:
    """Execute ``claude -p`` and return ``(returncode, parsed_json_or_empty, stderr)``."""
    cmd = [
        "claude",
        "-p",
        prompt,
        "--max-turns",
        _MAX_TURNS,
        "--allowedTools",
        _ALLOWED_TOOLS,
        "--output-format",
        "json",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=workspace if os.path.isdir(workspace) else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        raise
    except subprocess.TimeoutExpired:
        return (124, {}, "claude -p timed out")
    payload: dict = {}
    stdout = (proc.stdout or "").strip()
    if stdout:
        try:
            parsed = json.loads(stdout)
            if isinstance(parsed, dict):
                payload = parsed
        except (ValueError, TypeError):
            # Non-JSON stdout can still happen for warnings before the result;
            # keep the raw text so the marker can be found on success.
            payload = {"type": "result", "subtype": "success", "result": stdout}
    return proc.returncode, payload, (proc.stderr or "")


def _build_prompt(task) -> str:
    body = (task.body or "").strip()[:_MAX_PROMPT_BODY_CHARS]
    return (
        "You are completing a task on a Hermes kanban board as an executor "
        "fallback. Work in the current working directory only.\n\n"
        f"TASK ID: {task.id}\n"
        f"TITLE: {task.title}\n"
        f"ASSIGNEE PROFILE: {task.assignee}\n"
        f"TASK BODY (contract):\n{body}\n\n"
        "RULES:\n"
        "1. Do the task described in the body. Implement, verify, and leave the "
        "deliverable in this directory.\n"
        "2. You have no kanban tools. Do NOT attempt to call kanban_complete or "
        "kanban_block — they do not exist here.\n"
        "3. When the work is genuinely finished, your FINAL reply must begin "
        "with exactly: " + _COMPLETION_MARKER + " followed by a one-line summary "
        "of what you did and where.\n"
        "4. If you cannot complete the task (missing inputs, contradictory "
        "contract), stop and say so plainly — do not invent a marker."
    )


def _terminal_block(conn, tid: str, reason: str, *, kind: Optional[str] = None,
                    meta: Optional[dict] = None,
                    task: Any = None) -> None:
    try:
        kb.block_task(conn, tid, reason=reason[:500], kind=kind)
    except Exception:
        pass
    try:
        _append_fallback_stopped(
            conn,
            tid,
            {**(_attempt_identity(conn, task) if task is not None else {}),
             **(meta or {})},
            reason=reason,
        )
    except Exception:
        pass


def _append_fallback_stopped(conn, tid: str, meta: dict, *, reason: str) -> None:
    with kb.write_txn(conn):
        kb._append_event(
            conn,
            tid,
            kb._EXECUTOR_FALLBACK_STOPPED_EVENT,
            {**meta, "reason": reason, "stopped_at": int(__import__("time").time())},
        )


def _attempt_identity(conn, task) -> dict:
    """Snapshot the run's claim-time attempt identity for the closing metadata."""
    if not task.current_run_id:
        return {}
    try:
        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (task.current_run_id,),
        ).fetchone()
        meta = json.loads(row["metadata"] or "{}") if row and row["metadata"] else {}
    except (ValueError, TypeError):
        return {}
    if not isinstance(meta, dict):
        return {}
    keep = (
        "attempt_number", "card_id", "profile", "requested_executor",
        "fallback_from", "fallback_reason", "failed_executor",
    )
    return {k: meta[k] for k in keep if k in meta}


def _completed_meta(conn, task, result_payload: dict) -> dict:
    identity = _attempt_identity(conn, task)
    return {
        **identity,
        "actual_executor": "claude-p",
        "actual_provider": "anthropic",
        "actual_model": _actual_model_from(result_payload) or "UNKNOWN",
        "outcome": "completed",
    }


def run_fallback_attempt(task_id: str, workspace: str) -> int:
    """Execute one claude-p attempt for ``task_id`` and return the exit code."""
    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        if task is None:
            return 0
        prompt = _build_prompt(task)
        try:
            returncode, payload, stderr = _run_claude(prompt, workspace)
        except FileNotFoundError:
            # claude binary absent = executor backend unavailable (qualifying).
            return kb.KANBAN_RATE_LIMIT_EXIT_CODE

        subtype = str(payload.get("subtype") or "")
        is_success = bool(payload) and subtype == "success" and returncode == 0

        if is_success:
            result_text = str(payload.get("result") or "")
            marker_summary = _marker_summary(result_text)
            if marker_summary is not None:
                ok = kb.complete_task(
                    conn,
                    task_id,
                    summary=f"[claude-p fallback] {marker_summary}",
                    result=result_text[:2000],
                    metadata=_completed_meta(conn, task, payload),
                )
                return 0 if ok else 1
            # Successful run that never delivered the marker → task-side
            # contract violation (narrated instead of executed): block.
            _terminal_block(
                conn,
                task_id,
                "claude -p succeeded but never emitted "
                + _COMPLETION_MARKER
                + " — task contract not delivered; no reroute.",
                kind="transient",
                meta={"actual_executor": "claude-p", "claude_subtype": subtype},
            )
            return 0

        error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else None
        if _is_qualifying_failure(error_payload, stderr, returncode):
            # Backend wall (rate-limit / billing / auth). Leave the card
            # untouched; exit 75 so the dispatcher queues the lower fallback.
            return kb.KANBAN_RATE_LIMIT_EXIT_CODE

        reason = (
            f"claude -p failed (returncode={returncode}, subtype={subtype or 'none'}) "
            f"{error_payload or stderr or ''}".strip()[:400]
        )
        _terminal_block(
            conn,
            task_id,
            reason or f"claude -p failed with returncode {returncode}",
            kind="transient",
            meta={"actual_executor": "claude-p", "claude_subtype": subtype},
        )
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main(argv: Optional[list] = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if len(args) < 2:
        print(
            "usage: python -m hermes_cli.kanban_executor_fallback <task_id> <workspace>",
            file=sys.stderr,
        )
        return 2
    return run_fallback_attempt(args[0], args[1])


if __name__ == "__main__":
    sys.exit(main())