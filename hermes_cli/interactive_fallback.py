"""Session-level provider fallback helpers (INTERACTIVE_SESSION_FALLBACK_V1).

Shared layer used by every interactive session surface (TUI, one-shot,
goal-mode, gateway/messaging, ACP) to seed the conversation loop's fallback
chain and to observe provider failovers.  This is the NATIVE in-process
design of IS_D1_DESIGN_VERDICT.md: no subprocess bridge, no kanban coupling,
no ``fallback_providers`` policy edits.  The chain itself is still walked by
the runtime's ``try_activate_fallback`` — these helpers only decide WHICH
chain to seed and how to REPORT a switch, never how the loop behaves.

Design contract (research/interactive-session-fallback-v1/IS_D1_DESIGN_VERDICT.md):
  * No secrets: labels/logs carry provider/model identifiers only.
  * ``fallback_providers`` config is never written here (runtime-only chain).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from hermes_cli.fallback_config import get_fallback_chain

logger = logging.getLogger("hermes_cli.interactive_fallback")

# Marker keyed on chain entries that came from the EXPLICIT interactive-session
# route (``interactive_session_fallback.providers`` or the env override).  The
# conversation-loop notice hook uses it to tell the session-explicit route from
# the legacy ``fallback_providers`` route, so existing UX stays untouched when
# the feature is not configured.  Entries are plain dicts consumed via
# ``entry.get(...)`` everywhere in the runtime, so the marker is inert
# (``agent/agent_init.py:1570`` filters by provider/model only).
_SESSION_EXPLICIT_MARK = "_session_explicit_route"


def _mark_session_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a fresh copy of a session-route entry carrying the explicit mark."""
    marked = dict(entry)
    marked[_SESSION_EXPLICIT_MARK] = True
    return marked


def build_session_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Seed the session fallback chain from config, in precedence order:

    1. ``HERMES_INTERACTIVE_FALLBACK_PROVIDERS`` (JSON list) — replaces any
       other source (mirrors the PR #8 env-override pattern).
    2. ``interactive_session_fallback.providers`` when the master switch
       ``interactive_session_fallback.enabled`` is on (env
       ``HERMES_INTERACTIVE_FALLBACK=0`` turns it off).
    3. Otherwise the legacy chain from ``fallback_providers`` /
       ``fallback_model`` (``hermes_cli.fallback_config.get_fallback_chain``).

    Always returns a list (possibly empty) so the loop keeps its current
    behavior when no chain is configured.  Entries from routes 1-2 carry the
    ``_session_explicit_route`` marker; route 3 entries are returned verbatim.
    Missing ``enabled`` defaults to True, matching ``DEFAULT_CONFIG``.
    """
    config = config or {}
    block = config.get("interactive_session_fallback")
    if not isinstance(block, dict):
        block = {}
    enabled = bool(block.get("enabled", True))
    if os.environ.get("HERMES_INTERACTIVE_FALLBACK") == "0":
        enabled = False

    env_json = os.environ.get("HERMES_INTERACTIVE_FALLBACK_PROVIDERS", "").strip()
    if env_json:
        try:
            env_providers = json.loads(env_json)
            if isinstance(env_providers, list):
                return [
                    _mark_session_entry(entry)
                    for entry in env_providers
                    if isinstance(entry, dict) and entry.get("provider") and entry.get("model")
                ]
        except (ValueError, TypeError):
            logger.debug("HERMES_INTERACTIVE_FALLBACK_PROVIDERS is not valid JSON — ignoring")

    if enabled:
        providers = block.get("providers")
        if isinstance(providers, list) and providers:
            return [
                _mark_session_entry(entry)
                for entry in providers
                if isinstance(entry, dict) and entry.get("provider") and entry.get("model")
            ]

    return get_fallback_chain(config)


def is_qualifying_provider_failure(classified: Any) -> bool:
    """Public qualifying-failure policy for session fallbacks.

    Re-expresses the PR #8 semantics (only provider-availability failures
    reroute) over the runtime's ``ClassifiedError``: only classified, fallback-
    eligible provider failures in ``{rate_limit, billing, upstream_rate_limit,
    overloaded}`` qualify.  Never for ``content_policy_blocked`` / ``unknown``
    / ``format_error`` / tool / safety / command errors — those are routing
    decisions for the loop itself, which has its own identical gate
    (``conversation_loop.py`` ``_should_fallback``) that this helper does not
    duplicate at runtime.
    """
    from agent.error_classifier import FailoverReason

    reason = getattr(classified, "reason", None)
    if reason is None:
        return False
    return bool(
        getattr(classified, "should_fallback", False)
        and reason in {
            FailoverReason.rate_limit,
            FailoverReason.billing,
            FailoverReason.upstream_rate_limit,
            FailoverReason.overloaded,
        }
    )


def format_session_fallback_notice(primary_label: str, fallback_label: str) -> str:
    """Brief secret-free UX notice: ``⚠️ {Primary} unavailable — continuing
    with {Fallback}``.  Labels are real ``provider/model`` identifiers
    (never hardcoded brand names / keys / tokens)."""
    primary = (primary_label or "Provider").strip() or "Provider"
    fallback = (fallback_label or "the fallback provider").strip() or "the fallback provider"
    return f"⚠️ {primary} unavailable — continuing with {fallback}"


def record_session_fallback(
    *,
    turn_id: Any,
    from_provider: str,
    from_model: str,
    to_provider: str,
    to_model: str,
    reason: str,
) -> None:
    """Best-effort structured telemetry for a session provider switch.

    Emits one ``[session-fallback]`` log line (provider/model identifiers
    only — no keys, tokens, URLs or payloads).  Total try/except keeps the
    observing path inert: a logging failure can never affect the turn.
    """
    try:
        logger.info(
            "[session-fallback] turn=%s from=%s/%s to=%s/%s reason=%s",
            turn_id, from_provider, from_model, to_provider, to_model, reason,
        )
    except Exception:
        pass