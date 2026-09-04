"""Declarative per-profile observation contract for kanban workers.

AGENT_OBSERVATION_CONTRACT_V1 (faceta P2): a profile declares which
``task_events`` it wants to observe via a ``watch`` block, either in the
profile's ``config.yaml`` (top-level ``watch`` or ``kanban.watch``) or in
``~/.hermes/profiles/<p>/watch.yaml``. The runtime projects a filtered,
read-only view of those events (faceta P4, ``kanban_db.build_observation_feed``)
into the worker's context.

This module ONLY parses and validates rules — it never executes anything,
never touches the board DB, and never spawns workers. Observation is
advisory by contract: an invalid rule is skipped with a warning, never a
crash (fail-open). ``mode: surface`` is declared in the design but is a
no-op here — surface/triggering (P5) is explicitly out of scope for V1.

Rule schema (all fields optional except ``name``)::

    watch:
      - name: "blocked-alert"
        match:
          kinds: [blocked, block_loop_detected]   # event kinds; omit/[*] = any
          scope: {tenants: [*], assignees: [*]}   # task tenant/assignee filters
          payload_contains: null                  # substring over serialized payload
        mode: observe                             # observe (V1); surface ignored
        window_s: 3600                            # look-back window in seconds
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

_log = logging.getLogger(__name__)

DEFAULT_WATCH_WINDOW_S = 3600
VALID_WATCH_MODES = ("observe", "surface")
# Kinds written by the kanban kernel itself (kanban_db._append_event call
# sites) plus the terminal kinds the notifier watches. Kept as the doc
# reference for `kinds`; an unknown kind in a rule is NOT an error (the
# kernel may gain kinds in the future) — it simply never matches.
KERNEL_EVENT_KINDS = (
    "created",
    "claimed",
    "spawned",
    "completed",
    "blocked",
    "block_loop_detected",
    "assigned",
    "commented",
    "promoted",
    "linked",
    "unlinked",
)


@dataclass(frozen=True)
class WatchRule:
    """One parsed ``watch`` rule: a relevance plan over ``task_events``.

    ``None`` in any field means "no constraint" (match anything for that
    dimension). This mirrors the design's ``match`` block semantics:
    ``kinds``/``scope`` limit which events are relevant, and
    ``payload_contains`` further narrows on the serialized payload.
    """

    name: str
    kinds: Optional[frozenset[str]] = None
    tenants: Optional[frozenset[str]] = None
    assignees: Optional[frozenset[str]] = None
    payload_contains: Optional[str] = None
    mode: str = "observe"
    window_s: int = DEFAULT_WATCH_WINDOW_S

    def __post_init__(self) -> None:
        # Dataclass is frozen → object.__setattr__ for normalization.
        object.__setattr__(self, "name", (self.name or "").strip())
        if not self.kinds:
            object.__setattr__(self, "kinds", None)
        if not self.tenants:
            object.__setattr__(self, "tenants", None)
        if not self.assignees:
            object.__setattr__(self, "assignees", None)
        if not self.payload_contains:
            object.__setattr__(self, "payload_contains", None)
        if self.window_s is None or self.window_s <= 0:
            object.__setattr__(self, "window_s", DEFAULT_WATCH_WINDOW_S)

    @property
    def kinds_label(self) -> str:
        return ",".join(sorted(self.kinds)) if self.kinds else "*"


def _as_str_set(value: Any) -> Optional[frozenset[str]]:
    """Normalize ``[*]`` / ``["a","b"]`` / ``None`` to a frozenset or None."""
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f"expected a list of strings, got {type(value).__name__}")
    out = set()
    for item in value:
        item = str(item).strip()
        if item in ("", "*"):
            continue  # wildcard = no constraint
        out.add(item)
    return frozenset(out) if out else None


def _parse_scope(raw: Any) -> tuple[Optional[frozenset[str]], Optional[frozenset[str]]]:
    """Parse ``scope: {tenants: [...], assignees: [...]}``."""
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        raise ValueError("scope must be a mapping {tenants: [...], assignees: [...]}")
    return _as_str_set(raw.get("tenants")), _as_str_set(raw.get("assignees"))


def parse_watch_rules(raw: Any) -> tuple[list[WatchRule], list[str]]:
    """Parse a ``watch`` block into validated rules + human-readable warnings.

    Never raises on malformed input: each invalid rule is skipped with a
    warning string appended to the returned list (fail-open). Accepts
    either a bare list of rules or a dict with a ``watch`` key
    (the ``watch.yaml`` / config shape).
    """
    warnings: list[str] = []
    if isinstance(raw, dict):
        raw = raw.get("watch", raw.get("rules", []))
    if raw is None:
        return [], warnings
    if not isinstance(raw, list):
        warnings.append(f"watch: expected a list of rules, got {type(raw).__name__}; ignored")
        return [], warnings

    rules: list[WatchRule] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            warnings.append(f"watch rule #{idx}: expected a mapping, got {type(item).__name__}; skipped")
            continue
        raw_name = item.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            warnings.append(f"watch rule #{idx}: missing required string 'name'; skipped")
            continue
        name = raw_name.strip()

        match = item.get("match") or {}
        if not isinstance(match, dict):
            warnings.append(f"watch rule {name!r}: 'match' must be a mapping; skipped")
            continue

        try:
            kinds = _as_str_set(match.get("kinds"))
            tenants, assignees = _parse_scope(match.get("scope"))
        except ValueError as exc:
            warnings.append(f"watch rule {name!r}: invalid match ({exc}); skipped")
            continue

        payload_contains = match.get("payload_contains")
        if payload_contains is not None and not isinstance(payload_contains, str):
            warnings.append(
                f"watch rule {name!r}: 'payload_contains' must be a string or null; skipped"
            )
            continue

        mode = str(item.get("mode") or "observe").strip().lower()
        if mode not in VALID_WATCH_MODES:
            warnings.append(
                f"watch rule {name!r}: unknown mode {mode!r} (expected observe); skipped"
            )
            continue
        if mode == "surface":
            warnings.append(
                f"watch rule {name!r}: mode 'surface' (P5 triggering) is out of scope "
                "for V1 — rule ignored as observe-only"
            )
            continue

        try:
            window_s = int(item.get("window_s", DEFAULT_WATCH_WINDOW_S))
        except (TypeError, ValueError):
            warnings.append(f"watch rule {name!r}: invalid window_s; using {DEFAULT_WATCH_WINDOW_S}")
            window_s = DEFAULT_WATCH_WINDOW_S

        rules.append(
            WatchRule(
                name=name,
                kinds=kinds,
                tenants=tenants,
                assignees=assignees,
                payload_contains=payload_contains,
                mode="observe",
                window_s=window_s,
            )
        )
    return rules, warnings


def _watch_rules_from_mapping(raw: Any) -> tuple[list[WatchRule], list[str]]:
    return parse_watch_rules(raw)


def load_watch_rules(
    profile: Optional[str],
    *,
    hermes_home: Optional[Path] = None,
) -> tuple[list[WatchRule], list[str]]:
    """Load + parse the ``watch`` rules for a profile.

    Sources, in order (later wins on duplicate rule names):

    1. ``config.yaml`` of the profile — top-level ``watch``, or
       ``kanban.watch`` (both accepted; ``kanban.watch`` preferred).
    2. ``watch.yaml`` in the profile's HERMES_HOME directory
       (``~/.hermes/profiles/<p>/watch.yaml`` for a named profile,
       ``~/.hermes/watch.yaml`` for the default profile).

    ``hermes_home`` overrides the resolved profile home (used by tests and
    by callers that already know the exact profile dir). Returns
    ``([], [])`` when the profile has no rules — never raises for missing
    files.
    """
    warnings: list[str] = []
    if not profile:
        return [], warnings

    home = hermes_home or _profile_home(profile)
    if home is None:
        return [], warnings

    rules: dict[str, WatchRule] = {}

    # Source 1: config.yaml of the profile (only when we're not already
    # inside that profile's config context — config loading is profile-aware
    # via HERMES_HOME, so reuse load_config with an override).
    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import load_config

        token = set_hermes_home_override(str(home))
        try:
            cfg = load_config() or {}
        finally:
            reset_hermes_home_override(token)
    except Exception as exc:
        _log.debug("kanban watch: config load failed for %s (%s)", profile, exc)
        cfg = {}

    config_raw = None
    kanban_cfg = cfg.get("kanban") if isinstance(cfg.get("kanban"), dict) else {}
    if "watch" in kanban_cfg:
        config_raw = kanban_cfg.get("watch")
        _log.debug("kanban watch: %s rules from config kanban.watch for %s", profile, profile)
    elif "watch" in cfg:
        config_raw = cfg.get("watch")
        _log.debug("kanban watch: %s rules from config top-level watch for %s", profile, profile)
    if config_raw is not None:
        parsed, ws = _watch_rules_from_mapping(config_raw)
        warnings.extend(ws)
        for rule in parsed:
            rules[rule.name] = rule

    # Source 2: watch.yaml in the profile home.
    yaml_path = home / "watch.yaml"
    if yaml_path.is_file():
        try:
            import yaml

            with open(yaml_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            parsed, ws = _watch_rules_from_mapping(data)
            warnings.extend(ws)
            for rule in parsed:
                rules[rule.name] = rule
        except Exception as exc:
            warnings.append(f"watch.yaml for {profile!r} failed to load: {exc}")

    return list(rules.values()), warnings


def _profile_home(profile: str) -> Optional[Path]:
    """Resolve a profile name to its HERMES_HOME directory, best-effort."""
    try:
        from hermes_cli.profiles import normalize_profile_name

        canon = normalize_profile_name(profile)
        if canon == "default":
            from hermes_constants import get_hermes_home

            return get_hermes_home()
        from hermes_cli.profiles import get_profile_dir

        return get_profile_dir(canon)
    except Exception as exc:
        _log.debug("kanban watch: could not resolve home for profile %r (%s)", profile, exc)
        return None


def rule_matches_event(
    rule: WatchRule,
    *,
    kind: str,
    payload: Optional[dict],
    tenant: Optional[str],
    assignee: Optional[str],
    now: Optional[int] = None,
) -> bool:
    """Return whether an event satisfies a rule's relevance plan.

    Pure predicate over already-fetched data — no I/O. ``payload`` is the
    parsed event payload dict (may be None); ``tenant``/``assignee`` come
    from the event's task row.
    """
    if rule.kinds is not None and kind not in rule.kinds:
        return False
    if rule.tenants is not None and (tenant or None) not in rule.tenants:
        return False
    if rule.assignees is not None and (assignee or None) not in rule.assignees:
        return False
    if rule.payload_contains:
        if payload is None:
            return False
        needle = rule.payload_contains.lower()
        try:
            blob = json.dumps(payload, ensure_ascii=False).lower()
        except (TypeError, ValueError):
            blob = str(payload).lower()
        if needle not in blob:
            return False
    return True