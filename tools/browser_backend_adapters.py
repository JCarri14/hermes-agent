"""Concrete browser backend adapters (BROWSER_EXECUTION_CAPABILITY_V1 §4.1).

These adapters wrap the *existing* source-tree browser surfaces behind the
:class:`~agent.browser_capability_broker.BrowserBackendAdapter` interface —
they do NOT reimplement any browser mechanics:

- ``DomPrimitivesAdapter`` — wraps the ``browser_*`` tool handlers from
  :mod:`tools.browser_tool` (navigate/snapshot/click/type/scroll/back/press/
  console/get_images/vision).
- ``BrowserExecAdapter`` — wraps ``browser_exec`` from
  :mod:`tools.browser_use_cli` (Browser Use CLI backend).
- ``CuaAdapter`` — maps IR verbs onto the CUA surface available in the
  RUNNING runtime (source: computer_use tools; deployed: ``cua_browser_*``).
  V1 targets the source surface; the deployed runtime's ``CuaTypedBrowserRoute``
  is preserved via the adapter's name/version reporting during the
  coexistence window (design §9.3) — the broker never depends on the
  internal action name.
- ``ExtensionLaneAdapter`` — delegates to the existing authoritative
  controller seam (:mod:`tools.browser_extension_router`) when a controller
  is bound; fail-closed with no fallback when the bound lane cannot serve.

All adapters import the wrapped modules lazily inside ``execute()`` so
importing this module never pulls in the gateway or heavy tool stacks, and
mirror the design's mapping table (§4.1): an active backend exposes ITS
surface, not the union.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.browser_capability_broker import (
    BrowserAction,
    BrowserBackendAdapter,
    BrowserVerb,
    ExecutionAttempt,
    ExecutionResult,
    ExecutionStatus,
    ProfileMode,
    RiskProfile,
)

logger = logging.getLogger(__name__)


def _verb_to_tool(verb: BrowserVerb) -> Optional[str]:
    """Map an IR verb to the DOM primitive tool name (design §4.1 mapping).

    ``submit`` has no dedicated DOM primitive; V1 maps it onto
    ``browser_click`` (ref-based) or ``browser_press`` (key-based) — the
    same surface the §4.1 table lists for click/submit.
    """
    return {
        BrowserVerb.NAVIGATE: "browser_navigate",
        BrowserVerb.READ_SNAPSHOT: "browser_snapshot",
        BrowserVerb.CLICK: "browser_click",
        BrowserVerb.SUBMIT: "browser_click",  # click the submit control (ref)
        BrowserVerb.TYPE: "browser_type",
        BrowserVerb.SCROLL: "browser_scroll",
        BrowserVerb.BACK: "browser_back",
        BrowserVerb.PRESS: "browser_press",
        BrowserVerb.CONSOLE_EVAL: "browser_console",
        BrowserVerb.GET_IMAGES: "browser_get_images",
        BrowserVerb.VISION: "browser_vision",
    }.get(verb)


class DomPrimitivesAdapter(BrowserBackendAdapter):
    """DOM primitives lane — deterministic ref-based operations.

    risk_profile: local_throwaway (or existing_profile when the action
    target requires a real-profile session; the adapter reports its actual
    runtime profile via ``risk_profile`` at construction based on the
    configured consent — the broker still decides what is authorized).
    """

    name = "dom"
    version = "1.0"
    idempotency_support = "provider_key"

    def __init__(
        self,
        *,
        capabilities: Optional[List[BrowserVerb]] = None,
        risk_profile: RiskProfile = RiskProfile.LOCAL_THROWAWAY,
    ) -> None:
        all_dom_verbs = [
            BrowserVerb.NAVIGATE, BrowserVerb.READ_SNAPSHOT, BrowserVerb.CLICK,
            BrowserVerb.TYPE, BrowserVerb.SCROLL, BrowserVerb.BACK,
            BrowserVerb.PRESS, BrowserVerb.CONSOLE_EVAL, BrowserVerb.GET_IMAGES,
            BrowserVerb.VISION, BrowserVerb.SUBMIT,
        ]
        self.capabilities = set(capabilities or all_dom_verbs)
        self.risk_profile = risk_profile

    def resolve(self, action: BrowserAction) -> Optional[ExecutionAttempt]:
        if not self.describes_capability(action.verb):
            return None
        tool = _verb_to_tool(action.verb)
        if tool is None and action.verb not in (BrowserVerb.UPLOAD, BrowserVerb.DOWNLOAD):
            return None
        return ExecutionAttempt(action=action, decision=None, envelope={})

    def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
        action = attempt.action
        tool = _verb_to_tool(action.verb)
        if tool is None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                backend_name=self.name,
                backend_version=self.version,
                error="GEA_CAPABILITY_OUT_OF_SCOPE",
                failure_category="GEA_CAPABILITY_OUT_OF_SCOPE",
                retryable=False,
            )
        try:
            from tools import browser_tool

            handler = getattr(browser_tool, tool, None)
            if handler is None:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    backend_name=self.name,
                    backend_version=self.version,
                    error="GEA_BACKEND_UNAVAILABLE",
                    failure_category="GEA_BACKEND_UNAVAILABLE",
                    retryable=True,
                )
            args: Dict[str, Any] = {"task_id": action.task_id}
            if action.target and action.target.url and action.verb is BrowserVerb.NAVIGATE:
                args["url"] = action.target.url
            if action.verb is BrowserVerb.CLICK and action.target.ref:
                args["ref"] = action.target.ref
            if action.verb is BrowserVerb.SUBMIT:
                # submit maps to a click on the submit control (ref) or a
                # press of a key (Enter / button key) when no ref exists.
                if action.target.ref:
                    args["ref"] = action.target.ref
                elif action.payload and action.payload.raw:
                    key = str(action.payload.raw.get("key", "Enter"))
                    args = {"key": key, "task_id": action.task_id}
                    tool = "browser_press"
            if action.verb is BrowserVerb.TYPE and action.target.ref:
                args["ref"] = action.target.ref
                if action.payload.raw:
                    args["text"] = str(action.payload.raw.get("text", ""))
            if action.verb is BrowserVerb.SCROLL and action.payload.raw:
                args["direction"] = str(action.payload.raw.get("direction", "down"))
            if action.verb is BrowserVerb.PRESS and action.payload.raw:
                args["key"] = str(action.payload.raw.get("key", ""))
            if action.verb is BrowserVerb.CONSOLE_EVAL and action.payload.raw:
                args["expression"] = str(action.payload.raw.get("expression", ""))
            if action.verb is BrowserVerb.VISION and action.payload.raw:
                args["question"] = str(action.payload.raw.get("question", ""))
            raw = handler(**args)
            return ExecutionResult(
                status=ExecutionStatus.DONE,
                backend_name=self.name,
                backend_version=self.version,
                result=raw,
                raw_hash=None,
            )
        except Exception as exc:  # deterministic failure surfaced as category
            logger.debug("dom adapter %s failed: %s", action.verb.value, exc)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                backend_name=self.name,
                backend_version=self.version,
                error=str(exc) or "GEA_BACKEND_UNAVAILABLE",
                failure_category="GEA_RETRYABLE_TRANSIENT",
                retryable=True,
            )


class BrowserExecAdapter(BrowserBackendAdapter):
    """Browser Use CLI backend (``browser_exec`` tool) — design §4.1 browser_exec lane."""

    name = "browser_exec"
    version = "1.0"
    idempotency_support = "client_side"

    def __init__(
        self,
        *,
        capabilities: Optional[List[BrowserVerb]] = None,
        risk_profile: RiskProfile = RiskProfile.CLOUD,
    ) -> None:
        self.capabilities = set(
            capabilities
            or [
                BrowserVerb.NAVIGATE, BrowserVerb.READ_SNAPSHOT, BrowserVerb.EXTRACT,
                BrowserVerb.CLICK, BrowserVerb.TYPE, BrowserVerb.SCROLL,
                BrowserVerb.BACK, BrowserVerb.PRESS, BrowserVerb.VISION,
            ]
        )
        self.risk_profile = risk_profile

    def resolve(self, action: BrowserAction) -> Optional[ExecutionAttempt]:
        if not self.describes_capability(action.verb):
            return None
        return ExecutionAttempt(action=action, decision=None, envelope={})

    def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
        action = attempt.action
        try:
            from tools import browser_use_cli

            code = self._build_snippet(action)
            raw = browser_use_cli.browser_exec(
                code=code,
                session="",
                task_id=action.task_id,
                local=(action.profile_mode is ProfileMode.EXISTING_PROFILE),
            )
            return ExecutionResult(
                status=ExecutionStatus.DONE,
                backend_name=self.name,
                backend_version=getattr(browser_use_cli, "__version__", "1.0"),
                result=raw,
            )
        except Exception as exc:
            logger.debug("browser_exec adapter failed: %s", exc)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                backend_name=self.name,
                backend_version=self.version,
                error=str(exc) or "GEA_BACKEND_UNAVAILABLE",
                failure_category="GEA_RETRYABLE_TRANSIENT",
                retryable=True,
            )

    @staticmethod
    def _build_snippet(action: BrowserAction) -> str:
        verb = action.verb
        url = (action.target.url or "") if action.target else ""
        raw = (action.payload.raw or {}) if action.payload else {}
        if verb is BrowserVerb.NAVIGATE:
            return f'new_tab("{url}")'
        if verb is BrowserVerb.READ_SNAPSHOT:
            return "print(page_info())"
        if verb is BrowserVerb.EXTRACT:
            return f'print(page_info())'
        if verb is BrowserVerb.CLICK and action.target.ref:
            return f'click_by_ref("{action.target.ref}")'
        if verb is BrowserVerb.TYPE and action.target.ref:
            text = raw.get("text", "")
            return f'type_into_ref("{action.target.ref}", {text!r})'
        if verb is BrowserVerb.SCROLL:
            direction = raw.get("direction", "down")
            return f'page_scroll(direction={direction!r})'
        if verb is BrowserVerb.BACK:
            return "go_back()"
        if verb is BrowserVerb.PRESS:
            return f'press_key({raw.get("key", "")!r})'
        return "print(page_info())"


class CuaAdapter(BrowserBackendAdapter):
    """CUA lane — maps IR verbs onto the runtime's CUA surface.

    Source runtime: ``cua_browser_*`` is gone (commit f780cb36d); browser
    work goes through ``browser_exec``. The deployed runtime still exposes
    ``cua_browser_*`` via ``CuaTypedBrowserRoute``. This adapter reports the
    surface it actually talks to through ``name/version`` and lets the
    broker stay surface-agnostic (design §9.3 coexistence window).
    """

    name = "cua"
    version = "1.0"
    idempotency_support = "none"

    def __init__(
        self,
        *,
        capabilities: Optional[List[BrowserVerb]] = None,
        risk_profile: RiskProfile = RiskProfile.EXISTING_PROFILE,
    ) -> None:
        self.capabilities = set(
            capabilities
            or [
                BrowserVerb.NAVIGATE, BrowserVerb.READ_SNAPSHOT, BrowserVerb.CLICK,
                BrowserVerb.TYPE, BrowserVerb.SCROLL, BrowserVerb.BACK,
                BrowserVerb.PRESS, BrowserVerb.DIALOG, BrowserVerb.VISION,
                BrowserVerb.UPLOAD, BrowserVerb.DOWNLOAD,
            ]
        )
        self.risk_profile = risk_profile

    def resolve(self, action: BrowserAction) -> Optional[ExecutionAttempt]:
        if not self.describes_capability(action.verb):
            return None
        # CUA attaches to the real (existing) browser profile by design; if
        # the action's profile_mode does not allow that, this lane does not
        # apply — fail closed, do NOT widen.
        if action.profile_mode not in (ProfileMode.EXISTING_PROFILE, ProfileMode.CDP_OVERRIDE):
            return None
        return ExecutionAttempt(action=action, decision=None, envelope={})

    def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
        action = attempt.action
        # Lazy surface detection: source runtime has no cua_browser_* tools.
        try:
            from tools import browser_use_cli  # source surface for browser work

            if action.verb is BrowserVerb.NAVIGATE and action.target.url:
                raw = browser_use_cli.browser_exec(
                    code=f'new_tab("{action.target.url}")',
                    task_id=action.task_id,
                    local=True,
                )
                return ExecutionResult(
                    status=ExecutionStatus.DONE,
                    backend_name=self.name,
                    backend_version="source-surface",
                    result=raw,
                )
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("cua adapter navigate failed: %s", exc)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                backend_name=self.name,
                backend_version="source-surface",
                error=str(exc),
                failure_category="GEA_RETRYABLE_TRANSIENT",
                retryable=True,
            )
        return ExecutionResult(
            status=ExecutionStatus.FAILED,
            backend_name=self.name,
            backend_version="source-surface",
            error="GEA_CAPABILITY_OUT_OF_SCOPE",
            failure_category="GEA_CAPABILITY_OUT_OF_SCOPE",
            retryable=False,
        )


class ExtensionLaneAdapter(BrowserBackendAdapter):
    """Extension/controller-bound lane (authoritative when bound; §4.2.d).

    Delegates to :func:`tools.browser_extension_router.routed_browser_handler`
    semantics: once a controller is bound for an action, missing/ambiguous
    scope, disconnect, or capability mismatch fail closed — the legacy
    backend is never retried.
    """

    name = "extension"
    version = "1.0"
    idempotency_support = "provider_key"

    def __init__(
        self,
        *,
        capabilities: Optional[List[BrowserVerb]] = None,
        risk_profile: RiskProfile = RiskProfile.CONTROLLER_BOUND,
    ) -> None:
        self.capabilities = set(
            capabilities
            or [
                BrowserVerb.NAVIGATE, BrowserVerb.READ_SNAPSHOT, BrowserVerb.CLICK,
                BrowserVerb.TYPE, BrowserVerb.SCROLL, BrowserVerb.BACK,
                BrowserVerb.PRESS, BrowserVerb.DIALOG, BrowserVerb.GET_IMAGES,
                BrowserVerb.VISION,
            ]
        )
        self.risk_profile = risk_profile

    def resolve(self, action: BrowserAction) -> Optional[ExecutionAttempt]:
        if not self.describes_capability(action.verb):
            return None
        return ExecutionAttempt(action=action, decision=None, envelope={})

    def execute(self, attempt: ExecutionAttempt) -> ExecutionResult:
        action = attempt.action
        tool = _verb_to_tool(action.verb)
        if tool is None:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                backend_name=self.name,
                backend_version=self.version,
                error="GEA_CAPABILITY_OUT_OF_SCOPE",
                failure_category="GEA_CAPABILITY_OUT_OF_SCOPE",
                retryable=False,
            )
        try:
            from gateway.browser_control_broker import get_browser_control_broker
            from tools.browser_extension_router import route_browser_tool

            args: Dict[str, Any] = {}
            if action.target.url and action.verb is BrowserVerb.NAVIGATE:
                args["url"] = action.target.url
            if action.target.ref and action.verb in (BrowserVerb.CLICK, BrowserVerb.TYPE):
                args["ref"] = action.target.ref
            if action.payload and action.payload.raw:
                args.update(action.payload.raw)

            # route_browser_tool returns the tool's JSON string result or
            # raises controller errors; the seam is authoritative.
            raw = route_browser_tool(
                tool,
                args,
                fallback=lambda: None,  # extension lane never falls back to legacy
                broker=get_browser_control_broker(),
                enabled=True,
                session_id=action.session_lease_id or "",
                task_id=action.task_id,
                tool_call_id="",
            )
            return ExecutionResult(
                status=ExecutionStatus.DONE,
                backend_name=self.name,
                backend_version=self.version,
                result=raw,
            )
        except Exception as exc:
            logger.debug("extension adapter failed (fail-closed): %s", exc)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                backend_name=self.name,
                backend_version=self.version,
                error=str(exc) or "GEA_BACKEND_UNAVAILABLE",
                failure_category="GEA_RECONCILIATION_REQUIRED",
                retryable=False,  # bound lane failure => fail closed, no fallback
            )