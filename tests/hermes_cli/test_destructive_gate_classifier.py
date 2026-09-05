"""Unit tests for the destructive_gate classifier.

Covers ``hermes_cli.destructive_gate._classify`` / ``evaluate`` / the GO-comment
matcher. The gate is deterministic and read-only: classification must never
depend on external state, LLM, or network. Ambiguity is treated as
DESTRUCTIVE_LIVE (require human GO) — the fail-safe bias.
"""

from hermes_cli.destructive_gate import (
    DESTRUCTIVE_LIVE,
    SAFE,
    GateInput,
    compile_allowlist,
    evaluate,
    _classify,
    _go_precedes_run_start,
    _is_human_go,
    author_allowed_by_allowlist,
    worker_env_guard_reason,
)


# ─────────────────────────── classifier (_classify) ───────────────────────────


def test_safe_no_verbs():
    v = _classify("Add a new endpoint", "Implements /v1/health in the fastapi app.")
    assert v.cls == SAFE


def test_safe_dev_card_remove_word_not_live():
    # "remove unused import" is a dev action, no live-resource marker -> SAFE.
    v = _classify("Remove unused imports", "Remove the unused requests import in runner.py")
    assert v.cls == SAFE


def test_destructive_verb_without_live_marker_is_safe():
    v = _classify("Delete a temp file", "Delete /tmp/scratch.txt locally")
    assert v.cls == SAFE


def test_strict_destructive_verb_without_live_marker_requires_go():
    """Productive boards must not silently allow ambiguous destructive work."""
    v = _classify("Delete a temp file", "Delete /tmp/scratch.txt locally", strict=True)
    assert v.cls == DESTRUCTIVE_LIVE
    assert "strict productive board" in "; ".join(v.reasons)


def test_explicit_allowlist_allows_only_matching_benign_operation():
    allowlist = compile_allowlist([{"pattern": r"delete /tmp/.*", "reason": "local scratch"}])
    allowed = evaluate(GateInput(
        "t1", "Delete a temp file", "Delete /tmp/scratch.txt locally",
        strict=True, allowlist=allowlist,
    ))
    blocked = evaluate(GateInput(
        "t2", "Delete a temp file", "Delete /var/data.txt locally",
        strict=True, allowlist=allowlist,
    ))
    assert allowed.cls == SAFE
    assert "allowlist: local scratch" in "; ".join(allowed.reasons)
    assert blocked.cls == DESTRUCTIVE_LIVE


def test_invalid_allowlist_entry_never_broadens_access():
    allowlist = compile_allowlist([{"pattern": "[", "reason": "broken"}])
    v = evaluate(GateInput(
        "t1", "Delete a temp file", "Delete /tmp/scratch.txt locally",
        strict=True, allowlist=allowlist,
    ))
    assert v.cls == DESTRUCTIVE_LIVE


def test_destructive_live_bucket():
    v = _classify(
        "Cleanup client bucket",
        "Delete the R2 bucket erp-client-a-docs (purge all objects).",
    )
    assert v.cls == DESTRUCTIVE_LIVE


def test_destructive_verb_with_tenant_marker():
    v = _classify("Teardown tenant", "Tear down the live tenant CLIENT_A isolation.")
    assert v.cls == DESTRUCTIVE_LIVE


def test_destructive_live_explicit_directive():
    v = _classify(
        "Drop the staging DB",
        "live_action: destructive\nDrop the database schema staging.",
    )
    assert v.cls == DESTRUCTIVE_LIVE


# ─────────────────────────────── GO matcher ───────────────────────────────────


def test_go_matcher_canonical_forms():
    assert _is_human_go("@go destructive t_abc123", "t_abc123")
    assert _is_human_go("GO_DESTRUCTIVE t_abc123", "t_abc123")
    assert _is_human_go("go destructive t_abc123", "t_abc123")
    assert _is_human_go("\n@go destructive t_abc123\napproved", "t_abc123")


def test_go_matcher_rejects_non_go():
    assert not _is_human_go("delete the bucket, no approval yet", "t_abc123")
    assert not _is_human_go("@go teardown", "t_abc123")  # wrong verb
    assert not _is_human_go("I already did the cleanup", "t_abc123")


def test_go_matcher_wrong_task_id_blocked():
    """A GO intended for another card must NOT authorize this card."""
    assert not _is_human_go("@go destructive t_OTHER", "t_target")
    assert not _is_human_go("go destructive t_wrong", "t_correct")
    # Prefix-only match must fail when task_id differs
    assert not _is_human_go("@go destructive t_target_extra", "t_target")


def test_go_matcher_correct_task_id_allowed():
    """The exact matching task_id must be accepted."""
    assert _is_human_go("@go destructive t_target", "t_target")
    assert _is_human_go("go destructive t_abc123", "t_abc123")


def test_go_matcher_multiline_comment_body():
    """GO marker must be found anywhere in the comment body, including
    after preceding text on a different line (contract: 'anywhere in
    the comment body'). Reproduces QA rejection #4 (no MULTILINE)."""
    assert _is_human_go("approval granted:\n@go destructive t_target", "t_target")
    assert _is_human_go("some note\n@go destructive t_abc123\nmore text", "t_abc123")
    assert _is_human_go("Reviewed the plan.\n\ngo destructive t_card", "t_card")
    # Still rejects wrong task_id even in multiline
    assert not _is_human_go("approval granted:\n@go destructive t_OTHER", "t_target")


# ───────────────────────────────── evaluate ───────────────────────────────────


def test_evaluate_safe_returns_safe():
    inp = GateInput("t1", "Add endpoint", "plain dev change", ["@go destructive t1"])
    assert evaluate(inp).cls == SAFE


def test_evaluate_destructive_no_go_blocks():
    inp = GateInput(
        "t1",
        "Cleanup client bucket",
        "Delete the R2 bucket erp-client-a-docs.",
        [],  # no human GO
    )
    v = evaluate(inp)
    assert v.cls == DESTRUCTIVE_LIVE
    assert "NO human GO recorded" in "; ".join(v.reasons)


def test_evaluate_destructive_with_go_allows():
    inp = GateInput(
        "t1",
        "Cleanup client bucket",
        "Delete the R2 bucket erp-client-a-docs.",
        ["@go destructive t1"],  # human GO recorded
    )
    v = evaluate(inp)
    assert v.cls == DESTRUCTIVE_LIVE  # still classified destructive-live
    assert "human GO recorded" in "; ".join(v.reasons)


def test_evaluate_destructive_self_go_excluded():
    # The dispatcher excludes executor-authored GO comments; here we pass a
    # non-GO body and an executor author is filtered upstream. At the pure
    # layer, an empty human_go_comments list means no GO -> block.
    inp = GateInput("t1", "Drop table", "Drop the live orders table.", [])
    v = evaluate(inp)
    assert v.cls == DESTRUCTIVE_LIVE


# ────────────────────────── v1.1 vocabulary extensions ─────────────────────────


def test_classify_truncate_live():
    v = _classify("Truncate live orders table", "Truncate the live orders table in postgres.")
    assert v.cls == DESTRUCTIVE_LIVE
    assert "destructive verb + live-resource marker" in "; ".join(v.reasons)


def test_classify_revoke_credential_live():
    v = _classify("Revoke credential", "Revoke the live API key for the ERP tenant.")
    assert v.cls == DESTRUCTIVE_LIVE


def test_classify_replace_secret_live():
    v = _classify("Rename user", "Replace the production secret for CLIENT_A.")
    assert v.cls == DESTRUCTIVE_LIVE


# ────────────────────────── v1.1 GO matcher (action_id) ───────────────────────


def test_go_matcher_with_action_id():
    # v1.1 canonical form with a bound action_id.
    assert _is_human_go("@go destructive t_x sha256:ab12", "t_x")
    assert _is_human_go("go destructive t_x sha256:ab12", "t_x")
    assert _is_human_go("GO_DESTRUCTIVE t_x sha256:ab12", "t_x")
    assert _is_human_go("approval:\n@go destructive t_x sha256:ab12\nok", "t_x")
    # Legacy form without action_id still matches (backward compat).
    assert _is_human_go("@go destructive t_x", "t_x")
    # No id at all -> not a GO line.
    assert not _is_human_go("@go destructive", "t_x")


def test_action_id_from_go_line_extracts_bound_action():
    from hermes_cli.destructive_gate import action_id_from_go_line

    assert action_id_from_go_line("@go destructive t_x sha256:ab12") == "sha256:ab12"
    assert action_id_from_go_line("@go destructive t_x") is None
    assert action_id_from_go_line("not a go") is None


# ─────────────────────── deterministic action_id digest ───────────────────────


def test_digest_deterministic():
    from hermes_cli.destructive_gate import compute_destructive_action_id

    t = "Truncate live orders table"
    b = "Truncate the live orders table in postgres."
    a1 = compute_destructive_action_id(t, b, tenant="CLIENT_A", assignee="dev")
    a2 = compute_destructive_action_id(t, b, tenant="CLIENT_A", assignee="dev")
    assert a1 == a2  # same input -> same digest
    assert a1.startswith("sha256:")
    assert len(a1) == len("sha256:") + 16
    # Different tenant -> different digest.
    b_other = compute_destructive_action_id(t, b, tenant="CLIENT_B", assignee="dev")
    assert a1 != b_other
    # Different executor -> different digest (GO not portable between workers).
    c_other = compute_destructive_action_id(t, b, tenant="CLIENT_A", assignee="other")
    assert a1 != c_other


def test_declared_action_id_wins_over_derived_digest():
    from hermes_cli.destructive_gate import compute_destructive_action_id

    body = "destructive_action_id: sha256:declaredabc\n" \
           "destructive_verb: delete\ndestructive_resource: r2://x\n" \
           "destructive_tenant: CLIENT_A"
    assert compute_destructive_action_id("Cleanup", body, tenant="WHATEVER", assignee="dev") \
        == "sha256:declaredabc"


# ────────────────── v1.1 fail-closed vocabulary (QA rework) ──────────────────


def test_classify_unknown_destructive_verb_live_marker_blocks():
    """Verbs outside DESTRUCTIVE_VERBS that act on a live INFRA resource must
    NOT fall through to SAFE — UNKNOWN/ambiguity => DESTRUCTIVE_LIVE
    (reproduces QA findings: erase/terminate/rm -rf were SAFE)."""
    v1 = _classify("Erase the production bucket",
                   "Erase the R2 bucket erp-client-a-docs before migration.")
    assert v1.cls == DESTRUCTIVE_LIVE
    assert "UNKNOWN => DESTRUCTIVE_LIVE" in "; ".join(v1.reasons)
    v2 = _classify("Terminate the live R2 object storage",
                   "Terminate the live R2 object storage endpoint CLIENT_A.")
    assert v2.cls == DESTRUCTIVE_LIVE
    v3 = _classify("rm -rf the live bucket", "rm -rf the live bucket r2://erp-client-a-docs")
    assert v3.cls == DESTRUCTIVE_LIVE


def test_classify_infra_marker_without_verb_conservative():
    """A live INFRA resource marker without any known destructive verb is
    ambiguous => gated (conservative, fail-closed)."""
    v = _classify("Update the production deployment",
                  "Update the production deployment notes and status page.")
    assert v.cls == DESTRUCTIVE_LIVE
    assert "UNKNOWN => DESTRUCTIVE_LIVE" in "; ".join(v.reasons)


def test_classify_identity_marker_without_verb_is_safe():
    """v1.1 IDENTITY markers (user/credential/secret/api key) only signal a
    destructive target WITH a destructive verb — a benign feature card like
    'Add user profile page' must stay SAFE (no infra marker, no verb)."""
    v = _classify("Add user profile page", "Add a new user profile page to the app.")
    assert v.cls == SAFE


# ─────────────────────── v1.2 adversarial fixes (pure units) ──────────────────

def test_worker_env_guard_reason_detects_spawn_env(monkeypatch):
    """HIGH-1: the env-context guard fires only when worker/dispatcher spawn
    variables (HERMES_KANBAN_TASK / WORKSPACE / DB) are present — the exact
    markers `agent/delegation_context.py` injects into real worker spawns."""
    assert worker_env_guard_reason() is None  # operator terminal: no markers
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_x")
    reason = worker_env_guard_reason()
    assert reason is not None
    assert "operator terminal" in reason
    assert "HERMES_KANBAN_TASK" in reason
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "/tmp/w")
    assert worker_env_guard_reason() is not None
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE")
    monkeypatch.setenv("HERMES_KANBAN_DB", "/tmp/k.db")
    assert worker_env_guard_reason() is not None


def test_author_allowed_by_allowlist_fail_closed():
    """HIGH-1 allowlist semantics: empty/unset -> unrestricted (compat);
    non-empty -> exact case-insensitive membership; empty author never passes."""
    assert author_allowed_by_allowlist("alice", go_allowlist=None) is True
    assert author_allowed_by_allowlist("alice", go_allowlist=[]) is True
    assert author_allowed_by_allowlist("alice", go_allowlist=["alice"]) is True
    assert author_allowed_by_allowlist("ALICE", go_allowlist=["alice"]) is True
    assert author_allowed_by_allowlist("bob", go_allowlist=["alice"]) is False
    assert author_allowed_by_allowlist("", go_allowlist=["alice"]) is False
    assert author_allowed_by_allowlist(None, go_allowlist=["alice"]) is False
    assert author_allowed_by_allowlist("alice", go_allowlist=[" alice ", "bob"]) is True


def test_go_precedes_run_start_ordering_fail_closed():
    """HIGH-2 pure ordering: GO must be at-or-before the run start; missing
    either timestamp (incl. zero) is fail-closed."""
    ok, why = _go_precedes_run_start(100, 200)
    assert ok is True and why == ""
    ok, _ = _go_precedes_run_start(200, 200)  # boundary: at run start -> OK
    assert ok is True
    ok, why = _go_precedes_run_start(201, 200)  # ex-post -> blocked
    assert ok is False
    assert "AFTER the run" in why
    ok, why = _go_precedes_run_start(None, 200)
    assert ok is False and "missing" in why
    ok, why = _go_precedes_run_start(100, None)
    assert ok is False and "missing" in why
    ok, why = _go_precedes_run_start(0, 200)  # zero timestamp = missing
    assert ok is False and "missing" in why
    ok, why = _go_precedes_run_start(100, 0)
    assert ok is False and "missing" in why
