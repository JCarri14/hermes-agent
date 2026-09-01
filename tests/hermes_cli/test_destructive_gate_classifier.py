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
    _is_human_go,
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