"""Tests for the fix_prompt module and the ``argus fix`` command."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from argus.cli.main import app
from argus.fix_prompt import (
    FixPromptError,
    build_fix_prompt,
    build_fix_prompt_for_record,
)
from argus.models import (
    FieldMismatch,
    InspectionResult,
    NodeEvent,
    RunRecord,
    SemanticSignal,
    ToolFailure,
)
from argus.storage import save_run

# Internal names that must never reach a coding agent (spec 5.6).
_JARGON = [
    "degraded_input",
    "semantic_fail",
    "is_silent_failure",
    "suspicious_empty_keys",
    "missing_fields",
    "empty_fields",
    "type_mismatches",
    "SemanticSignal",
    "InspectionResult",
    "CM-003",
    "PH-001",
]


# ── Builders ──────────────────────────────────────────────────────────────────


def _inspection(**kw) -> InspectionResult:
    base = dict(
        is_silent_failure=False,
        missing_fields=[],
        empty_fields=[],
        type_mismatches=[],
        severity="critical",
        message="",
    )
    base.update(kw)
    return InspectionResult(**base)


def _event(step_index: int, node_name: str, status: str, **kw) -> NodeEvent:
    base = dict(
        step_index=step_index,
        node_name=node_name,
        status=status,
        input_state={},
        output_dict={},
        duration_ms=1.0,
        timestamp_utc="2026-08-14T00:00:00Z",
    )
    base.update(kw)
    return NodeEvent(**base)


def _record(**kw) -> RunRecord:
    base = dict(
        run_id="a1b2c3d4e5f6",
        argus_version="0.8.0",
        started_at="2026-08-14T00:00:00Z",
        completed_at="2026-08-14T00:00:01Z",
        duration_ms=1000.0,
        overall_status="crashed",
        first_failure_step=None,
        root_cause_chain=[],
        graph_node_names=["retrieve", "summarize", "classify"],
        graph_edge_map={"retrieve": ["summarize"], "summarize": ["classify"]},
        initial_state={},
    )
    base.update(kw)
    return RunRecord(**base)


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project tree whose node source files actually exist on disk."""
    nodes = tmp_path / "src" / "nodes"
    nodes.mkdir(parents=True)
    (nodes / "retrieval.py").write_text(
        textwrap.dedent("""\
        def retrieve(state):
            return {"docs": []}
        """)
    )
    (nodes / "summarize.py").write_text(
        textwrap.dedent("""\
        def summarize(state):
            return {"summary": ""}
        """)
    )
    (nodes / "classify.py").write_text(
        textwrap.dedent("""\
        def classify(state):
            return {"label": state["summary"]}
        """)
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _cascade_record() -> RunRecord:
    """retrieve (rate-limited, empty docs) → summarize (degraded) → classify (crash)."""
    return _record(
        overall_status="crashed",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve", "summarize", "classify"],
        node_fn_paths={
            "retrieve": "src/nodes/retrieval.py:1",
            "summarize": "src/nodes/summarize.py:1",
            "classify": "src/nodes/classify.py:1",
        },
        steps=[
            _event(
                0,
                "retrieve",
                "fail",
                input_state={"query": "Q3 revenue breakdown", "top_k": 5},
                output_dict={"docs": []},
                inspection=_inspection(
                    empty_fields=["docs"],
                    has_tool_failure=True,
                    tool_failures=[
                        ToolFailure(
                            failure_type="rate_limit",
                            field_name="docs",
                            severity="critical",
                            evidence="HTTP 429 from search API",
                        )
                    ],
                    message="Tool failures: rate_limit on docs",
                ),
            ),
            _event(
                1,
                "summarize",
                "degraded_input",
                input_state={"query": "Q3 revenue breakdown", "docs": []},
                output_dict={"summary": ""},
                inspection=_inspection(
                    degraded_fields=["docs"],
                    degraded_upstream_node="retrieve",
                    message="Degraded input",
                ),
            ),
            _event(
                2,
                "classify",
                "crashed",
                input_state={"query": "Q3 revenue breakdown"},
                output_dict=None,
                exception=(
                    "Traceback (most recent call last):\n"
                    '  File "src/nodes/classify.py", line 2, in classify\n'
                    '    return {"label": state["summary"]}\n'
                    "KeyError: 'summary'"
                ),
            ),
        ],
    )


# ── 1. Happy path: crashed node ───────────────────────────────────────────────


def test_crashed_node_prompt_has_location_exception_and_cause(project: Path) -> None:
    record = _record(
        overall_status="crashed",
        first_failure_step="classify",
        root_cause_chain=["classify"],
        node_fn_paths={"classify": "src/nodes/classify.py:1"},
        steps=[
            _event(
                0,
                "classify",
                "crashed",
                input_state={"summary": "ok"},
                output_dict=None,
                exception=(
                    "Traceback (most recent call last):\n"
                    '  File "src/nodes/classify.py", line 2, in classify\n'
                    "KeyError: 'summary'"
                ),
            )
        ],
    )
    result = build_fix_prompt_for_record(record)

    assert result.node == "classify"
    assert "src/nodes/classify.py:1" in result.prompt
    assert "KeyError: 'summary'" in result.prompt
    assert result.prompt.startswith("# Fix: ")
    # The three pillars are all present as named sections.
    assert "**Edit this file:**" in result.prompt
    assert "## What went wrong" in result.prompt
    assert "## Done when" in result.prompt
    assert f"argus replay {record.run_id} classify" in result.prompt


# ── 2. semantic_fail: no exception, inspection signals only ───────────────────


def test_semantic_fail_node_without_exception(project: Path) -> None:
    record = _record(
        overall_status="silent_failure",
        first_failure_step="summarize",
        root_cause_chain=["summarize"],
        node_fn_paths={"summarize": "src/nodes/summarize.py:1"},
        steps=[
            _event(
                0,
                "summarize",
                "semantic_fail",
                input_state={"docs": ["a", "b"]},
                output_dict={"summary": "TODO: fill in"},
                inspection=_inspection(
                    semantic_signals=[
                        SemanticSignal(
                            sig_id="CM-003",
                            category="placeholder_outputs",
                            severity="critical",
                            description="the value is placeholder text, not real content",
                            field_path=("summary",),
                            evidence="TODO: fill in",
                        )
                    ],
                    message="Placeholder output",
                ),
            )
        ],
    )
    prompt = build_fix_prompt_for_record(record).prompt

    assert "placeholder text" in prompt
    assert "summary" in prompt
    assert "TODO: fill in" in prompt
    # No traceback section, because nothing raised.
    assert "Traceback" not in prompt
    # The signal id itself must not leak.
    assert "CM-003" not in prompt


# ── 3. degraded_input points at the upstream origin, not the symptom ──────────


def test_degraded_cascade_targets_origin_not_crash_site(project: Path) -> None:
    record = _cascade_record()
    result = build_fix_prompt_for_record(record)

    # Q4: one prompt, aimed at the origin.
    assert result.node == "retrieve"
    assert "src/nodes/retrieval.py:1" in result.prompt
    assert "**Edit this file:** `src/nodes/retrieval.py:1`" in result.prompt

    # The causal section must appear and must forbid editing the symptom —
    # scoped to the actual crash site (`classify`), not a blanket claim
    # about every other node in the chain (`summarize` never had its own
    # bug either, but the prompt must not assert that as fact).
    assert "## Why this file and not the crash site" in result.prompt
    assert "**Do not edit `classify`.**" in result.prompt
    assert "`summarize`" in result.prompt  # still named in the propagation narrative

    # The downstream traceback is still shown as evidence.
    assert "KeyError: 'summary'" in result.prompt
    # Rate limiting is described in plain English.
    assert "rate-limited" in result.prompt


def test_causal_section_never_exonerates_the_true_root_cause(project: Path) -> None:
    """--node overriding to a downstream node must not certify an upstream
    node (possibly the real bug) as 'behaving correctly'."""
    record = _cascade_record()
    result = build_fix_prompt_for_record(record, node="classify")

    # classify's own crash is the target now, not a symptom of something
    # else — there is nothing else to (correctly or incorrectly) blame.
    assert "## Why this file and not the crash site" not in result.prompt
    assert "behave correctly" not in result.prompt
    assert "**Do not edit `retrieve`" not in result.prompt


def test_symptom_unrelated_to_target_is_not_attributed(project: Path) -> None:
    """A crash elsewhere in the run must not be blamed on target's failure
    unless it is graph-reachable from target."""
    record = _record(
        overall_status="silent_failure",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        graph_node_names=["retrieve", "summarize", "classify", "audit"],
        graph_edge_map={"retrieve": ["summarize"], "summarize": ["classify"]},
        node_fn_paths={"retrieve": "src/nodes/retrieval.py:1"},
        steps=[
            _event(
                0,
                "retrieve",
                "fail",
                output_dict={"docs": []},
                inspection=_inspection(empty_fields=["docs"], message="empty docs"),
            ),
            _event(
                1,
                "audit",
                "crashed",
                output_dict=None,
                exception="RuntimeError: audit db unreachable",
            ),
        ],
    )
    prompt = build_fix_prompt_for_record(record).prompt
    # `audit` is unrelated (not downstream of `retrieve` in graph_edge_map) —
    # it must not appear as a fabricated symptom/causal claim.
    assert "## Why this file and not the crash site" not in prompt
    assert "audit" not in prompt
    assert "RuntimeError" not in prompt


def test_causal_section_omitted_for_single_node_failure(project: Path) -> None:
    record = _record(
        overall_status="crashed",
        first_failure_step="classify",
        root_cause_chain=["classify"],
        node_fn_paths={"classify": "src/nodes/classify.py:1"},
        steps=[
            _event(
                0,
                "classify",
                "crashed",
                output_dict=None,
                exception="KeyError: 'summary'",
            )
        ],
    )
    prompt = build_fix_prompt_for_record(record).prompt
    assert "## Why this file and not the crash site" not in prompt


# ── 4. Missing node_fn_paths falls back to the source locator ─────────────────


def test_missing_node_fn_paths_falls_back_to_source_locator(project: Path) -> None:
    record = _record(
        overall_status="crashed",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        node_fn_paths=None,
        steps=[
            _event(
                0,
                "retrieve",
                "crashed",
                output_dict=None,
                exception="RuntimeError: boom",
            )
        ],
    )
    result = build_fix_prompt_for_record(record)

    # Located by grepping for `def retrieve(` — no network, no LLM.
    assert result.source_path is not None
    assert "retrieval.py" in result.source_path


def test_unresolvable_path_still_produces_a_usable_prompt(project: Path) -> None:
    record = _record(
        overall_status="crashed",
        first_failure_step="ghost",
        root_cause_chain=["ghost"],
        graph_node_names=["ghost"],
        graph_edge_map={},
        node_fn_paths=None,
        steps=[
            _event(0, "ghost", "crashed", output_dict=None, exception="RuntimeError: boom"),
        ],
    )
    prompt = build_fix_prompt_for_record(record).prompt
    assert "ghost" in prompt
    assert "## Done when" in prompt


def test_path_recorded_from_another_directory_is_reanchored(project: Path) -> None:
    """node_fn_paths is cwd-relative at capture time (spec section 8)."""
    record = _record(
        overall_status="crashed",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        node_fn_paths={"retrieve": "some/other/checkout/retrieval.py:34"},
        steps=[
            _event(0, "retrieve", "crashed", output_dict=None, exception="RuntimeError: boom"),
        ],
    )
    result = build_fix_prompt_for_record(record)
    assert result.source_path == "src/nodes/retrieval.py:34"


def test_windows_drive_letter_path_is_not_mistaken_for_a_missing_colon(
    project: Path,
) -> None:
    """A recorded Windows path has an earlier colon after the drive letter —
    splitting on the first colon (instead of the last) truncates the path to
    just "C" and breaks resolution entirely."""
    record = _record(
        overall_status="crashed",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        node_fn_paths={
            "retrieve": str(project / "src" / "nodes" / "retrieval.py") + ":1"
        },
        steps=[
            _event(0, "retrieve", "crashed", output_dict=None, exception="RuntimeError: boom"),
        ],
    )
    result = build_fix_prompt_for_record(record)
    # Resolves via the exists() fast path — proves the split kept the whole
    # file path, line number included, rather than truncating at a colon
    # embedded earlier in the path (as a Windows drive letter would be).
    assert result.source_path is not None
    assert result.source_path.endswith("retrieval.py:1")


def test_reanchor_ignores_vendored_copies(project: Path) -> None:
    """A stale path must not resolve to a same-named file under .venv —
    _reanchor has to exclude the same noise directories source_locator does."""
    vendored = project / ".venv" / "lib" / "site-packages" / "retrieval.py"
    vendored.parent.mkdir(parents=True)
    vendored.write_text("# not the real file\n")

    record = _record(
        overall_status="crashed",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        node_fn_paths={"retrieve": "some/other/checkout/retrieval.py:1"},
        steps=[
            _event(0, "retrieve", "crashed", output_dict=None, exception="RuntimeError: boom"),
        ],
    )
    result = build_fix_prompt_for_record(record)
    # The real project file is the only legitimate match once .venv is
    # excluded — must not silently point the agent at the vendored copy.
    assert result.source_path == "src/nodes/retrieval.py:1"


# ── 5. --sanitized strips values, keeps diagnostics ───────────────────────────


def test_sanitized_strips_values_but_keeps_diagnostics(project: Path) -> None:
    record = _cascade_record()
    plain = build_fix_prompt_for_record(record).prompt
    scrubbed = build_fix_prompt_for_record(record, sanitized=True).prompt

    # Real recorded values appear in the default output...
    assert "Q3 revenue breakdown" in plain
    # ...and not in the sanitized one.
    assert "Q3 revenue breakdown" not in scrubbed

    # Diagnostics survive.
    assert "docs" in scrubbed
    assert "rate-limited" in scrubbed
    assert "## Done when" in scrubbed
    assert "Values omitted" in scrubbed


def test_sanitized_reports_shapes_not_contents(project: Path) -> None:
    record = _record(
        overall_status="silent_failure",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        node_fn_paths={"retrieve": "src/nodes/retrieval.py:1"},
        steps=[
            _event(
                0,
                "retrieve",
                "fail",
                input_state={"api_token": "sk-live-secret-value"},
                output_dict={"docs": []},
                inspection=_inspection(empty_fields=["docs"], message="Empty fields: docs"),
            )
        ],
    )
    scrubbed = build_fix_prompt_for_record(record, sanitized=True).prompt
    assert "sk-live-secret-value" not in scrubbed
    assert "list (0 items)" in scrubbed


def test_sanitized_does_not_leak_field_mismatch_value(project: Path) -> None:
    """FieldMismatch.actual_value_repr is a repr of the real recorded value —
    it must never survive --sanitized, in the headline or the body."""
    record = _record(
        overall_status="silent_failure",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        node_fn_paths={"retrieve": "src/nodes/retrieval.py:1"},
        steps=[
            _event(
                0,
                "retrieve",
                "fail",
                inspection=_inspection(
                    type_mismatches=[
                        FieldMismatch(
                            field_name="api_key",
                            expected_type="int",
                            actual_type="str",
                            actual_value_repr=repr("sk-live-secret-abc123"),
                        )
                    ],
                    message="type mismatch",
                ),
            )
        ],
    )
    scrubbed = build_fix_prompt_for_record(record, sanitized=True).prompt
    assert "sk-live-secret-abc123" not in scrubbed
    assert "api_key" in scrubbed  # field name is still useful, safe to keep


def test_sanitized_does_not_leak_tool_failure_evidence(project: Path) -> None:
    record = _record(
        overall_status="silent_failure",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        node_fn_paths={"retrieve": "src/nodes/retrieval.py:1"},
        steps=[
            _event(
                0,
                "retrieve",
                "fail",
                inspection=_inspection(
                    has_tool_failure=True,
                    tool_failures=[
                        ToolFailure(
                            failure_type="error_in_data",
                            field_name="docs",
                            severity="critical",
                            evidence="raw response: password=hunter2",
                        )
                    ],
                    message="tool failure",
                ),
            )
        ],
    )
    scrubbed = build_fix_prompt_for_record(record, sanitized=True).prompt
    assert "hunter2" not in scrubbed


def test_sanitized_does_not_leak_semantic_signal_evidence(project: Path) -> None:
    record = _record(
        overall_status="silent_failure",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        node_fn_paths={"retrieve": "src/nodes/retrieval.py:1"},
        steps=[
            _event(
                0,
                "retrieve",
                "semantic_fail",
                inspection=_inspection(
                    semantic_signals=[
                        SemanticSignal(
                            sig_id="CM-003",
                            category="placeholder_outputs",
                            severity="critical",
                            description="placeholder text",
                            field_path=("notes",),
                            evidence="user SSN 123-45-6789",
                        )
                    ],
                    message="placeholder",
                ),
            )
        ],
    )
    scrubbed = build_fix_prompt_for_record(record, sanitized=True).prompt
    assert "123-45-6789" not in scrubbed


def test_sanitized_does_not_leak_exception_message(project: Path) -> None:
    """A traceback's exception message frequently embeds the literal value
    that triggered it (KeyError('<value>')) — only the exception type name
    is safe to show under --sanitized."""
    record = _record(
        overall_status="crashed",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        node_fn_paths={"retrieve": "src/nodes/retrieval.py:1"},
        steps=[
            _event(
                0,
                "retrieve",
                "crashed",
                output_dict=None,
                exception=(
                    "Traceback (most recent call last):\n"
                    "KeyError: 'customer_ssn=123-45-6789'"
                ),
            )
        ],
    )
    scrubbed = build_fix_prompt_for_record(record, sanitized=True).prompt
    assert "123-45-6789" not in scrubbed
    assert "customer_ssn" not in scrubbed
    # The exception type itself is still useful and carries no recorded data.
    assert "KeyError" in scrubbed
    # The section's own claim must now actually be true.
    assert "Values omitted" in scrubbed


def test_sanitized_handles_non_string_dict_keys(project: Path) -> None:
    """A state value that's a dict with non-string keys (e.g. an int-indexed
    mapping) must not crash --sanitized rendering."""
    record = _record(
        overall_status="silent_failure",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        node_fn_paths={"retrieve": "src/nodes/retrieval.py:1"},
        steps=[
            _event(
                0,
                "retrieve",
                "fail",
                output_dict={"scores": {0: 0.9, 1: 0.5}},
                inspection=_inspection(empty_fields=["scores"], message="empty"),
            )
        ],
    )
    # Must not raise.
    scrubbed = build_fix_prompt_for_record(record, sanitized=True).prompt
    assert "dict" in scrubbed


# ── 6. --node override ────────────────────────────────────────────────────────


def test_node_override_targets_a_non_root_cause_node(project: Path) -> None:
    record = _cascade_record()

    default = build_fix_prompt_for_record(record)
    assert default.node == "retrieve"

    override = build_fix_prompt_for_record(record, node="classify")
    assert override.node == "classify"
    assert "src/nodes/classify.py:1" in override.prompt
    assert "KeyError: 'summary'" in override.prompt


def test_node_override_rejects_unknown_node(project: Path) -> None:
    record = _cascade_record()
    with pytest.raises(FixPromptError) as exc:
        build_fix_prompt_for_record(record, node="nope")
    assert "not part of run" in str(exc.value)
    # The error lists what the developer could have asked for.
    assert "retrieve" in str(exc.value)


# ── 7. Clean run errors clearly instead of emitting a garbage prompt ──────────


def test_clean_run_raises_clear_error(project: Path) -> None:
    record = _record(
        overall_status="clean",
        first_failure_step=None,
        root_cause_chain=[],
        steps=[_event(0, "retrieve", "pass", output_dict={"docs": ["a"]})],
    )
    with pytest.raises(FixPromptError) as exc:
        build_fix_prompt_for_record(record)
    assert "no recorded failure" in str(exc.value)


def test_node_override_on_healthy_node_raises(project: Path) -> None:
    record = _record(
        overall_status="clean",
        first_failure_step=None,
        root_cause_chain=[],
        steps=[_event(0, "retrieve", "pass", output_dict={"docs": ["a"]})],
    )
    with pytest.raises(FixPromptError) as exc:
        build_fix_prompt_for_record(record, node="retrieve")
    assert "no detected problem" in str(exc.value)


# ── Non-functional guarantees (spec 5.6) ──────────────────────────────────────


def test_prompt_is_deterministic(project: Path) -> None:
    record = _cascade_record()
    assert (
        build_fix_prompt_for_record(record).prompt
        == build_fix_prompt_for_record(record).prompt
    )


def test_no_internal_jargon_leaks(project: Path) -> None:
    prompt = build_fix_prompt_for_record(_cascade_record()).prompt
    leaked = [term for term in _JARGON if term in prompt]
    assert leaked == [], f"internal jargon leaked into the prompt: {leaked}"


def test_looped_node_uses_last_real_attempt(project: Path) -> None:
    record = _record(
        overall_status="silent_failure",
        first_failure_step="retrieve",
        root_cause_chain=["retrieve"],
        node_fn_paths={"retrieve": "src/nodes/retrieval.py:1"},
        steps=[
            _event(
                0,
                "retrieve",
                "retried",
                output_dict={"docs": []},
                attempt_index=0,
                inspection=_inspection(empty_fields=["docs"], message="first attempt"),
            ),
            _event(
                1,
                "retrieve",
                "fail",
                output_dict={"docs": []},
                attempt_index=1,
                inspection=_inspection(
                    type_mismatches=[
                        FieldMismatch(
                            field_name="docs",
                            expected_type="list",
                            actual_type="str",
                            actual_value_repr="''",
                        )
                    ],
                    message="final attempt",
                ),
            ),
        ],
    )
    prompt = build_fix_prompt_for_record(record).prompt
    # The surviving attempt's diagnostics are the ones described.
    assert "`docs` is list." in prompt


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_build_fix_prompt_loads_by_id_prefix(project: Path) -> None:
    """The locked public entry point: run id (or 8-char prefix) in, markdown out."""
    save_run(_cascade_record())

    by_prefix = build_fix_prompt("a1b2c3d4")
    by_full_id = build_fix_prompt("a1b2c3d4e5f6")

    assert by_prefix == by_full_id
    assert by_prefix.startswith("# Fix: ")
    assert "src/nodes/retrieval.py:1" in by_prefix
    assert build_fix_prompt("a1b2c3d4", node="classify") != by_prefix
    assert "Q3 revenue breakdown" not in build_fix_prompt("a1b2c3d4", sanitized=True)


def test_cli_writes_prompt_to_stdout(project: Path) -> None:
    save_run(_cascade_record())
    result = CliRunner().invoke(app, ["fix", "a1b2c3d4"])
    assert result.exit_code == 0
    assert result.stdout.startswith("# Fix: ")
    assert "src/nodes/retrieval.py:1" in result.stdout


def test_cli_writes_prompt_to_file(project: Path) -> None:
    save_run(_cascade_record())
    out = project / "fix.md"
    result = CliRunner().invoke(app, ["fix", "a1b2c3d4", "-o", str(out)])
    assert result.exit_code == 0
    assert out.read_text(encoding="utf-8").startswith("# Fix: ")


def test_cli_node_and_sanitized_flags(project: Path) -> None:
    save_run(_cascade_record())
    result = CliRunner().invoke(
        app, ["fix", "a1b2c3d4", "--node", "classify", "--sanitized"]
    )
    assert result.exit_code == 0
    assert "Q3 revenue breakdown" not in result.stdout


def test_cli_unknown_run_exits_nonzero(project: Path) -> None:
    result = CliRunner().invoke(app, ["fix", "doesnotexist"])
    assert result.exit_code == 1
