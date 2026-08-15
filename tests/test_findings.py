"""VAR-110: terminal findings after invoke(); silence on clean runs."""

from __future__ import annotations

import pytest
from conftest import make_event, make_inspection, make_run_record

from argus.findings import format_run_finding, print_run_finding
from argus.models import ToolFailure
from argus.session import ArgusSession
from argus.storage import load_run


@pytest.mark.unit
def test_format_run_finding_silent_on_clean():
    record = make_run_record(events=[make_event()], status="clean")
    assert format_run_finding(record) is None


@pytest.mark.unit
def test_format_run_finding_silent_failure_missing_dropped_by():
    search = make_event(
        node_name="search",
        status="fail",
        inspection=make_inspection(
            missing=["documents"],
            is_silent=True,
            severity="critical",
            message="missing documents",
        ),
        step_index=0,
    )
    retrieve = make_event(
        node_name="retrieve",
        status="fail",
        inspection=make_inspection(
            missing=["documents"],
            is_silent=True,
            severity="critical",
            message="missing documents",
        ),
        step_index=1,
    )
    record = make_run_record(
        events=[search, retrieve],
        status="silent_failure",
        run_id="20260815-221530-8f3a1c02",
    )
    record.first_failure_step = "retrieve"
    record.root_cause_chain = ["search"]

    text = format_run_finding(record)
    assert text is not None
    assert "[argus] run 8f3a1c02  silent_failure on retrieve" in text
    assert "missing: documents  (dropped by search)" in text
    assert "argus show last" in text
    assert "argus ui" in text


@pytest.mark.unit
def test_format_run_finding_tool_failure_without_missing():
    event = make_event(
        node_name="search",
        status="fail",
        inspection=make_inspection(
            is_silent=True,
            has_tool_failure=True,
            severity="critical",
            tool_failures=[
                ToolFailure(
                    failure_type="empty_result",
                    field_name="results",
                    severity="critical",
                    evidence="empty list",
                )
            ],
        ),
    )
    record = make_run_record(events=[event], status="silent_failure", run_id="abc12345")
    record.first_failure_step = "search"
    text = format_run_finding(record)
    assert text is not None
    assert "silent_failure on search" in text
    assert "empty_result on results" in text


@pytest.mark.unit
def test_format_run_finding_crashed():
    event = make_event(
        node_name="process",
        status="crashed",
        exception="Traceback (most recent call last):\nKeyError: 'documents'",
    )
    record = make_run_record(events=[event], status="crashed", run_id="deadbeef")
    record.first_failure_step = "process"
    text = format_run_finding(record)
    assert text is not None
    assert "crashed on process" in text
    assert "KeyError: 'documents'" in text


@pytest.mark.unit
def test_format_run_finding_interrupted_no_crash_detail():
    event = make_event(node_name="ask_human", status="interrupted")
    record = make_run_record(events=[event], status="interrupted", run_id="pause-01")
    record.interrupt_node = "ask_human"
    text = format_run_finding(record)
    assert text is not None
    assert "interrupted on ask_human" in text
    assert "argus show last" in text


@pytest.mark.unit
def test_print_run_finding_respects_quiet(monkeypatch, capsys):
    event = make_event(node_name="search", status="fail")
    record = make_run_record(events=[event], status="silent_failure")
    record.first_failure_step = "search"
    monkeypatch.setenv("ARGUS_QUIET", "1")
    print_run_finding(record)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


@pytest.mark.unit
def test_finalize_prints_on_failure(capsys):
    session = ArgusSession()
    session.set_node_names(["search"])
    search = session.wrap("search", lambda s: {"results": []})
    search({})
    session.finalize()

    loaded = load_run(session.run_id)
    assert loaded.overall_status == "silent_failure"
    err = capsys.readouterr().err
    assert "[argus] run" in err
    assert "silent_failure" in err
    assert "argus show last" in err


@pytest.mark.unit
def test_finalize_silent_on_clean(capsys):
    session = ArgusSession()
    session.set_node_names(["fetch"])
    fetch = session.wrap(
        "fetch",
        lambda s: {"data": "The quarterly revenue analysis shows a 15% increase"},
    )
    fetch({})
    session.finalize()

    loaded = load_run(session.run_id)
    assert loaded.overall_status == "clean"
    err = capsys.readouterr().err
    assert "[argus]" not in err
