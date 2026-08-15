"""Short terminal summary after invoke() — silent on clean runs (VAR-110)."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argus.models import NodeEvent, RunRecord

_QUIET_VALUES = frozenset({"1", "true", "yes", "on"})
_SKIP_STATUSES = frozenset({"retried", "skipped", "pass"})
_FAIL_STATUSES = frozenset({"fail", "crashed", "semantic_fail", "degraded_input", "interrupted"})


def _quiet() -> bool:
    return os.environ.get("ARGUS_QUIET", "").strip().lower() in _QUIET_VALUES


def _short_run_id(run_id: str) -> str:
    """Prefer the unique tail of ``YYYYMMDD-HHMMSS-hex`` ids; else first 8 chars."""
    parts = run_id.rsplit("-", 1)
    if len(parts) == 2 and 6 <= len(parts[1]) <= 12 and parts[1].isalnum():
        return parts[1][:8]
    return run_id[:8] if len(run_id) > 8 else run_id


def _failing_event(record: RunRecord) -> NodeEvent | None:
    name = record.first_failure_step or record.interrupt_node
    if name:
        for event in record.steps:
            if event.node_name == name and event.status not in _SKIP_STATUSES:
                return event
    for event in record.steps:
        if event.status in _FAIL_STATUSES:
            return event
    return None


def _dropped_by(record: RunRecord, event: NodeEvent, fail_node: str | None) -> str | None:
    insp = event.inspection
    if insp is not None and insp.degraded_upstream_node:
        culprit = insp.degraded_upstream_node
        if culprit != fail_node:
            return culprit
    if record.root_cause_chain:
        culprit = record.root_cause_chain[0]
        if culprit != fail_node:
            return culprit
    return None


def _detail_line(record: RunRecord, event: NodeEvent | None, fail_node: str | None) -> str | None:
    if event is None:
        return None

    insp = event.inspection
    if insp is not None and insp.missing_fields:
        fields = ", ".join(insp.missing_fields)
        culprit = _dropped_by(record, event, fail_node)
        if culprit:
            return f"missing: {fields}  (dropped by {culprit})"
        return f"missing: {fields}"

    if insp is not None and insp.empty_fields:
        return f"empty: {', '.join(insp.empty_fields)}"

    if insp is not None and insp.tool_failures:
        tf = insp.tool_failures[0]
        name = tf.field_name or tf.failure_type
        return f"{tf.failure_type} on {name}" if tf.field_name else tf.failure_type

    if event.exception:
        first = event.exception.strip().splitlines()[-1].strip()
        if first:
            return first[:120]

    if insp is not None and insp.message and insp.message != "All checks passed":
        return insp.message[:120]

    if event.status == "semantic_fail":
        return "semantic_fail"

    return None


def format_run_finding(record: RunRecord) -> str | None:
    """Return a 2–3 line terminal summary, or ``None`` when the run is clean."""
    if record.overall_status == "clean":
        return None

    short_id = _short_run_id(record.run_id)
    fail_node = record.first_failure_step or record.interrupt_node
    header = f"[argus] run {short_id}  {record.overall_status}"
    if fail_node:
        header += f" on {fail_node}"

    lines = [header]
    detail = _detail_line(record, _failing_event(record), fail_node)
    if detail:
        lines.append(f"        {detail}")
    lines.append("        argus show last   |  argus ui")
    return "\n".join(lines)


def print_run_finding(record: RunRecord) -> None:
    """Print to stderr when something is wrong. No-op on clean runs or ARGUS_QUIET."""
    if _quiet():
        return
    text = format_run_finding(record)
    if text:
        print(text, file=sys.stderr)  # noqa: T201
