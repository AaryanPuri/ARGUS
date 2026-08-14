"""Deterministic assembly of coding-agent-ready fix prompts.

Turns a recorded :class:`~argus.models.RunRecord` into a plain-markdown prompt a
developer can paste into any coding agent (Claude Code, Cursor, Windsurf, ...).

Three properties are load-bearing and must not regress:

* **Offline** — no network call, no ``argus login``. The source-locator fallback
  is invoked with ``use_llm=False`` for exactly this reason.
* **Deterministic** — the same run id always renders byte-identical output.
* **Jargon-free** — internal status enums and signal ids (``degraded_input``,
  ``CM-003``) are translated to plain English before they reach the prompt.

The prompt targets the *origin* of the failure, not the node where it surfaced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from argus.models import NodeEvent, RunRecord
from argus.storage import load_run

__all__ = [
    "FixPromptError",
    "FixPromptResult",
    "build_fix_prompt",
    "build_fix_prompt_for_record",
]

# Longest rendered value kept in the evidence section. State snapshots are
# capped at max_field_size (50_000) at capture time — dumping that into a
# prompt buries the instruction for a smaller model.
_MAX_VALUE_CHARS = 800

# Statuses that mean "this node did something wrong", as opposed to
# bookkeeping statuses like "retried".
_FAILING_STATUSES = ("crashed", "fail", "degraded_input", "semantic_fail")

# ── Jargon translation ────────────────────────────────────────────────────────
# Every internal enum a developer would have to decode gets a plain-English
# rendering here. Nothing below this line should leak an ARGUS-internal name.

_STATUS_PLAIN = {
    "crashed": "raised an error and stopped",
    "fail": "produced output missing fields the next node requires",
    "degraded_input": (
        "ran with incomplete input because an earlier node did not produce "
        "a field it needed"
    ),
    "semantic_fail": (
        "produced output that looks structurally correct but is wrong in content"
    ),
    "interrupted": "was interrupted before it finished",
    "pass": "completed without an error being raised",
    "retried": "was retried",
}

_TOOL_FAILURE_PLAIN = {
    "error_response": "the external call returned an error response",
    "rate_limit": "the external call was rate-limited",
    "empty_result": "the external call returned an empty result",
    "error_in_data": "the data that came back contained an error payload",
    "partial_failure": "the external call only partly succeeded",
}

# Present tense, for the "Done when" conditions — the past-tense phrasings above
# read wrong in a forward-looking success criterion.
_TOOL_FAILURE_CONDITION = {
    "error_response": "the external call returns an error response",
    "rate_limit": "the external call is rate-limited",
    "empty_result": "the external call returns an empty result",
    "error_in_data": "the data that comes back contains an error payload",
    "partial_failure": "the external call only partly succeeds",
}


class FixPromptError(Exception):
    """Raised when a run cannot produce a meaningful fix prompt."""


@dataclass
class FixPromptResult:
    """Rendered prompt plus the metadata the dashboard route needs."""

    prompt: str
    node: str
    source_path: Optional[str]


# ── Public API ────────────────────────────────────────────────────────────────


def build_fix_prompt(
    run_id: str,
    *,
    node: Optional[str] = None,
    sanitized: bool = False,
) -> str:
    """Deterministically assemble a coding-agent-ready fix prompt for a run's
    root-cause failure. No LLM call. Node defaults to the origin of
    root_cause_chain; pass `node` to target a specific step instead."""
    record = load_run(run_id)
    return build_fix_prompt_for_record(record, node=node, sanitized=sanitized).prompt


def build_fix_prompt_for_record(
    record: RunRecord,
    *,
    node: Optional[str] = None,
    sanitized: bool = False,
) -> FixPromptResult:
    """Same as :func:`build_fix_prompt` but against an already-loaded record.

    Kept separate so the dashboard route can report the targeted node without
    re-loading, and so tests can build records in memory.
    """
    target = _resolve_target_node(record, node)
    event = _find_node_event(record, target)

    if event is None:
        raise FixPromptError(
            f"Node '{target}' has no recorded step in run {record.run_id}."
        )

    if not _has_failure_signal(event):
        raise FixPromptError(
            f"Node '{target}' in run {record.run_id} shows no detected problem — "
            "there is nothing to write a fix prompt about."
        )

    source_path, path_note = _resolve_source(record, target)
    symptom_event = _find_symptom_event(record, target)

    prompt = _render(
        record=record,
        target=target,
        event=event,
        source_path=source_path,
        path_note=path_note,
        symptom_event=symptom_event,
        sanitized=sanitized,
    )
    return FixPromptResult(prompt=prompt, node=target, source_path=source_path)


# ── Target resolution ─────────────────────────────────────────────────────────


def _resolve_target_node(record: RunRecord, node: Optional[str]) -> str:
    """Explicit node → root-cause origin → first failing step."""
    if node:
        known = set(record.graph_node_names or []) | {
            e.node_name for e in record.steps
        }
        if node not in known:
            available = ", ".join(sorted(known)) or "(none recorded)"
            raise FixPromptError(
                f"Node '{node}' is not part of run {record.run_id}. "
                f"Available nodes: {available}"
            )
        return node

    # root_cause_chain is built by walking backward to the origin and is stored
    # in chronological order of first occurrence — chain[0] is the origin.
    if record.root_cause_chain:
        return record.root_cause_chain[0]

    if record.first_failure_step:
        return record.first_failure_step

    raise FixPromptError(
        f"Run {record.run_id} has no recorded failure (status: "
        f"{record.overall_status}). Nothing to fix — pass --node to target a "
        "specific step anyway."
    )


def _find_node_event(record: RunRecord, node_name: str) -> Optional[NodeEvent]:
    """Return the node's event — the last real attempt if the node looped."""
    candidates = [e for e in record.steps if e.node_name == node_name]
    if not candidates:
        return None

    # "retried" marks a superseded attempt; prefer the attempt that stuck.
    real = [e for e in candidates if e.status != "retried"]
    pool = real or candidates
    return max(pool, key=lambda e: (e.attempt_index, e.step_index))


def _find_symptom_event(record: RunRecord, target: str) -> Optional[NodeEvent]:
    """The node where the failure actually surfaced, if not the target itself."""
    crashed = [e for e in record.steps if e.status == "crashed"]
    if crashed:
        last_crash = max(crashed, key=lambda e: e.step_index)
        if last_crash.node_name != target:
            return last_crash

    if record.first_failure_step and record.first_failure_step != target:
        return _find_node_event(record, record.first_failure_step)

    chain = record.root_cause_chain or []
    if len(chain) > 1 and chain[-1] != target:
        return _find_node_event(record, chain[-1])

    return None


def _has_failure_signal(event: NodeEvent) -> bool:
    """Whether this event carries anything worth writing a fix prompt about."""
    if event.status in _FAILING_STATUSES or event.exception:
        return True
    if event.anomaly_signals:
        return True
    if event.semantic_check is not None and not event.semantic_check.passed:
        return True

    insp = event.inspection
    if insp is None:
        return False
    return bool(
        insp.is_silent_failure
        or insp.missing_fields
        or insp.empty_fields
        or insp.type_mismatches
        or insp.tool_failures
        or insp.semantic_signals
        or insp.degraded_fields
        or insp.suspicious_empty_keys
    )


# ── Source resolution ─────────────────────────────────────────────────────────


def _resolve_source(record: RunRecord, node_name: str) -> tuple[Optional[str], str]:
    """Resolve ``file.py:line`` for a node. Returns (path, note).

    ``node_fn_paths`` is recorded relative to the cwd at capture time, so a
    prompt generated from a different directory may hold a path that does not
    resolve. When that happens we re-anchor by basename under the current tree
    and fall back to a note rather than emitting a path that leads nowhere.
    """
    recorded = (record.node_fn_paths or {}).get(node_name)

    if not recorded:
        try:
            # use_llm=False keeps the offline guarantee — the locator's step 4
            # is an LLM call and its default is use_llm=True.
            resolved = _locate_offline(record)
            recorded = resolved.get(node_name)
        except Exception:
            recorded = None

    if not recorded:
        return None, ""

    file_part, _, line_part = recorded.partition(":")
    if Path(file_part).exists():
        return recorded, ""

    reanchored = _reanchor(file_part)
    if reanchored:
        suffix = f":{line_part}" if line_part else ""
        return f"{reanchored}{suffix}", ""

    note = (
        f"> Path recorded as `{recorded}`, relative to the directory the run was "
        "captured in. It does not resolve from the current directory — locate "
        f"`{Path(file_part).name}` in the project before editing."
    )
    return recorded, note


def _locate_offline(record: RunRecord) -> dict:
    from argus.source_locator import locate_node_sources

    return locate_node_sources(record, use_llm=False)


def _reanchor(file_part: str) -> Optional[str]:
    """Find a recorded file by basename under cwd. Unique match only."""
    name = Path(file_part).name
    if not name.endswith(".py"):
        return None
    matches = sorted(
        p for p in Path.cwd().rglob(name) if ".argus" not in p.parts
    )
    if len(matches) != 1:
        return None
    try:
        return str(matches[0].relative_to(Path.cwd()))
    except ValueError:
        return str(matches[0])


# ── Evidence selection ────────────────────────────────────────────────────────


def _fields_of_interest(event: NodeEvent) -> list[str]:
    """Ordered, de-duplicated field names the failure actually implicates.

    Emitting the whole state buries the signal for a smaller model, so evidence
    is narrowed to the fields the diagnostics name.
    """
    out: list[str] = []

    def add(name: Any) -> None:
        if isinstance(name, str) and name and name not in out:
            out.append(name)

    insp = event.inspection
    if insp is not None:
        for f in insp.missing_fields:
            add(f)
        for f in insp.empty_fields:
            add(f)
        for m in insp.type_mismatches:
            add(m.field_name)
        for tf in insp.tool_failures:
            add(tf.field_name)
        for ss in insp.semantic_signals:
            if ss.field_path:
                add(ss.field_path[0])
        for f in insp.degraded_fields:
            add(f)
        for f in insp.suspicious_empty_keys:
            add(f)

    for a in event.anomaly_signals:
        if a.field_path:
            add(a.field_path.split(".")[0])

    for f in _fields_from_exception(event.exception):
        add(f)

    return out


def _fields_from_exception(exc: Optional[str]) -> list[str]:
    """Pull key/attribute names out of a KeyError / AttributeError traceback."""
    if not exc:
        return []
    found: list[str] = []
    for pattern in (
        r"KeyError: '([^']+)'",
        r'KeyError: "([^"]+)"',
        r"AttributeError: .*? has no attribute '([^']+)'",
    ):
        for m in re.finditer(pattern, exc):
            if m.group(1) not in found:
                found.append(m.group(1))
    return found


def _render_state(
    state: Optional[dict],
    fields: list[str],
    *,
    sanitized: bool,
) -> Optional[str]:
    """Render the implicated fields of a state dict, plus a note on the rest."""
    if not state:
        return None

    present = [f for f in fields if f in state]
    if not present:
        # Nothing named — show a bounded sample so the agent still sees shape.
        present = list(state.keys())[:6]

    if sanitized:
        lines = [f"{k}: {_describe_shape(state[k])}" for k in present]
        body = "\n".join(lines)
        block = f"```\n{body}\n```"
    else:
        subset = {k: state[k] for k in present}
        block = f"```json\n{_dumps(subset)}\n```"

    remaining = [k for k in state.keys() if k not in present]
    if remaining:
        block += f"\n\nOther keys present: {', '.join(remaining)}"
    return block


def _dumps(value: Any) -> str:
    """Deterministic JSON rendering, truncated to keep the prompt readable."""
    try:
        text = json.dumps(value, indent=2, default=repr)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) > _MAX_VALUE_CHARS:
        text = text[:_MAX_VALUE_CHARS] + "\n... (truncated)"
    return text


def _describe_shape(value: Any) -> str:
    """Type/shape description with no underlying values — for --sanitized."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return type(value).__name__
    if isinstance(value, str):
        return f"str ({len(value)} chars)" if value else "str (empty)"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} ({len(value)} items)"
    if isinstance(value, dict):
        keys = ", ".join(list(value.keys())[:8])
        return f"dict (keys: {keys})" if keys else "dict (empty)"
    return type(value).__name__


# ── Problem description ───────────────────────────────────────────────────────


def _successors(record: RunRecord, node_name: str) -> list[str]:
    return list((record.graph_edge_map or {}).get(node_name, []))


def _join_names(names: list[str]) -> str:
    """'`a`', '`a` and `b`', '`a`, `b` and `c`' — reads as prose, not a dump."""
    ticked = [f"`{n}`" for n in names]
    if len(ticked) == 1:
        return ticked[0]
    return ", ".join(ticked[:-1]) + f" and {ticked[-1]}"


def _headline(record: RunRecord, target: str, event: NodeEvent) -> str:
    """One-line symptom for the title. Most concrete signal wins."""
    insp = event.inspection

    if event.status == "crashed" and event.exception:
        return f"`{target}` raises {_exception_label(event.exception)} and stops the pipeline"

    if insp is not None:
        critical_tools = [tf for tf in insp.tool_failures if tf.severity == "critical"]
        if critical_tools:
            tf = critical_tools[0]
            plain = _TOOL_FAILURE_PLAIN.get(tf.failure_type, "the external call failed")
            return f"`{target}` returns a result even though {plain}"
        if insp.missing_fields:
            succ = _successors(record, target)
            field = insp.missing_fields[0]
            if succ:
                return (
                    f"`{target}` does not produce the `{field}` field that "
                    f"`{succ[0]}` needs"
                )
            return f"`{target}` does not produce the `{field}` field"
        if insp.empty_fields:
            return f"`{target}` produces an empty `{insp.empty_fields[0]}`"
        if insp.type_mismatches:
            m = insp.type_mismatches[0]
            return (
                f"`{target}` produces `{m.field_name}` as {m.actual_type} "
                f"instead of {m.expected_type}"
            )
        if insp.semantic_signals:
            ss = insp.semantic_signals[0]
            where = ss.dotted_path or "its output"
            return f"`{target}` produces unusable content in `{where}`"
        if insp.degraded_fields:
            upstream = insp.degraded_upstream_node or "an earlier node"
            return (
                f"`{target}` runs without the `{insp.degraded_fields[0]}` "
                f"field from `{upstream}`"
            )

    if event.anomaly_signals:
        return f"`{target}` behaves unexpectedly: {event.anomaly_signals[0].reason}"

    plain = _STATUS_PLAIN.get(event.status, "produced an unexpected result")
    return f"`{target}` {plain}"


def _exception_label(exc: str) -> str:
    """Last traceback line, trimmed to the exception type and message."""
    lines = [ln.strip() for ln in exc.strip().splitlines() if ln.strip()]
    if not lines:
        return "an error"
    last = lines[-1]
    return last if len(last) <= 120 else last[:120] + "..."


def _what_went_wrong(
    record: RunRecord,
    target: str,
    event: NodeEvent,
) -> list[str]:
    """Plain-English paragraphs. No enum names, no signal ids."""
    paras: list[str] = []
    insp = event.inspection

    if event.status == "crashed":
        paras.append(
            f"`{target}` raised an error and stopped. The traceback is below."
        )
    elif event.status == "degraded_input" and insp is not None:
        upstream = insp.degraded_upstream_node or "an earlier node"
        fields = ", ".join(f"`{f}`" for f in insp.degraded_fields) or "a field"
        paras.append(
            f"`{target}` ran without {fields}, because `{upstream}` never "
            "produced it. It did not raise an error — it carried on with "
            "incomplete input."
        )
    else:
        paras.append(
            f"`{target}` ran without raising an error, but its output is wrong."
        )

    if insp is None:
        return paras

    for tf in insp.tool_failures:
        plain = _TOOL_FAILURE_PLAIN.get(tf.failure_type, "the external call failed")
        detail = f" ({tf.evidence})" if tf.evidence else ""
        paras.append(
            f"While producing `{tf.field_name}`, {plain}{detail}. The result was "
            "kept and passed on as if the call had succeeded."
        )

    if insp.missing_fields:
        succ = _successors(record, target)
        for field in insp.missing_fields:
            if succ:
                readers = _join_names(succ[:2])
                paras.append(
                    f"The next node ({readers}) reads `state['{field}']`, but this "
                    f"node's output has no `{field}` key."
                )
            else:
                paras.append(
                    f"A later node reads `state['{field}']`, but this node's "
                    f"output has no `{field}` key."
                )

    for field in insp.empty_fields:
        paras.append(f"`{field}` is present in the output but empty.")

    for m in insp.type_mismatches:
        paras.append(
            f"`{m.field_name}` should be {m.expected_type}, but it is "
            f"{m.actual_type} ({m.actual_value_repr})."
        )

    for ss in insp.semantic_signals:
        where = ss.dotted_path or "the output"
        evidence = f' Example: "{ss.evidence}".' if ss.evidence else ""
        paras.append(f"In `{where}`: {ss.description}.{evidence}")

    for a in event.anomaly_signals:
        paras.append(
            f"Expected {a.expected_behavior}, but observed {a.observed_behavior}."
        )

    return paras


def _done_when(
    record: RunRecord,
    target: str,
    event: NodeEvent,
    symptom_event: Optional[NodeEvent],
) -> list[str]:
    """Checkable success conditions. This is what makes the prompt verifiable."""
    conds: list[str] = []
    insp = event.inspection

    if event.status == "crashed" and event.exception:
        conds.append(
            f"`{target}` runs to completion without raising "
            f"{_exception_label(event.exception)}."
        )

    if insp is not None:
        for tf in insp.tool_failures:
            plain = _TOOL_FAILURE_CONDITION.get(
                tf.failure_type, "the external call fails"
            )
            conds.append(
                f"When {plain}, `{target}` either raises or retries — it never "
                f"returns `{tf.field_name}` as a successful empty result."
            )
        for field in insp.missing_fields:
            conds.append(f"`{target}`'s output dict contains a `{field}` key.")
        for field in insp.empty_fields:
            conds.append(f"`{target}`'s `{field}` value is non-empty.")
        for m in insp.type_mismatches:
            conds.append(f"`{m.field_name}` is {m.expected_type}.")
        for ss in insp.semantic_signals:
            where = ss.dotted_path or "the output"
            conds.append(f"`{where}` no longer contains {ss.category.replace('_', ' ')}.")
        for field in insp.degraded_fields:
            upstream = insp.degraded_upstream_node
            if upstream:
                conds.append(
                    f"`{target}` receives a usable `{field}` from `{upstream}`."
                )

    if symptom_event is not None and symptom_event.exception:
        conds.append(
            f"`{symptom_event.node_name}` no longer fails with "
            f"{_exception_label(symptom_event.exception)}."
        )

    if not conds:
        conds.append(f"`{target}` produces the output the next node expects.")
    return conds


# ── Rendering ─────────────────────────────────────────────────────────────────


def _render(
    *,
    record: RunRecord,
    target: str,
    event: NodeEvent,
    source_path: Optional[str],
    path_note: str,
    symptom_event: Optional[NodeEvent],
    sanitized: bool,
) -> str:
    fn_ref = (record.node_fn_refs or {}).get(target, "")
    fn_name = fn_ref.split(":")[-1] if fn_ref else target
    parts: list[str] = []

    # 1 — Objective. First, so it survives truncation in a small context.
    parts.append(f"# Fix: {_headline(record, target, event)}")

    # 2 — Location.
    if source_path:
        parts.append(f"**Edit this file:** `{source_path}` — function `{fn_name}`")
    else:
        parts.append(
            f"**Edit the function that implements the `{target}` node.** "
            "Its source file was not recorded — search the project for a "
            f"function named `{fn_name}`."
        )
    if path_note:
        parts.append(path_note)

    # 3 — What to do / what went wrong.
    parts.append("## What to do")
    parts.append(
        f"Fix the `{target}` node so the problem described below cannot happen "
        "again. Change the behaviour, not the symptom."
    )

    parts.append("## What went wrong")
    parts.extend(_what_went_wrong(record, target, event))

    # 4 — Evidence.
    evidence = _evidence_section(record, target, event, symptom_event, sanitized)
    if evidence:
        parts.append("## Evidence")
        parts.extend(evidence)

    # 5 — Causal note. Only when the failure surfaced somewhere else.
    if symptom_event is not None:
        parts.append("## Why this file and not the crash site")
        parts.extend(_causal_section(record, target, symptom_event))

    # 6 — Done when.
    parts.append("## Done when")
    conds = _done_when(record, target, event, symptom_event)
    parts.append("\n".join(f"- {c}" for c in conds))

    # 7 — Constraints.
    parts.append("## Constraints")
    parts.append("\n".join(_constraints(record, target, symptom_event)))

    # 8 — Verify.
    parts.append("## Verify")
    parts.append(f"```bash\nargus replay {record.run_id} {target}\n```")
    parts.append(
        "This re-runs the pipeline from this node using the recorded input and "
        "reports whether the failure is gone."
    )

    return "\n\n".join(parts).rstrip() + "\n"


def _evidence_section(
    record: RunRecord,
    target: str,
    event: NodeEvent,
    symptom_event: Optional[NodeEvent],
    sanitized: bool,
) -> list[str]:
    out: list[str] = []
    fields = _fields_of_interest(event)

    rendered_in = _render_state(event.input_state, fields, sanitized=sanitized)
    if rendered_in:
        out.append(f"Input `{target}` received:")
        out.append(rendered_in)

    rendered_out = _render_state(event.output_dict, fields, sanitized=sanitized)
    if rendered_out:
        out.append(f"Output `{target}` produced:")
        out.append(rendered_out)
    elif event.output_dict is None and event.status == "crashed":
        out.append(f"`{target}` produced no output — it raised before returning.")

    if event.exception:
        out.append(f"`{target}` raised:")
        out.append(f"```\n{event.exception.strip()}\n```")

    if symptom_event is not None and symptom_event.exception:
        position = _relative_position(record, target, symptom_event.node_name)
        out.append(f"{position}, `{symptom_event.node_name}` crashed:")
        out.append(f"```\n{symptom_event.exception.strip()}\n```")

    if sanitized:
        out.append(
            "_Values omitted — field names and shapes only. Re-run without "
            "`--sanitized` to include recorded values._"
        )
    return out


def _relative_position(record: RunRecord, target: str, symptom: str) -> str:
    """'Two nodes later' — reads better than raw step indices for an agent."""
    chain = record.root_cause_chain or []
    if target in chain and symptom in chain:
        gap = chain.index(symptom) - chain.index(target)
        if gap == 1:
            return "Immediately after"
        if gap > 1:
            return f"{gap} nodes later"
    return "Later in the run"


def _causal_section(
    record: RunRecord,
    target: str,
    symptom_event: NodeEvent,
) -> list[str]:
    out: list[str] = []
    symptom = symptom_event.node_name
    symptom_path = (record.node_fn_paths or {}).get(symptom)
    where = f"`{symptom_path}`" if symptom_path else f"the `{symptom}` node"

    chain = record.root_cause_chain or []
    if len(chain) > 1:
        narrative = " → ".join(f"`{n}`" for n in chain)
        out.append(
            f"The failure surfaced in {where}, but that is not the bug. It "
            f"propagated: {narrative}."
        )
    else:
        out.append(f"The failure surfaced in {where}, but that is not the bug.")

    do_not_edit = [n for n in chain if n != target]
    if not do_not_edit and symptom != target:
        do_not_edit = [symptom]
    if do_not_edit:
        names = _join_names(do_not_edit)
        out.append(
            f"**Do not edit {names}.** They behave correctly given the input "
            f"they received. Fix `{target}` only."
        )

    if record.correlation and record.correlation.causal_summary:
        out.append(f"Recorded analysis: {record.correlation.causal_summary}")

    return out


def _constraints(
    record: RunRecord,
    target: str,
    symptom_event: Optional[NodeEvent],
) -> list[str]:
    lines = [f"- Change only the `{target}` node's function."]
    lines.append(
        "- Do not change the pipeline's state schema, key names, or the graph wiring."
    )
    lines.append("- Do not edit other node functions.")
    if symptom_event is not None:
        lines.append(
            f"- Do not add defensive guards in `{symptom_event.node_name}` to "
            "hide the missing data — fix the source."
        )
    return lines
