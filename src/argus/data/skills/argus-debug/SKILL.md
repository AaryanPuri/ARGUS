---
name: argus-debug
description: >-
  Wire ARGUS onto a LangGraph pipeline and debug ARGUS-instrumented agent runs.
  Use when the user wants to attach ArgusWatcher, wire ARGUS, add monitoring,
  or run first-time setup; when a pipeline failed, a silent failure dropped
  fields, an agent run returned empty documents; or they mention ARGUS,
  argus show, argus check, pytest --argus, replay, or .argus/runs. Find the graph and call
  ArgusWatcher.attach() — do not rewrite types first. Read the RunRecord
  JSON instead of guessing from logs.
---

# ARGUS: attach, then debug

This project uses ARGUS to watch agent pipelines. On first setup, find the
graph and call `ArgusWatcher.attach()`. After a run, ARGUS recorded a
structured artifact you do not get from a normal Python crash. **Read it
instead of guessing from terminal logs.**

## First wiring (attach ARGUS)

Do this when ARGUS is not attached yet, or the user asks to add monitoring,
wire ARGUS, or set up `ArgusWatcher`. Attach with a small diff. Do not
rewrite the state type or node signatures.

Heuristics, tool-failure scan, placeholders, empty outputs, and crashes work
without TypedDict. Do not convert plain-dict state to TypedDict. Do not
change node return shapes — returning `{**state, ...}` is fine. Type hints
can be suggested after the first run; they are not a setup gate.

### 1. Find the graph

Find the file where the `StateGraph` is defined (or the already-compiled app).
Note whether nodes are sync or async. Linear, fan-out/fan-in, and cyclic
graphs all persist automatically after the outermost `invoke()` / `batch()` /
`stream()` returns — no `finalize()` call needed.

Print a short summary, then integrate. Do not "fix compatibility" by rewriting
types first.

### 2. Install

```bash
pip install argus-agents
```

The PyPI package is `argus-agents`, not `argus`. Default install includes the
CLI, LangGraph adapter, and UI. LLM judge is off by default — heuristics only.

### 3. Attach (happy path)

Heuristics-first. LLM judge off. No login. No TypedDict rewrite.

```python
from argus import ArgusWatcher

watcher = ArgusWatcher()            # semantic_judge=False by default
app = watcher.attach(graph)         # StateGraph or already-compiled app
result = app.invoke(initial_state)  # persists when invoke/batch/stream returns
print(watcher.run_id)
```

If you prefer compiling yourself:

```python
watcher = ArgusWatcher(graph)       # uncompiled StateGraph
app = graph.compile()
result = app.invoke(initial_state)
```

If node functions are async, use `await app.ainvoke()`.

- Keep existing node signatures. `{**state, ...}` returns are fine.
- `finalize()` is optional; `attach()` persists on the outermost graph call.
- Do not enable `semantic_judge=True` unless the user already ran `argus key set`.
- `argus login` is optional hosted sync only — not required for local detection.
- Defaults: `record_http` and `persist_state` are on. Only add extra kwargs
  (`redact_keys`, `validators`, `strict=True`) if needed. Do not add a large
  config block by default.

After the first invoke:

```bash
argus show last         # first aha is in the terminal if something is wrong
argus list              # see all recorded runs
argus show <id>         # inspect a specific run by ID
argus check last        # CI gate — exit 1 on crash / silent failure / semantic fail
argus ui                # empty table = wrong dir or no runs yet
```

If skills are missing, run `argus init` and commit the files it writes.

## CI gate (fail the build)

After a standalone run (script, notebook, or CI job):

```bash
argus check last          # exit 0 if clean, 1 if not
argus check <run-id>      # same gate for a specific run
```

Fails on crash, silent failure, semantic fail, missing fields, or tool failures —
not just when someone opens the dashboard.

In pytest, add `--argus` so graph invokes are watched automatically (no
`ArgusWatcher` in the test file required):

```bash
pytest --argus
```

Clean pipelines stay passing tests; unclean instrumented invokes fail that test.
Tests that never invoke a graph are unchanged. Heuristics only (judge off).

## Where runs live

- Directory: `<project-root>/.argus/runs/<run-id>.json`
- Project root: nearest ancestor with `.git` or `pyproject.toml`, or `$ARGUS_DIR`
- Each file is a full `RunRecord` (nodes, I/O, inspections, root-cause chain)

## Debug a failed or empty run

1. `argus list` — newest runs first; copy the id.
2. `argus show last` or `argus show <id>` — statuses, warnings, root cause.
3. `argus fix <id>` — paste-ready prompt aimed at the root-cause node (not the crash site).
4. `argus show <id> --json` or open `.argus/runs/<id>.json` — do not reconstruct
   the story from stdout/stderr.
5. In the JSON, start with:
   - `overall_status` (`clean` | `silent_failure` | `crashed` | `semantic_fail` | `interrupted`)
   - `first_failure_step`, `root_cause_chain`
   - `steps[]`: `node_name`, `status`, `output_dict`, `exception`, `inspection`
     (`missing_fields`, `empty_fields`, `tool_failures`, `semantic_signals`)
6. Empty documents / dropped keys are usually a **silent failure** on an
   upstream node, not the node that crashed later.

## Replay after a fix

```bash
argus replay <run-id> <node>
argus diff <original-id> <replay-id>
argus ui
```

`argus replay <id> <node> --only` re-runs one node in isolation.
