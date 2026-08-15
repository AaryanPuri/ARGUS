---
name: argus-debug
description: >-
  Debug ARGUS-instrumented LangGraph pipelines and agent runs. Use when a
  pipeline failed, a silent failure dropped fields, an agent run returned
  empty documents, or the user mentions ARGUS, argus show, replay, or
  .argus/runs. Read the RunRecord JSON instead of guessing from logs.
---

# ARGUS debug loop

This project uses ARGUS to watch agent pipelines. ARGUS already recorded a
structured artifact you do not get from a normal Python crash. **Read it
instead of guessing from terminal logs.**

## Where runs live

- Directory: `<project-root>/.argus/runs/<run-id>.json`
- Project root: nearest ancestor with `.git` or `pyproject.toml`, or `$ARGUS_DIR`
- Each file is a full `RunRecord` (nodes, I/O, inspections, root-cause chain)

## Debug a failed or empty run

1. `argus list` — newest runs first; copy the id.
2. `argus show last` or `argus show <id>` — statuses, warnings, root cause.
3. `argus show <id> --json` or open `.argus/runs/<id>.json` — do not reconstruct
   the story from stdout/stderr.
4. In the JSON, start with:
   - `overall_status` (`clean` | `silent_failure` | `crashed` | `semantic_fail` | `interrupted`)
   - `first_failure_step`, `root_cause_chain`
   - `steps[]`: `node_name`, `status`, `output_dict`, `exception`, `inspection`
     (`missing_fields`, `empty_fields`, `tool_failures`, `semantic_signals`)
5. Empty documents / dropped keys are usually a **silent failure** on an
   upstream node, not the node that crashed later.

## Replay after a fix

```bash
argus replay <run-id> <node>
argus diff <original-id> <replay-id>
argus ui
```

`argus replay <id> <node> --only` re-runs one node in isolation.

## Attach happy path (first wiring)

Use `ArgusWatcher.attach()`. Heuristics-first. LLM judge off. No login.
No TypedDict rewrite.

Package: `pip install argus-agents` (not `argus`).

```python
from argus import ArgusWatcher

watcher = ArgusWatcher()            # semantic_judge=False by default
app = watcher.attach(graph)         # StateGraph or already-compiled app
result = app.invoke(initial_state)  # persists when invoke/batch/stream returns
print(watcher.run_id)
```

- Do not convert plain-dict state to TypedDict. `{**state, ...}` returns are fine.
- `finalize()` is optional; `attach()` persists on the outermost graph call.
- Do not enable `semantic_judge=True` unless the user already ran `argus key set`.
- `argus login` is optional hosted sync only — not required for local detection.

If skills are missing, run `argus init` and commit the files it writes.
