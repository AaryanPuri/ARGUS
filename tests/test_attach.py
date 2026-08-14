"""VAR-107: ArgusWatcher.attach() — one API, persist without finalize()."""

from __future__ import annotations

from typing import TypedDict

import pytest

from argus.storage import _runs_path, load_run
from argus.watcher import ArgusWatcher, _is_compiled_graph

pytest.importorskip("langgraph")

from langgraph.graph import END, StateGraph  # noqa: E402

_WATCH_KW = dict(semantic_judge=False, investigate=False, record_http=False)


class _S(TypedDict):
    n: int


def _linear_state_graph() -> StateGraph:
    g = StateGraph(_S)
    g.add_node("a", lambda s: {"n": s["n"] + 1})
    g.add_node("b", lambda s: {"n": s["n"] + 1})
    g.set_entry_point("a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g


def _cyclic_state_graph() -> StateGraph:
    def inc(state: _S) -> dict:
        return {"n": state["n"] + 1}

    def route(state: _S) -> str:
        return "inc" if state["n"] < 3 else END

    g = StateGraph(_S)
    g.add_node("inc", inc)
    g.set_entry_point("inc")
    g.add_conditional_edges("inc", route, {"inc": "inc", END: END})
    return g


@pytest.mark.unit
def test_is_compiled_graph_detects_builder_vs_app():
    """attach() routing must not confuse a StateGraph builder with a compiled app."""

    class Builder:
        def add_node(self, *a, **k):
            pass

        def compile(self, *a, **k):
            return self

    class App:
        def invoke(self, state):
            return state

        builder = Builder()

    assert _is_compiled_graph(Builder()) is False
    assert _is_compiled_graph(App()) is True


@pytest.mark.unit
def test_attach_uncompiled_returns_invokable_and_persists():
    """attach(StateGraph) returns a compiled app; invoke persists without finalize()."""
    watcher = ArgusWatcher(**_WATCH_KW)
    app = watcher.attach(_linear_state_graph())
    result = app.invoke({"n": 0})
    assert result["n"] == 2
    assert watcher._session is not None
    assert watcher._session._completed
    record = load_run(watcher.run_id)
    assert record.overall_status == "clean"
    assert (_runs_path() / f"{watcher.run_id}.json").exists()


@pytest.mark.unit
def test_attach_compiled_and_persists_without_reassignment():
    """attach(compiled) instruments the original object — no 'wrong method' trap."""
    compiled = _linear_state_graph().compile()
    watcher = ArgusWatcher(**_WATCH_KW)
    returned = watcher.attach(compiled)
    result = compiled.invoke({"n": 0})
    assert result["n"] == 2
    assert returned is not None
    assert watcher._session is not None
    assert watcher._session._completed
    record = load_run(watcher.run_id)
    assert {e.node_name for e in record.steps} >= {"a", "b"}


@pytest.mark.unit
def test_constructor_uncompiled_persists_without_finalize():
    """ArgusWatcher(graph); graph.compile(); invoke() still persists (compat)."""
    graph = _linear_state_graph()
    watcher = ArgusWatcher(graph, **_WATCH_KW)
    app = graph.compile()
    app.invoke({"n": 0})
    assert watcher._session is not None
    assert watcher._session._completed
    assert load_run(watcher.run_id).overall_status == "clean"


@pytest.mark.unit
def test_watch_compiled_thin_wrapper_persists():
    """watch_compiled() stays compatible and persists without finalize()."""
    compiled = _linear_state_graph().compile()
    watcher = ArgusWatcher(**_WATCH_KW)
    app = watcher.watch_compiled(compiled)
    app.invoke({"n": 0})
    assert load_run(watcher.run_id) is not None


@pytest.mark.unit
def test_attach_cyclic_persists_without_finalize():
    """Cyclic graphs persist when invoke() returns — no manual finalize()."""
    watcher = ArgusWatcher(**_WATCH_KW)
    app = watcher.attach(_cyclic_state_graph())
    assert watcher._session is not None
    assert watcher._session._is_cyclic
    result = app.invoke({"n": 0})
    assert result["n"] == 3
    assert watcher._session._completed
    record = load_run(watcher.run_id)
    assert record.is_cyclic
    inc_steps = [e for e in record.steps if e.node_name == "inc"]
    assert len(inc_steps) == 3


@pytest.mark.unit
def test_attach_finalize_idempotent_no_double_save(monkeypatch):
    """invoke() persist + explicit finalize() writes the run file once."""
    from argus import session as session_mod

    calls: list[str] = []
    original = session_mod.save_run

    def counting_save(record):
        calls.append(record.run_id)
        return original(record)

    monkeypatch.setattr(session_mod, "save_run", counting_save)

    watcher = ArgusWatcher(**_WATCH_KW)
    app = watcher.attach(_linear_state_graph())
    app.invoke({"n": 0})
    watcher.finalize()
    watcher.finalize()

    assert len(calls) == 1
    run_files = list(_runs_path().glob(f"{watcher.run_id}.json"))
    assert len(run_files) == 1


@pytest.mark.unit
def test_attach_cyclic_finalize_idempotent_no_double_save(monkeypatch):
    """Cyclic persist-on-invoke + finalize() still saves once."""
    from argus import session as session_mod

    calls: list[str] = []
    original = session_mod.save_run

    def counting_save(record):
        calls.append(record.run_id)
        return original(record)

    monkeypatch.setattr(session_mod, "save_run", counting_save)

    watcher = ArgusWatcher(**_WATCH_KW)
    app = watcher.attach(_cyclic_state_graph())
    app.invoke({"n": 0})
    watcher.finalize()

    assert len(calls) == 1


@pytest.mark.unit
def test_watch_rejects_compiled_and_points_at_attach():
    """watch() on a compiled app still errors, but names attach() as the fix."""
    compiled = _linear_state_graph().compile()
    watcher = ArgusWatcher(**_WATCH_KW)
    with pytest.raises(ValueError, match="attach"):
        watcher.watch(compiled)
