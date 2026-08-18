"""Auto-instrument LangGraph graphs during ``pytest --argus``.

Patches ``StateGraph.compile`` (and ``Pregel.invoke`` as a fallback) so test
code that calls ``graph.compile().invoke(...)`` is watched by ARGUS without
an explicit ``ArgusWatcher.attach()``. Idempotent; call
``uninstall_auto_instrumentation()`` to restore originals (used by tests).
"""

from __future__ import annotations

import functools
import threading
from typing import Any

_tls = threading.local()
_installed = False
_originals: dict[str, Any] = {}


def install_auto_instrumentation() -> None:
    """Patch LangGraph compile/invoke so pytest can inspect pipeline runs."""
    global _installed
    if _installed:
        return
    try:
        from langgraph.graph.state import StateGraph
    except ImportError:
        return

    _wrap_compile(StateGraph)
    _wrap_pregel_invoke()
    _installed = True


def uninstall_auto_instrumentation() -> None:
    """Restore LangGraph methods patched by ``install_auto_instrumentation``."""
    global _installed
    if not _installed:
        return
    try:
        from langgraph.graph.state import StateGraph

        original = _originals.get("compile")
        if original is not None:
            StateGraph.compile = original  # type: ignore[method-assign]
    except ImportError:
        pass
    try:
        from langgraph.pregel import Pregel

        original_invoke = _originals.get("invoke")
        if original_invoke is not None:
            Pregel.invoke = original_invoke  # type: ignore[method-assign]
    except ImportError:
        pass
    _originals.clear()
    _installed = False


def _in_attach() -> bool:
    return bool(getattr(_tls, "in_attach", False))


def _attach(compiled: Any) -> Any:
    from argus import ArgusWatcher

    _tls.in_attach = True
    try:
        watcher = ArgusWatcher(
            semantic_judge=False,
            investigate=False,
            record_http=False,
        )
        return watcher.attach(compiled)
    finally:
        _tls.in_attach = False


def _wrap_compile(state_graph_cls: Any) -> None:
    original: Any = state_graph_cls.compile
    if getattr(original, "_argus_pytest_wrapped", False):
        return
    _originals["compile"] = original

    @functools.wraps(original)
    def compile(self: Any, *args: Any, **kwargs: Any) -> Any:
        compiled = original(self, *args, **kwargs)
        if _in_attach():
            return compiled
        # ArgusWatcher already owns this builder (instance compile wrapper).
        if getattr(self, "_argus_compile_wrapped", False):
            return compiled
        if getattr(compiled, "_argus_auto_persist", False):
            return compiled
        return _attach(compiled)

    compile._argus_pytest_wrapped = True  # type: ignore[attr-defined]
    state_graph_cls.compile = compile


def _wrap_pregel_invoke() -> None:
    try:
        from langgraph.pregel import Pregel
    except ImportError:
        return

    original = Pregel.invoke
    if getattr(original, "_argus_pytest_wrapped", False):
        return
    _originals["invoke"] = original

    @functools.wraps(original)
    def invoke(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _in_attach() or getattr(self, "_argus_auto_persist", False):
            return original(self, *args, **kwargs)
        app = _attach(self)
        return app.invoke(*args, **kwargs)

    invoke._argus_pytest_wrapped = True  # type: ignore[attr-defined]
    Pregel.invoke = invoke  # type: ignore[method-assign]
