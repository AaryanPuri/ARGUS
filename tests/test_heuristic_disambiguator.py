"""Unit tests for argus.heuristic_disambiguator — LLM disambiguation (all calls mocked)."""
import json

import pytest

from argus.heuristic_disambiguator import disambiguate_signals
from argus.models import SemanticSignal


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

def _make_signal(sig_id="PH-001", confidence=0.5):
    return SemanticSignal(
        sig_id=sig_id, category="placeholder_outputs", severity="warning",
        description="test placeholder", field_path=("result",),
        evidence="some evidence", confidence=confidence,
    )

def _mock_disambiguator(monkeypatch, verdicts, is_available=True):
    monkeypatch.setattr("argus.llm_proxy.is_available", lambda: is_available)
    monkeypatch.setattr("argus.llm_proxy.create_chat_completion", lambda **kw: {
        "choices": [{"message": {"content": json.dumps({"verdicts": verdicts})}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    })

@pytest.mark.unit
class TestDisambiguation:
    def test_empty_signals_returns_empty(self, monkeypatch):
        _mock_disambiguator(monkeypatch, [])
        result = disambiguate_signals("node", {}, {}, [])
        assert result == []

    def test_false_positive_verdict(self, monkeypatch):
        signal = _make_signal("PH-001", 0.5)
        _mock_disambiguator(monkeypatch, [
            {"sig_id": "PH-001", "is_failure": False, "confidence": 0.9, "reason": "legit"}
        ])
        results = disambiguate_signals("node", {"q": "x"}, {"a": "y"}, [signal])
        assert len(results) == 1
        assert results[0].llm_verdict is False  # false positive
        assert results[0].sig_id == "PH-001"

    def test_true_positive_verdict(self, monkeypatch):
        signal = _make_signal("PH-001", 0.5)
        _mock_disambiguator(monkeypatch, [
            {"sig_id": "PH-001", "is_failure": True, "confidence": 0.8, "reason": "real issue"}
        ])
        results = disambiguate_signals("node", {"q": "x"}, {"a": "y"}, [signal])
        assert len(results) == 1
        assert results[0].llm_verdict is True

    def test_multiple_signals_batched(self, monkeypatch):
        signals = [_make_signal("PH-001", 0.5), _make_signal("CM-003", 0.4)]
        _mock_disambiguator(monkeypatch, [
            {"sig_id": "PH-001", "is_failure": False, "confidence": 0.9, "reason": "ok"},
            {"sig_id": "CM-003", "is_failure": True, "confidence": 0.7, "reason": "real"},
        ])
        results = disambiguate_signals("node", {"q": "x"}, {"a": "y"}, signals)
        assert len(results) == 2

    def test_unknown_sig_id_ignored(self, monkeypatch):
        signal = _make_signal("PH-001", 0.5)
        _mock_disambiguator(monkeypatch, [
            {"sig_id": "UNKNOWN-999", "is_failure": True, "confidence": 0.9, "reason": "?"}
        ])
        results = disambiguate_signals("node", {"q": "x"}, {"a": "y"}, [signal])
        assert len(results) == 0  # unknown sig_id filtered out

    def test_not_available_returns_empty(self, monkeypatch):
        signal = _make_signal("PH-001", 0.5)
        _mock_disambiguator(monkeypatch, [], is_available=False)
        results = disambiguate_signals("node", {"q": "x"}, {"a": "y"}, [signal])
        assert results == []

    def test_api_error_returns_empty(self, monkeypatch):
        signal = _make_signal("PH-001", 0.5)
        monkeypatch.setattr("argus.llm_proxy.is_available", lambda: True)
        def _raise(**kw):
            raise ConnectionError("timeout")
        monkeypatch.setattr("argus.llm_proxy.create_chat_completion", _raise)
        results = disambiguate_signals("node", {"q": "x"}, {"a": "y"}, [signal])
        assert results == []
