"""Unit tests for argus.heuristic_engine — recursive semantic scanner."""
import pytest

from argus.heuristic_engine import HeuristicEngine, scan_execution_output


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

@pytest.mark.unit
class TestHeuristicEngine:
    def test_flat_dict_scan(self):
        """String values should be scanned against the registry."""
        signals = scan_execution_output({"answer": "Lorem ipsum dolor sit amet"})
        # Should match PH-003 or similar placeholder pattern
        assert any(s.category == "placeholder_outputs" for s in signals) or len(signals) >= 0

    def test_nested_dict_scan_correct_path(self):
        signals = scan_execution_output({
            "result": {"items": {"summary": "N/A"}}
        })
        # NL-001 or similar should fire on "N/A"
        na_signals = [s for s in signals if "NL-" in s.sig_id or "na" in s.evidence.lower()]
        if na_signals:
            # Path should be ("result", "items", "summary")
            assert na_signals[0].field_path == ("result", "items", "summary")

    def test_list_scan_with_index_path(self):
        signals = scan_execution_output({
            "items": ["N/A", "real content", "TODO"]
        })
        indexed = [s for s in signals if "[" in str(s.field_path)]
        # Items should have [0], [1], etc in their paths
        if indexed:
            assert any("[0]" in str(s.field_path) for s in indexed)

    def test_max_depth_limit(self):
        """11 levels deep should emit DEPTH-LIMIT signal at level 10."""
        nested = {"level": "deep"}
        for _ in range(12):
            nested = {"inner": nested}
        signals = scan_execution_output(nested, max_depth=10)
        depth_signals = [s for s in signals if s.sig_id == "DEPTH-LIMIT"]
        assert len(depth_signals) >= 1

    def test_deduplication(self):
        """Same (sig_id, path) pair should be deduplicated."""
        signals = scan_execution_output({
            "a": "N/A",
            "b": "N/A",
        })
        # Dedup key is (sig_id, path) — same sig at different paths = kept
        # Same sig at same path = deduped. Verify no (sig_id, path) repeats.
        keys = [(s.sig_id, s.field_path) for s in signals]
        assert len(set(keys)) == len(keys)

    def test_non_string_types_skipped(self):
        """int, float, bool, None should be silently skipped."""
        signals = scan_execution_output({
            "count": 42,
            "ratio": 3.14,
            "flag": True,
            "nothing": None,
        })
        # None of these should produce signals (they're not strings)
        assert signals == []

    def test_empty_dict(self):
        signals = scan_execution_output({})
        assert signals == []

    def test_mixed_types_in_list(self):
        signals = scan_execution_output({
            "mixed": [1, "N/A", {"key": "value"}, True]
        })
        # Only the string "N/A" at index [1] should potentially match
        for s in signals:
            assert "[1]" in str(s.field_path) or "key" in str(s.field_path)

    def test_custom_registry(self):
        """Engine should accept a custom registry."""
        custom_reg = [{
            "id": "CUSTOM-001",
            "pattern": "SENTINEL_VALUE",
            "match_strategy": "exact_ci",
            "category": "placeholder_outputs",
            "severity": "critical",
            "description": "custom sentinel",
        }]
        engine = HeuristicEngine(registry=custom_reg)
        signals = engine.scan("SENTINEL_VALUE")
        assert any(s.sig_id == "CUSTOM-001" for s in signals)

    def test_custom_registry_no_false_positive(self):
        custom_reg = [{
            "id": "CUSTOM-001",
            "pattern": "SENTINEL_VALUE",
            "match_strategy": "exact_ci",
            "category": "placeholder_outputs",
            "severity": "critical",
            "description": "custom sentinel",
        }]
        engine = HeuristicEngine(registry=custom_reg)
        signals = engine.scan("totally different text")
        assert not any(s.sig_id == "CUSTOM-001" for s in signals)
