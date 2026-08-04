"""Document known false-positive-prone signatures in the bundled registry."""
import pytest

from argus.registry import get_registry, scan_value


@pytest.fixture
def registry():
    return get_registry()

@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

@pytest.mark.unit
class TestKnownFalsePositives:
    def test_cm008_base64_padding_false_positive(self, registry):
        """CM-008 matches strings ending with '==' after alphanumeric chars."""
        cm008 = [s for s in registry if s["id"] == "CM-008"]
        if cm008:
            matches = scan_value("if score == ", cm008)
            if matches:
                pytest.xfail("CM-008 false-positives on code comparison operators")

    def test_nl007_na_false_positive(self, registry):
        """NL-007 matches 'na' exact_ci — catches chemical symbol Na."""
        nl007 = [s for s in registry if s["id"] == "NL-007"]
        if nl007:
            matches = scan_value("Na", nl007)
            if matches:
                pytest.xfail("NL-007 false-positives on 'Na' (sodium)")

    def test_mp007_python_true_false_positive(self, registry):
        """MP-007 matches 'True'/'False'/'None' as raw JSON primitives."""
        mp007 = [s for s in registry if s["id"] == "MP-007"]
        if mp007:
            matches = scan_value("True", mp007)
            if matches:
                pytest.xfail("MP-007 false-positives on Python str(True)")

@pytest.mark.unit
class TestLegitimateOutputsClean:
    def test_normal_analysis_no_signals(self, registry):
        matches = scan_value(
            "Tesla stock rose 5.3% today on strong Q3 earnings, "
            "beating analyst expectations by $0.12 per share.",
            registry,
        )
        assert len(matches) == 0

    def test_code_snippet_no_critical(self, registry):
        code = 'const result = data.filter(x => x.value !== null && x.active === true);'
        matches = scan_value(code, registry)
        critical = [m for m in matches if m.severity == "critical"]
        assert len(critical) == 0
