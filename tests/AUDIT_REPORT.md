# ARGUS Detection Logic — Audit Report

**Date**: 2026-07-29 (updated 2026-07-30)
**Scope**: Detection pipeline only (no replay, CLI, UI, or storage)
**Method**: 210 new tests across 10 test files; zero source code changes
**Overall**: 195 passed, 3 xfailed (documented bugs), 2 xfailed (performance), 2 xfailed (false positives)

---

## Executive Summary

ARGUS detection logic is **functionally solid** for normal-sized outputs. All 16 inspector rules, 8 anomaly checks, the heuristic engine, correlator, disambiguator, and semantic checker produce correct results for their intended inputs. However, the audit uncovered **1 confirmed code bug**, **2 critical performance issues**, **2 false-positive-prone signatures**, and **1 dead code instance**.

**Coverage before this audit**: ~0% on detection modules (tests existed only in test_smoke.py covering session-level behavior)
**Coverage after**: 89% across 7 detection modules (1505 statements, 1343 covered)

---

## Bugs Found

### BUG-1: `tf.message` AttributeError in semantic_checker.py (CRITICAL)

- **File**: `src/argus/semantic_checker.py:154`
- **Test**: `test_semantic_checker_unit.py::TestSemanticCheckerBug::test_tool_failure_message_attribute_error`
- **Description**: Line 154 accesses `tf.message` but `ToolFailure` dataclass only has `tf.evidence`. This raises `AttributeError`.
- **Impact**: When a node has both tool failures AND the LLM semantic judge is enabled, the `AttributeError` propagates up uncaught because line 154 is **outside** the `try/except` block (which starts at line 165). Tool failure evidence is never forwarded to the LLM judge.
- **Fix**: Change `tf.message` to `tf.evidence` on line 154.
- **Severity**: CRITICAL — breaks the evidence pipeline silently when LLM judge is active.

### BUG-2: 1MB output takes 17+ minutes to inspect (CRITICAL PERF)

- **File**: `src/argus/inspector.py` (Rule 7 → heuristic scan)
- **Test**: `test_detection_stress.py::TestLargeOutputs::test_1mb_dict_completes`
- **Description**: `inspect_tool_outputs()` on a dict with 1000 fields (each ~1KB) took **1051 seconds**. The heuristic engine scans every string value against the full signature registry (55 signatures, 6 match strategies including regex and repetition detection). With 1000 fields this becomes O(fields * signatures * value_length).
- **Impact**: Any LangGraph node returning a large output dict will block the entire pipeline for minutes. This is a production-blocking issue for data-heavy pipelines.
- **Fix**: Add a configurable max-fields limit and/or a max-value-length cutoff for heuristic scanning. Skip heuristic scan for values that are clearly structured data (lists of numbers, etc.).

### BUG-3: Rule 14 SequenceMatcher O(n^2) on large strings (HIGH PERF)

- **File**: `src/argus/inspector.py` (Rule 14, input echo detection)
- **Test**: `test_detection_stress.py::TestLargeStrings::test_input_echo_10kb`
- **Description**: `SequenceMatcher` from `difflib` is O(n^2). Two 10KB strings take **7.3 seconds**. Two 50KB strings would take ~3 minutes. There is no size cap before calling SequenceMatcher.
- **Impact**: Any node that processes long documents (RAG, summarization) will trigger slow detection. The `_is_truncated` guard only helps for strings < 200 chars.
- **Fix**: Add a `max(len(in_val), len(out_val)) < 5000` guard before calling SequenceMatcher. For larger strings, use a cheaper hash-based similarity check.

---

## Known Limitations (documented via xfail)

### Inconsistent key casing in Rules 1 vs 2b/2c

- **Rule 1**: Uses exact key matching (`key in _ERROR_KEYS` where `_ERROR_KEYS` has lowercase "error")
- **Rule 2b/2c**: Uses `key.lower() in _SUCCESS_KEYS` / `key.lower() in _FAILURE_KEYS`
- **Impact**: A key like `"Error"` (capitalized) bypasses Rule 1 but `"Success"` (capitalized) IS caught by Rule 2b. Inconsistent behavior.
- **Test**: `test_inspector_unit.py::TestRule1ErrorKeys::test_capitalized_error_key_not_caught` (passes, documenting the gap)

### Rule 5 falsy error values not counted

- `{"error": 0}` or `{"error": False}` are falsy and skipped by `if value:` check
- **Test**: `test_inspector_unit.py::TestRule5PartialFailures::test_falsy_error_not_counted`

### Rule 10 string confidence not caught

- `{"confidence": "0.95"}` (string) skips the `isinstance(value, (int, float))` check
- **Test**: `test_inspector_unit.py::TestRule10ConfidenceMismatch::test_string_confidence_not_caught`

### Rule 4 mid-string errors not caught

- Regex only matches error patterns at start of string. `"got an error from API"` is not caught.
- **Test**: `test_inspector_unit.py::TestRule4ErrorStrings::test_mid_string_error_not_caught`

### Dead code: `_EMPTY_VALUES` tuple

- `inspector.py` defines `_EMPTY_VALUES = (None, "", [], {})` but uses `_is_empty()` function everywhere. The tuple is never referenced.

---

## False Positive Analysis

### NL-007: `"na"` exact_ci match

- **Trigger**: Any string that is exactly "Na" (e.g., sodium symbol, abbreviations)
- **Status**: xfail in `test_signatures_false_positives.py`
- **Recommendation**: Change to `contains_ci` with longer pattern like `"n/a"` or add min-length guard

### MP-007: Raw JSON primitive detection

- **Trigger**: Python `str(True)` = `"True"`, `str(None)` = `"None"`
- **Status**: xfail in `test_signatures_false_positives.py`
- **Recommendation**: Only match when the entire field value is the primitive, AND the field is expected to contain substantive content

### CM-008: Base64 padding pattern

- **Trigger**: Code comparisons like `"score == "` end with `==` after alphanumeric chars
- **Status**: Test passes (CM-008 may have been fixed or pattern tightened since initial analysis)

---

## Coverage by Module

| Module | Stmts | Covered | Coverage |
|--------|-------|---------|----------|
| `anomaly_detector.py` | 256 | 243 | **95%** |
| `correlator.py` | 373 | 326 | **87%** |
| `heuristic_disambiguator.py` | 57 | 53 | **93%** |
| `heuristic_engine.py` | 44 | 44 | **100%** |
| `inspector.py` | 504 | 442 | **88%** |
| `registry.py` | 201 | 166 | **83%** |
| `semantic_checker.py` | 70 | 69 | **99%** |
| **TOTAL** | **1505** | **1343** | **89%** |

### Uncovered areas (by module)

- **inspector.py** (88%): Some edge cases in `_check_fields()` with `node_provided_keys` sibling patterns, root cause chain parallel fan-out deduplication
- **registry.py** (83%): Shared signature cache loading from Supabase, `semantic_similarity` match strategy (requires embedding store)
- **correlator.py** (87%): `compare_replay()` function, some propagation chain assembly edge cases

---

## Test Files Created

| File | Tests | What it covers |
|------|-------|----------------|
| `conftest.py` | — | Shared factories: `make_event`, `make_inspection`, `make_run_record` |
| `test_inspector_unit.py` | 74 | All 16 inspector rules, structural inspection, root cause chain, typo detection, TypedDict introspection |
| `test_anomaly_detector.py` | 34 | All 8 anomaly checks (BA-001 through BA-008), behavior inference, 3-tier resolution |
| `test_semantic_checker_unit.py` | 16 | LLM judge pass/fail/skip/malformed/evidence/bug/truncation |
| `test_heuristic_engine_unit.py` | 11 | Recursive scan, depth limits, dedup, custom registry |
| `test_heuristic_disambiguator.py` | 7 | FP/TP verdicts, batching, error handling |
| `test_correlator_unit.py` | 12 | Origins, propagation, timeline, signal weights, anomaly cascade detection |
| `test_registry_unit.py` | 21 | All 5 match strategies, short-circuit, bundled registry, custom signatures |
| `test_signatures_false_positives.py` | 5 | Known FP signatures, legitimate output validation |
| `test_detection_integration.py` | 19 | Full pipeline via ArgusSession: clean run, tool failures, validators, crashes, async, concurrency, degraded input, latency signals, LLM judge override |
| `test_detection_stress.py` | 12 | Large outputs, deep nesting, unicode, edge cases, rapid runs |
| **TOTAL** | **211** | |

---

## Recommendations (Priority Order)

1. **Fix `tf.message` → `tf.evidence`** in `semantic_checker.py:154`. One-character fix, critical impact.
2. **Add size guards to heuristic scanning**. Cap at ~100 fields or ~50KB total to prevent 17-minute scans.
3. **Add size guard to Rule 14** (input echo). Skip SequenceMatcher for strings > 5KB. Use hash-based similarity for large strings.
4. **Normalize key casing in Rule 1**. Use `key.lower() in _ERROR_KEYS` for consistency with Rules 2b/2c.
5. **Tighten NL-007 and MP-007 signatures** to reduce false positives on chemical symbols and Python primitives.
6. **Remove dead `_EMPTY_VALUES` tuple** from inspector.py.

---

*Generated by running 211 tests against ARGUS v0.8.10 detection pipeline. Source code: zero changes.*
