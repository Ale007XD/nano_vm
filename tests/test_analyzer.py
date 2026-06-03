"""tests/test_analyzer.py — TraceAnalyzer test suite (Sprint 6).

Test IDs: TA-01 .. TA-22
Coverage:
  TA-01..04  rollback_density
  TA-05..08  tool_churn_rate
  TA-09..13  path_variance
  TA-14..17  invariant_violation_rate
  TA-18..20  alert threshold triggers
  TA-21      empty trace edge case
  TA-22      analyze_batch helper
"""

from __future__ import annotations

import hashlib
import os

# Import from the local sprint6 module (copy analyzer.py next to tests or adjust sys.path)
import sys
from typing import Any

import pytest

from nano_vm.models import StepResult, StepStatus, Trace, TraceStatus

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from nano_vm.analyzer import TraceAnalyzer, TraceHealthReport, analyze_batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step(
    step_id: str,
    status: StepStatus = StepStatus.SUCCESS,
    retries: int = 0,
    output: Any = None,
    error: str | None = None,
) -> StepResult:
    return StepResult(
        step_id=step_id,
        status=status,
        output=output,
        error=error,
        retries=retries,
    )


def _snap(idx: int, data: str = "data") -> tuple[int, str]:
    fp = hashlib.sha256(f"{idx}:{data}".encode()).hexdigest()
    return (idx, fp)


def _trace(
    steps: list[StepResult] | None = None,
    snapshots: list[tuple[int, str]] | None = None,
    program_name: str = "test_program",
    status: TraceStatus = TraceStatus.SUCCESS,
) -> Trace:
    t = Trace(program_name=program_name, status=status)
    for s in steps or []:
        t = t.add_step(s)
    for idx, fp in snapshots or []:
        t = t.add_snapshot(idx, fp)
    return t


# ---------------------------------------------------------------------------
# TA-01..04  rollback_density
# ---------------------------------------------------------------------------


def test_ta01_rollback_density_zero() -> None:
    """TA-01: no retries → 0.0"""
    trace = _trace([_step("a"), _step("b"), _step("c")])
    assert TraceAnalyzer(trace).rollback_density() == 0.0


def test_ta02_rollback_density_all() -> None:
    """TA-02: all steps retried → 1.0"""
    trace = _trace([_step("a", retries=1), _step("b", retries=2), _step("c", retries=3)])
    assert TraceAnalyzer(trace).rollback_density() == 1.0


def test_ta03_rollback_density_partial() -> None:
    """TA-03: 1 of 4 steps retried → 0.25"""
    trace = _trace([_step("a"), _step("b", retries=1), _step("c"), _step("d")])
    assert TraceAnalyzer(trace).rollback_density() == pytest.approx(0.25)


def test_ta04_rollback_density_empty() -> None:
    """TA-04: empty trace → 0.0"""
    trace = _trace([])
    assert TraceAnalyzer(trace).rollback_density() == 0.0


# ---------------------------------------------------------------------------
# TA-05..08  tool_churn_rate
# ---------------------------------------------------------------------------


def test_ta05_tool_churn_zero() -> None:
    """TA-05: all unique step_ids → 0.0"""
    trace = _trace([_step("a"), _step("b"), _step("c")])
    assert TraceAnalyzer(trace).tool_churn_rate() == 0.0


def test_ta06_tool_churn_one_duplicate() -> None:
    """TA-06: step 'a' appears twice in 4 steps → 1/4 = 0.25"""
    trace = _trace([_step("a"), _step("b"), _step("a"), _step("c")])
    assert TraceAnalyzer(trace).tool_churn_rate() == pytest.approx(0.25)


def test_ta07_tool_churn_all_same() -> None:
    """TA-07: same step_id repeated 4 times → 3/4 = 0.75"""
    trace = _trace([_step("x"), _step("x"), _step("x"), _step("x")])
    assert TraceAnalyzer(trace).tool_churn_rate() == pytest.approx(0.75)


def test_ta08_tool_churn_empty() -> None:
    """TA-08: empty trace → 0.0"""
    trace = _trace([])
    assert TraceAnalyzer(trace).tool_churn_rate() == 0.0


# ---------------------------------------------------------------------------
# TA-09..13  path_variance
# ---------------------------------------------------------------------------


def test_ta09_path_variance_none_without_baseline() -> None:
    """TA-09: no baseline → path_variance returns None"""
    trace = _trace(snapshots=[_snap(0), _snap(1)])
    assert TraceAnalyzer(trace).path_variance() is None


def test_ta10_path_variance_identical() -> None:
    """TA-10: identical snapshots → 0.0"""
    snaps = [_snap(0), _snap(1), _snap(2)]
    trace = _trace(snapshots=snaps)
    baseline = _trace(snapshots=snaps)
    assert TraceAnalyzer(trace, baseline=baseline).path_variance() == pytest.approx(0.0)


def test_ta11_path_variance_all_different() -> None:
    """TA-11: completely different snapshots → 1.0"""
    snaps_a = [(0, "aaa"), (1, "bbb")]
    snaps_b = [(0, "xxx"), (1, "yyy")]
    trace = _trace(snapshots=snaps_a)
    baseline = _trace(snapshots=snaps_b)
    assert TraceAnalyzer(trace, baseline=baseline).path_variance() == pytest.approx(1.0)


def test_ta12_path_variance_partial() -> None:
    """TA-12: 1 of 2 snapshots differs → 0.5"""
    shared_fp = hashlib.sha256(b"shared").hexdigest()
    diff_fp_a = hashlib.sha256(b"trace_val").hexdigest()
    diff_fp_b = hashlib.sha256(b"base_val").hexdigest()
    snaps_a = [(0, shared_fp), (1, diff_fp_a)]
    snaps_b = [(0, shared_fp), (1, diff_fp_b)]
    trace = _trace(snapshots=snaps_a)
    baseline = _trace(snapshots=snaps_b)
    assert TraceAnalyzer(trace, baseline=baseline).path_variance() == pytest.approx(0.5)


def test_ta13_path_variance_empty_snapshots() -> None:
    """TA-13: both empty → 0.0"""
    trace = _trace(snapshots=[])
    baseline = _trace(snapshots=[])
    assert TraceAnalyzer(trace, baseline=baseline).path_variance() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TA-14..17  invariant_violation_rate
# ---------------------------------------------------------------------------


def test_ta14_invariant_violation_zero() -> None:
    """TA-14: no failures → 0.0"""
    trace = _trace([_step("a"), _step("b"), _step("c")])
    assert TraceAnalyzer(trace).invariant_violation_rate() == 0.0


def test_ta15_invariant_violation_all() -> None:
    """TA-15: all steps failed → 1.0"""
    trace = _trace(
        [
            _step("a", status=StepStatus.FAILED, error="err"),
            _step("b", status=StepStatus.FAILED, error="err"),
        ]
    )
    assert TraceAnalyzer(trace).invariant_violation_rate() == 1.0


def test_ta16_invariant_violation_partial() -> None:
    """TA-16: 1 of 3 failed → 1/3"""
    trace = _trace(
        [
            _step("a"),
            _step("b", status=StepStatus.FAILED, error="err"),
            _step("c"),
        ]
    )
    assert TraceAnalyzer(trace).invariant_violation_rate() == pytest.approx(1 / 3)


def test_ta17_invariant_violation_empty() -> None:
    """TA-17: empty trace → 0.0"""
    trace = _trace([])
    assert TraceAnalyzer(trace).invariant_violation_rate() == 0.0


# ---------------------------------------------------------------------------
# TA-18..20  alert threshold triggers
# ---------------------------------------------------------------------------


def test_ta18_alert_rollback_density() -> None:
    """TA-18: rollback_density > 0.3 → alert fired"""
    # 2 of 3 steps retried → 0.667
    trace = _trace([_step("a", retries=1), _step("b", retries=1), _step("c")])
    report = TraceAnalyzer(trace).report()
    assert any("rollback_density" in a for a in report.alerts)
    assert not report.is_healthy()


def test_ta19_alert_tool_churn() -> None:
    """TA-19: tool_churn_rate > 0.4 → alert fired"""
    # steps: a, a, a, b → churn = 2/4 = 0.5
    trace = _trace([_step("a"), _step("a"), _step("a"), _step("b")])
    report = TraceAnalyzer(trace).report()
    assert any("tool_churn_rate" in a for a in report.alerts)


def test_ta20_no_alert_healthy() -> None:
    """TA-20: healthy trace → no alerts, is_healthy() = True"""
    trace = _trace([_step("validate"), _step("reserve"), _step("capture"), _step("receipt")])
    report = TraceAnalyzer(trace).report()
    assert report.alerts == []
    assert report.is_healthy()


# ---------------------------------------------------------------------------
# TA-21  empty trace edge case
# ---------------------------------------------------------------------------


def test_ta21_empty_trace_report() -> None:
    """TA-21: completely empty trace → all metrics 0.0, no alerts"""
    trace = _trace([])
    report = TraceAnalyzer(trace).report()
    assert report.total_steps == 0
    assert report.rollback_density == 0.0
    assert report.tool_churn_rate == 0.0
    assert report.invariant_violation_rate == 0.0
    assert report.path_variance is None
    assert report.alerts == []


# ---------------------------------------------------------------------------
# TA-22  analyze_batch
# ---------------------------------------------------------------------------


def test_ta22_analyze_batch() -> None:
    """TA-22: analyze_batch returns one report per trace"""
    t1 = _trace([_step("a"), _step("b")], program_name="prog_a")
    t2 = _trace([_step("x", retries=2), _step("y"), _step("x")], program_name="prog_b")
    reports = analyze_batch([t1, t2])
    assert len(reports) == 2
    assert all(isinstance(r, TraceHealthReport) for r in reports)
    assert reports[0].program_name == "prog_a"
    assert reports[1].program_name == "prog_b"
    # t2 has churn: 'x' appears twice → 1/3
    assert reports[1].tool_churn_rate == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# TA-23..29  transition_sequence_variance
# ---------------------------------------------------------------------------


def test_ta23_tsv_none_without_baseline() -> None:
    """TA-23: no baseline → transition_sequence_variance returns None"""
    trace = _trace([_step("a"), _step("b"), _step("c")])
    assert TraceAnalyzer(trace).transition_sequence_variance() is None


def test_ta24_tsv_identical_paths() -> None:
    """TA-24: identical step sequences → 0.0"""
    steps = [_step("a"), _step("b"), _step("c")]
    trace = _trace(steps)
    baseline = _trace(steps)
    assert TraceAnalyzer(trace, baseline=baseline).transition_sequence_variance() == pytest.approx(
        0.0
    )


def test_ta25_tsv_completely_different_paths() -> None:
    """TA-25: A→B→C vs A→D→C — (A,B),(B,C) vs (A,D),(D,C) → 4 divergent / 4 total = 1.0"""
    trace = _trace([_step("a"), _step("b"), _step("c")])
    baseline = _trace([_step("a"), _step("d"), _step("c")])
    result = TraceAnalyzer(trace, baseline=baseline).transition_sequence_variance()
    # symmetric_difference: {(a,b),(b,c),(a,d),(d,c)} / union {(a,b),(b,c),(a,d),(d,c)} = 4/4
    assert result == pytest.approx(1.0)


def test_ta26_tsv_partial_divergence() -> None:
    """TA-26: A→B→C→D vs A→B→E→D — one pair differs"""
    # trace pairs: (a,b),(b,c),(c,d)
    # baseline pairs: (a,b),(b,e),(e,d)
    # union = 5 pairs, symmetric_diff = {(b,c),(c,d),(b,e),(e,d)} = 4
    trace = _trace([_step("a"), _step("b"), _step("c"), _step("d")])
    baseline = _trace([_step("a"), _step("b"), _step("e"), _step("d")])
    result = TraceAnalyzer(trace, baseline=baseline).transition_sequence_variance()
    assert result == pytest.approx(4 / 5)


def test_ta27_tsv_single_step_trace() -> None:
    """TA-27: single step → no pairs → 0.0"""
    trace = _trace([_step("a")])
    baseline = _trace([_step("a")])
    assert TraceAnalyzer(trace, baseline=baseline).transition_sequence_variance() == pytest.approx(
        0.0
    )


def test_ta28_tsv_empty_traces() -> None:
    """TA-28: both empty → 0.0"""
    trace = _trace([])
    baseline = _trace([])
    assert TraceAnalyzer(trace, baseline=baseline).transition_sequence_variance() == pytest.approx(
        0.0
    )


def test_ta29_tsv_alert_fires() -> None:
    """TA-29: tsv > 0.4 → alert in report"""
    # A→B→C vs A→D→E: pairs {(a,b),(b,c)} vs {(a,d),(d,e)} → 4/4 = 1.0
    trace = _trace([_step("a"), _step("b"), _step("c")])
    baseline = _trace([_step("a"), _step("d"), _step("e")])
    report = TraceAnalyzer(trace, baseline=baseline).report()
    assert any("transition_sequence_variance" in a for a in report.alerts)
