"""Tests for ExecutionReceipt — ER-01..08.

Run from nano-vm repo root:
    pytest tests/test_receipt.py -v

These tests import TraceAnalyzer and ExecutionReceipt from nano_vm.analyzer.
"""
from __future__ import annotations

import hashlib
import sys
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load analyzer_orig.py as if it were nano_vm.analyzer
# (replace with `from nano_vm.analyzer import ...` when file is in repo)
# ---------------------------------------------------------------------------

def _load_analyzer() -> Any:
    path = Path(__file__).parent / "analyzer_orig.py"
    spec = importlib.util.spec_from_file_location("analyzer_mod", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["analyzer_mod"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

_mod = _load_analyzer()
TraceAnalyzer = _mod.TraceAnalyzer
ExecutionReceipt = _mod.ExecutionReceipt
TraceHealthReport = _mod.TraceHealthReport

from nano_vm.models import (
    Trace,
    TraceStatus,
    StepResult,
    StepStatus,
    LLMUsage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(step_id: str, status: StepStatus = StepStatus.SUCCESS, retries: int = 0) -> StepResult:
    return StepResult(
        step_id=step_id,
        status=status,
        output="ok",
        retries=retries,
    )


def _make_trace(
    steps: list[StepResult] | None = None,
    status: TraceStatus = TraceStatus.SUCCESS,
    snapshots: list[tuple[int, str]] | None = None,
) -> Trace:
    t = Trace(program_name="test_program", status=status)
    for s in (steps or []):
        t = t.add_step(s)
    for idx, fp in (snapshots or []):
        t = t.add_snapshot(idx, fp)
    return t.finish(status)


# ---------------------------------------------------------------------------
# ER-01: receipt() is deterministic — two calls return identical object
# ---------------------------------------------------------------------------

def test_er01_deterministic() -> None:
    trace = _make_trace([_step("a"), _step("b")])
    analyzer = TraceAnalyzer(trace)
    r1 = analyzer.receipt()
    r2 = analyzer.receipt()
    assert r1 is r2, "receipt() must return cached instance"
    assert r1 == r2


# ---------------------------------------------------------------------------
# ER-02: trace_hash is SHA-256 over canonical_snapshot_hash (Merkle root)
# ---------------------------------------------------------------------------

def test_er02_trace_hash() -> None:
    snapshots = [(0, "aabbcc"), (1, "ddeeff")]
    trace = _make_trace([_step("a"), _step("b")], snapshots=snapshots)
    merkle = trace.canonical_snapshot_hash()
    expected = hashlib.sha256(merkle.encode()).hexdigest()
    r = TraceAnalyzer(trace).receipt()
    assert r.trace_hash == expected
    assert len(r.trace_hash) == 64  # hex SHA-256


# ---------------------------------------------------------------------------
# ER-03: resumable=True when status == SUSPENDED
# ---------------------------------------------------------------------------

def test_er03_resumable_suspended() -> None:
    t = Trace(program_name="p")
    t = t.add_step(_step("a"))
    t = t.suspend("a")
    r = TraceAnalyzer(t).receipt()
    assert r.resumable is True
    assert r.final_status == TraceStatus.SUSPENDED


# ---------------------------------------------------------------------------
# ER-04: resumable=False when status == SUCCESS
# ---------------------------------------------------------------------------

def test_er04_not_resumable_success() -> None:
    trace = _make_trace([_step("a"), _step("b")])
    r = TraceAnalyzer(trace).receipt()
    assert r.resumable is False
    assert r.final_status == TraceStatus.SUCCESS


# ---------------------------------------------------------------------------
# ER-05: replayable=False when status == RUNNING
# ---------------------------------------------------------------------------

def test_er05_not_replayable_running() -> None:
    t = Trace(program_name="p", status=TraceStatus.RUNNING)
    t = t.add_step(_step("a"))
    r = TraceAnalyzer(t).receipt()
    assert r.replayable is False


# ---------------------------------------------------------------------------
# ER-06: replayable=True for terminal statuses (SUCCESS, FAILED, SUSPENDED)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", [
    TraceStatus.SUCCESS,
    TraceStatus.FAILED,
    TraceStatus.SUSPENDED,
    TraceStatus.BUDGET_EXCEEDED,
    TraceStatus.STALLED,
])
def test_er06_replayable_terminal(status: TraceStatus) -> None:
    t = Trace(program_name="p", status=status)
    t = t.add_step(_step("a"))
    r = TraceAnalyzer(t).receipt()
    assert r.replayable is True, f"Expected replayable for status={status}"


# ---------------------------------------------------------------------------
# ER-07: failed_steps count
# ---------------------------------------------------------------------------

def test_er07_failed_steps_count() -> None:
    steps = [
        _step("a", StepStatus.SUCCESS),
        _step("b", StepStatus.FAILED),
        _step("c", StepStatus.FAILED),
        _step("d", StepStatus.SUCCESS),
    ]
    trace = _make_trace(steps, status=TraceStatus.FAILED)
    r = TraceAnalyzer(trace).receipt()
    assert r.failed_steps == 2


# ---------------------------------------------------------------------------
# ER-08: retried_steps count
# ---------------------------------------------------------------------------

def test_er08_retried_steps_count() -> None:
    steps = [
        _step("a", retries=0),
        _step("b", retries=2),
        _step("c", retries=1),
        _step("d", retries=0),
    ]
    trace = _make_trace(steps)
    r = TraceAnalyzer(trace).receipt()
    assert r.retried_steps == 2


# ---------------------------------------------------------------------------
# ER-09: health is TraceHealthReport embedded
# ---------------------------------------------------------------------------

def test_er09_health_embedded() -> None:
    trace = _make_trace([_step("a"), _step("b")])
    r = TraceAnalyzer(trace).receipt()
    assert isinstance(r.health, TraceHealthReport)
    assert r.health.trace_id == trace.trace_id


# ---------------------------------------------------------------------------
# ER-10: two independent TraceAnalyzer instances same trace → equal receipts
# ---------------------------------------------------------------------------

def test_er10_cross_instance_determinism() -> None:
    trace = _make_trace([_step("x"), _step("y"), _step("z")])
    r1 = TraceAnalyzer(trace).receipt()
    r2 = TraceAnalyzer(trace).receipt()
    assert r1.trace_hash == r2.trace_hash
    assert r1.trace_id == r2.trace_id
    assert r1.final_status == r2.final_status
    assert r1.failed_steps == r2.failed_steps
    assert r1.retried_steps == r2.retried_steps
