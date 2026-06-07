"""Tests for PV-13: no_failure_terminal check."""
from __future__ import annotations

import pytest

from nano_vm.models import Program, Step, StepType
from nano_vm.validator import IssueKind, ProgramValidator


def _make_program(steps: list[Step]) -> Program:
    return Program(name="test_prog", steps=steps)


def _llm(sid: str, **kwargs: object) -> Step:
    return Step(id=sid, type=StepType.LLM, prompt="x", **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PV-13a: only SUCCESS terminal → warning issued
# ---------------------------------------------------------------------------

def test_pv13a_only_success_terminal_warns() -> None:
    prog = _make_program([
        _llm("start"),
        _llm("done", is_terminal=True),
    ])
    report = ProgramValidator(prog).validate()
    pv13 = report.by_kind(IssueKind.NO_FAILURE_TERMINAL)
    assert len(pv13) == 1
    assert "loop" in pv13[0].detail


# ---------------------------------------------------------------------------
# PV-13b: failure terminal by step id keyword → no warning
# ---------------------------------------------------------------------------

def test_pv13b_failure_id_no_warning() -> None:
    prog = _make_program([
        _llm("start"),
        _llm("handle_failed", is_terminal=True),
    ])
    report = ProgramValidator(prog).validate()
    assert report.by_kind(IssueKind.NO_FAILURE_TERMINAL) == []


# ---------------------------------------------------------------------------
# PV-13c: failure terminal via allowed_outputs → no warning
# ---------------------------------------------------------------------------

def test_pv13c_failure_allowed_outputs_no_warning() -> None:
    prog = _make_program([
        _llm("start"),
        _llm("terminal", is_terminal=True, allowed_outputs=["SUCCESS", "REJECTED"]),
    ])
    report = ProgramValidator(prog).validate()
    assert report.by_kind(IssueKind.NO_FAILURE_TERMINAL) == []


# ---------------------------------------------------------------------------
# PV-13d: failure terminal unreachable → warning still issued
# ---------------------------------------------------------------------------

def test_pv13d_unreachable_failure_terminal_warns() -> None:
    # "orphan_fail" has no path from entry — not reachable
    prog = _make_program([
        _llm("start", next_step="done"),
        _llm("done", is_terminal=True),
        _llm("orphan_fail", is_terminal=True),  # unreachable
    ])
    report = ProgramValidator(prog).validate()
    # unreachable_step issue present
    unreachable = report.by_kind(IssueKind.UNREACHABLE_STEP)
    assert any(i.step_id == "orphan_fail" for i in unreachable)
    # PV-13 also fires — unreachable terminal doesn't count
    assert len(report.by_kind(IssueKind.NO_FAILURE_TERMINAL)) == 1
