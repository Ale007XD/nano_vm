"""Tests for ProgramValidator — PV-01..12.

Run from nano-vm repo root:
    pytest tests/test_validator.py -v
"""

from __future__ import annotations

from nano_vm.models import Program, Step, StepType
from nano_vm.validator import IssueKind, ProgramValidator, ValidationReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm(sid: str, *, next_step: str | None = None, terminal: bool = False) -> Step:
    return Step(id=sid, type=StepType.LLM, prompt="x", next_step=next_step, is_terminal=terminal)


def _tool(sid: str, *, next_step: str | None = None, terminal: bool = False) -> Step:
    return Step(id=sid, type=StepType.TOOL, tool="t", next_step=next_step, is_terminal=terminal)


def _cond(sid: str, *, then: str | None = None, otherwise: str | None = None) -> Step:
    # at least one branch required by Step validator
    return Step(
        id=sid,
        type=StepType.CONDITION,
        condition="x == 1",
        then=then,
        otherwise=otherwise or then,  # fallback to keep model happy
    )


def _program(*steps: Step, name: str = "prog") -> Program:
    return Program(name=name, steps=list(steps))


def _validate(*steps: Step) -> ValidationReport:
    return ProgramValidator(_program(*steps)).validate()


# ---------------------------------------------------------------------------
# PV-01: valid linear program — no issues
# ---------------------------------------------------------------------------


def test_pv01_valid_linear() -> None:
    report = _validate(
        _llm("a"),
        _llm("b"),
        _tool("c", terminal=True),
    )
    assert report.is_valid()


# ---------------------------------------------------------------------------
# PV-02: valid branching program — no issues
# ---------------------------------------------------------------------------


def test_pv02_valid_branching() -> None:
    report = _validate(
        _llm("entry", next_step="check"),
        _cond("check", then="ok", otherwise="fail"),
        _tool("ok", terminal=True),
        _tool("fail", terminal=True),
    )
    assert report.is_valid()


# ---------------------------------------------------------------------------
# PV-03: missing next_step target
# ---------------------------------------------------------------------------


def test_pv03_missing_next_step() -> None:
    report = _validate(
        _llm("a", next_step="ghost"),
        _tool("b", terminal=True),
    )
    assert not report.is_valid()
    issues = report.by_kind(IssueKind.MISSING_TARGET)
    assert len(issues) == 1
    assert issues[0].step_id == "a"
    assert "ghost" in issues[0].detail


# ---------------------------------------------------------------------------
# PV-04: missing condition then/otherwise target
# ---------------------------------------------------------------------------


def test_pv04_missing_condition_target() -> None:
    report = _validate(
        _cond("check", then="missing_step"),
        _tool("ok", terminal=True),
    )
    issues = report.by_kind(IssueKind.MISSING_TARGET)
    # then="missing_step" not in steps; otherwise fallback = "missing_step" too
    assert any("missing_step" in i.detail for i in issues)


# ---------------------------------------------------------------------------
# PV-05: unreachable step — isolated terminal
# ---------------------------------------------------------------------------


def test_pv05_unreachable_isolated() -> None:
    report = _validate(
        _llm("a", next_step="b"),
        _tool("b", terminal=True),
        _tool("orphan", terminal=True),  # never reachable
    )
    issues = report.by_kind(IssueKind.UNREACHABLE_STEP)
    assert len(issues) == 1
    assert issues[0].step_id == "orphan"


# ---------------------------------------------------------------------------
# PV-06: unreachable step — branch bypass
# ---------------------------------------------------------------------------


def test_pv06_unreachable_branch_bypass() -> None:
    # entry → skip_me explicitly, so 'middle' never reached
    report = _validate(
        _llm("entry", next_step="terminal"),
        _llm("middle"),  # unreachable: entry skips it
        _tool("terminal", terminal=True),
    )
    issues = report.by_kind(IssueKind.UNREACHABLE_STEP)
    assert any(i.step_id == "middle" for i in issues)


# ---------------------------------------------------------------------------
# PV-07: direct self-loop cycle
# ---------------------------------------------------------------------------


def test_pv07_self_loop() -> None:
    report = _validate(
        _llm("a", next_step="a"),  # a → a
    )
    issues = report.by_kind(IssueKind.CYCLE_DETECTED)
    assert len(issues) >= 1
    assert issues[0].step_id == "a"


# ---------------------------------------------------------------------------
# PV-08: two-step cycle A → B → A
# ---------------------------------------------------------------------------


def test_pv08_two_step_cycle() -> None:
    report = _validate(
        _llm("a", next_step="b"),
        _llm("b", next_step="a"),
    )
    issues = report.by_kind(IssueKind.CYCLE_DETECTED)
    assert len(issues) >= 1
    cycle_detail = " ".join(i.detail for i in issues)
    assert "a" in cycle_detail and "b" in cycle_detail


# ---------------------------------------------------------------------------
# PV-09: condition cycle via then branch
# ---------------------------------------------------------------------------


def test_pv09_condition_cycle() -> None:
    report = _validate(
        _cond("gate", then="gate", otherwise="end"),  # gate → gate
        _tool("end", terminal=True),
    )
    issues = report.by_kind(IssueKind.CYCLE_DETECTED)
    assert len(issues) >= 1


# ---------------------------------------------------------------------------
# PV-10: terminal step — no outgoing edges, never causes cycle
# ---------------------------------------------------------------------------


def test_pv10_terminal_no_outgoing() -> None:
    # terminal with next_step set — next_step must be ignored in adjacency
    step_t = Step(
        id="t",
        type=StepType.TOOL,
        tool="x",
        is_terminal=True,
        next_step="a",  # should be ignored
    )
    report = _validate(_llm("a", next_step="t"), step_t)
    assert report.is_valid(), report.summary()


# ---------------------------------------------------------------------------
# PV-11: multiple issues reported together
# ---------------------------------------------------------------------------


def test_pv11_multiple_issues() -> None:
    report = _validate(
        _llm("a", next_step="missing"),  # missing target
        _tool("orphan", terminal=True),  # unreachable (a skips sequential)
    )
    assert not report.is_valid()
    kinds = {i.kind for i in report.issues}
    assert IssueKind.MISSING_TARGET in kinds
    assert IssueKind.UNREACHABLE_STEP in kinds


# ---------------------------------------------------------------------------
# PV-12: single-step program with terminal — valid
# ---------------------------------------------------------------------------


def test_pv12_single_step_valid() -> None:
    report = _validate(_tool("only", terminal=True))
    assert report.is_valid()
