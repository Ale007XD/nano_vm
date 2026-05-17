"""
tests/test_v075.py
==================
sprint_v075: BUG-FSM-OTHERWISE-CHAIN fix + BUG-ASTENGINE-NO-METHOD-CALLS fix.

OC-01  otherwise→condition (3 levels): c1(F)→c2(F)→c3(T)→leaf_ok
OC-02  otherwise→condition→otherwise→tool: c1(F)→c2(F)→leaf_dead / c2(T)→leaf_ok
OC-03  otherwise→condition with next_step continuation
OC-04  otherwise chain symmetric: then-chain == otherwise-chain in length/execution
OC-05  mixed: then→condition→otherwise→tool
OC-06  regression: CB-04 then-chain still works (non-regression)
OC-07  regression: CB-05 2-level otherwise still works (non-regression)

MC-01  .lower() in condition → ASTEvalError (not silent False)
MC-02  .strip() in condition → ASTEvalError
MC-03  .upper() in condition → ASTEvalError
MC-04  .split() in condition → ASTEvalError
MC-05  valid operators still work after fix
MC-06  regression: dotted path $x.output.field still works
MC-07  vm._execute_condition: .lower() → VMError propagated
"""

from __future__ import annotations

import pytest

from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.adapters import MockLLMAdapter
from nano_vm.ast_engine import ASTEvalError, eval_condition
from nano_vm.models import StateContext, Step, StepType
from nano_vm.vm import VMError


def make_vm(tools=None):
    return ExecutionVM(llm=MockLLMAdapter("ok"), tools=tools or {})


ECHO = lambda: "done"  # noqa: E731


# ------------------------------------------------------------------ #
# OC: Otherwise-Chain tests                                           #
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_oc01_otherwise_chain_3level():
    """c1(F)→otherwise=c2, c2(F)→otherwise=c3, c3(T)→then=leaf_ok."""
    vm = make_vm({"echo": ECHO})
    prog = Program.from_dict(
        {
            "name": "oc01",
            "steps": [
                {
                    "id": "c1",
                    "type": "condition",
                    "condition": "'a'=='b'",
                    "then": "leaf_dead",
                    "otherwise": "c2",
                },
                {
                    "id": "c2",
                    "type": "condition",
                    "condition": "'a'=='b'",
                    "then": "leaf_dead",
                    "otherwise": "c3",
                },
                {
                    "id": "c3",
                    "type": "condition",
                    "condition": "'a'=='a'",
                    "then": "leaf_ok",
                    "otherwise": "leaf_dead",
                },
                {"id": "leaf_ok", "type": "tool", "tool": "echo", "is_terminal": True},
                {"id": "leaf_dead", "type": "tool", "tool": "echo", "is_terminal": True},
            ],
        }
    )
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS, f"status={trace.status} err={trace.error}"
    assert ids == ["c1", "c2", "c3", "leaf_ok"], f"ids={ids}"


@pytest.mark.asyncio
async def test_oc02_otherwise_chain_varied_leafs():
    """c1(F)→c2, c2(T)→leaf_ok; c2(F)→leaf_dead — two paths."""
    vm = make_vm({"echo": ECHO})

    # Path 1: c1=False → c2=True → leaf_ok
    prog_a = Program.from_dict(
        {
            "name": "oc02a",
            "steps": [
                {
                    "id": "c1",
                    "type": "condition",
                    "condition": "'a'=='b'",
                    "then": "leaf_dead",
                    "otherwise": "c2",
                },
                {
                    "id": "c2",
                    "type": "condition",
                    "condition": "'a'=='a'",
                    "then": "leaf_ok",
                    "otherwise": "leaf_dead",
                },
                {"id": "leaf_ok", "type": "tool", "tool": "echo", "is_terminal": True},
                {"id": "leaf_dead", "type": "tool", "tool": "echo", "is_terminal": True},
            ],
        }
    )
    trace = await vm.run(prog_a)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["c1", "c2", "leaf_ok"], f"path A ids={ids}"

    # Path 2: c1=False → c2=False → leaf_dead
    prog_b = Program.from_dict(
        {
            "name": "oc02b",
            "steps": [
                {
                    "id": "c1",
                    "type": "condition",
                    "condition": "'a'=='b'",
                    "then": "leaf_ok",
                    "otherwise": "c2",
                },
                {
                    "id": "c2",
                    "type": "condition",
                    "condition": "'a'=='b'",
                    "then": "leaf_ok",
                    "otherwise": "leaf_dead",
                },
                {"id": "leaf_ok", "type": "tool", "tool": "echo", "is_terminal": True},
                {"id": "leaf_dead", "type": "tool", "tool": "echo", "is_terminal": True},
            ],
        }
    )
    trace = await vm.run(prog_b)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["c1", "c2", "leaf_dead"], f"path B ids={ids}"


@pytest.mark.asyncio
async def test_oc03_otherwise_condition_with_next_step():
    """c1(F)→otherwise=c2 → c2(T)→then=step_a(next_step=step_b) → step_b linear."""
    order: list[str] = []
    vm = ExecutionVM(
        llm=MockLLMAdapter("ok"),
        tools={
            "a": lambda: order.append("a") or "a",
            "b": lambda: order.append("b") or "b",
        },
    )
    prog = Program.from_dict(
        {
            "name": "oc03",
            "steps": [
                {
                    "id": "c1",
                    "type": "condition",
                    "condition": "'x'=='y'",
                    "then": "leaf_dead",
                    "otherwise": "c2",
                },
                {
                    "id": "c2",
                    "type": "condition",
                    "condition": "'a'=='a'",
                    "then": "step_a",
                    "otherwise": "leaf_dead",
                },
                {"id": "leaf_dead", "type": "tool", "tool": "a", "is_terminal": True},
                {"id": "step_a", "type": "tool", "tool": "a", "next_step": "step_b"},
                {"id": "step_b", "type": "tool", "tool": "b"},
            ],
        }
    )
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS, f"status={trace.status} err={trace.error}"
    assert ids == ["c1", "c2", "step_a", "step_b"], f"ids={ids}"
    assert order == ["a", "b"]


@pytest.mark.asyncio
async def test_oc04_symmetric_then_otherwise():
    """then-path and otherwise-path execute equal number of steps."""
    vm = make_vm({"echo": ECHO})

    # then path: c1(T)→c2→leaf_t
    prog_then = Program.from_dict(
        {
            "name": "oc04_then",
            "steps": [
                {
                    "id": "c1",
                    "type": "condition",
                    "condition": "'a'=='a'",
                    "then": "c2",
                    "otherwise": "leaf_dead",
                },
                {
                    "id": "c2",
                    "type": "condition",
                    "condition": "'a'=='a'",
                    "then": "leaf_t",
                    "otherwise": "leaf_dead",
                },
                {"id": "leaf_t", "type": "tool", "tool": "echo", "is_terminal": True},
                {"id": "leaf_dead", "type": "tool", "tool": "echo", "is_terminal": True},
            ],
        }
    )
    t1 = await vm.run(prog_then)

    # otherwise path: c1(F)→c2→leaf_o
    prog_oth = Program.from_dict(
        {
            "name": "oc04_otherwise",
            "steps": [
                {
                    "id": "c1",
                    "type": "condition",
                    "condition": "'a'=='b'",
                    "then": "leaf_dead",
                    "otherwise": "c2",
                },
                {
                    "id": "c2",
                    "type": "condition",
                    "condition": "'a'=='a'",
                    "then": "leaf_o",
                    "otherwise": "leaf_dead",
                },
                {"id": "leaf_o", "type": "tool", "tool": "echo", "is_terminal": True},
                {"id": "leaf_dead", "type": "tool", "tool": "echo", "is_terminal": True},
            ],
        }
    )
    t2 = await vm.run(prog_oth)

    assert t1.status == TraceStatus.SUCCESS
    assert t2.status == TraceStatus.SUCCESS
    assert len(t1.steps) == len(t2.steps) == 3, (
        f"then={[s.step_id for s in t1.steps]} oth={[s.step_id for s in t2.steps]}"
    )


@pytest.mark.asyncio
async def test_oc05_mixed_then_otherwise():
    """then→condition→otherwise→tool: c1(T)→c2, c2(F)→leaf_no."""
    vm = make_vm({"echo": ECHO})
    prog = Program.from_dict(
        {
            "name": "oc05",
            "steps": [
                {
                    "id": "c1",
                    "type": "condition",
                    "condition": "'a'=='a'",
                    "then": "c2",
                    "otherwise": "leaf_dead",
                },
                {
                    "id": "c2",
                    "type": "condition",
                    "condition": "'a'=='b'",
                    "then": "leaf_yes",
                    "otherwise": "leaf_no",
                },
                {"id": "leaf_yes", "type": "tool", "tool": "echo", "is_terminal": True},
                {"id": "leaf_no", "type": "tool", "tool": "echo", "is_terminal": True},
                {"id": "leaf_dead", "type": "tool", "tool": "echo", "is_terminal": True},
            ],
        }
    )
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["c1", "c2", "leaf_no"], f"ids={ids}"


@pytest.mark.asyncio
async def test_oc06_regression_then_chain():
    """Regression: CB-04 then-chain still works."""
    vm = make_vm({"echo": ECHO})
    prog = Program.from_dict(
        {
            "name": "oc06_reg",
            "steps": [
                {
                    "id": "c1",
                    "type": "condition",
                    "condition": "'a'=='a'",
                    "then": "c2",
                    "otherwise": "leaf_no",
                },
                {
                    "id": "c2",
                    "type": "condition",
                    "condition": "'b'=='b'",
                    "then": "leaf_yes",
                    "otherwise": "leaf_no",
                },
                {"id": "leaf_yes", "type": "tool", "tool": "echo", "is_terminal": True},
                {"id": "leaf_no", "type": "tool", "tool": "echo", "is_terminal": True},
            ],
        }
    )
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["c1", "c2", "leaf_yes"]


@pytest.mark.asyncio
async def test_oc07_regression_2level_otherwise():
    """Regression: CB-05 2-level otherwise chain still works."""
    vm = make_vm({"echo": ECHO})
    prog = Program.from_dict(
        {
            "name": "oc07_reg",
            "steps": [
                {
                    "id": "c1",
                    "type": "condition",
                    "condition": "'a'=='b'",
                    "then": "leaf_yes",
                    "otherwise": "c2",
                },
                {
                    "id": "c2",
                    "type": "condition",
                    "condition": "'x'=='y'",
                    "then": "leaf_yes",
                    "otherwise": "leaf_no",
                },
                {"id": "leaf_yes", "type": "tool", "tool": "echo", "is_terminal": True},
                {"id": "leaf_no", "type": "tool", "tool": "echo", "is_terminal": True},
            ],
        }
    )
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["c1", "c2", "leaf_no"]


# ------------------------------------------------------------------ #
# MC: Method Call detection                                           #
# ------------------------------------------------------------------ #


def test_mc01_lower_raises():
    """.lower() in condition → ASTEvalError."""
    with pytest.raises(ASTEvalError, match="Method calls are not supported"):
        eval_condition("'yes' in $decision.lower()", {"decision": "YES"})


def test_mc02_strip_raises():
    """.strip() in condition → ASTEvalError."""
    with pytest.raises(ASTEvalError, match="Method calls are not supported"):
        eval_condition("$decision.strip() == 'yes'", {"decision": " yes "})


def test_mc03_upper_raises():
    """.upper() in condition → ASTEvalError."""
    with pytest.raises(ASTEvalError, match="Method calls are not supported"):
        eval_condition("$decision.upper() == 'YES'", {"decision": "yes"})


def test_mc04_split_raises():
    """.split() in condition → ASTEvalError."""
    with pytest.raises(ASTEvalError, match="Method calls are not supported"):
        eval_condition("'yes' in $decision.split()", {"decision": "yes no"})


def test_mc05_valid_operators_unchanged():
    """Valid operators still work after METHOD_CALL pattern added."""
    assert eval_condition("'yes' == 'yes'", {}) is True
    assert eval_condition("$x == 'ok'", {"x": "ok"}) is True
    assert eval_condition("$x != 'ok'", {"x": "fail"}) is True
    assert eval_condition("'yes' in $x", {"x": "yes maybe"}) is True
    assert eval_condition("$x > 5", {"x": 10}) is True
    assert eval_condition("$x == 'yes' and $y == 'ok'", {"x": "yes", "y": "ok"}) is True


def test_mc06_regression_dotted_path():
    """Regression: $step.output.field dotted path still works."""
    assert (
        eval_condition(
            "$poll.output.payment_status == 'SUCCESS'",
            {"poll": {"output": {"payment_status": "SUCCESS"}}},
        )
        is True
    )


def test_mc07_vm_execute_condition_method_call_raises():
    """.lower() in condition propagates as VMError from _execute_condition."""
    vm = make_vm()
    state = StateContext(data={}, step_outputs={"decision": "YES"})
    step = Step(
        id="chk",
        type=StepType.CONDITION,
        condition="'yes' in $decision.lower()",
        then="yes",
        otherwise="no",
    )
    with pytest.raises(VMError, match="Condition eval error"):
        vm._execute_condition(step, state)
