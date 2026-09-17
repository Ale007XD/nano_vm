"""
tests/test_llm_optional.py
===========================
ExecutionVM(llm=None): tool-only execution, and pre-flight rejection of any
program that declares llm steps when no adapter is configured.

Q2 (2026-09-16): the rejection must happen BEFORE any step executes, not at
the point the llm step itself runs. A program like tool_step -> llm_step
must produce zero tool side effects when llm=None, not run the tool step
and then fail. The check deliberately over-approximates reachability: it
rejects if the program declares ANY llm step at all, even one that the
actual execution path would never reach, matching ProgramValidator's own
conservative stance (better a false-positive reject than a partial run).
"""

from __future__ import annotations

import pytest

from nano_vm import ExecutionVM, Program, StateContext, Step, StepType, TraceStatus, VMError
from nano_vm.adapters import MockLLMAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TOOL_ONLY = {
    "name": "tool_only",
    "steps": [
        {"id": "t1", "type": "tool", "tool": "spy"},
        {"id": "t2", "type": "tool", "tool": "spy", "is_terminal": True},
    ],
}

# t1 (tool, real side effect) runs first, THEN ask (llm) -- the exact shape
# that must produce zero calls to `spy` when llm=None.
TOOL_THEN_LLM = {
    "name": "tool_then_llm",
    "steps": [
        {"id": "t1", "type": "tool", "tool": "spy"},
        {"id": "ask", "type": "llm", "prompt": "hi", "output_key": "x", "is_terminal": True},
    ],
}

# The llm step is declared but structurally unreachable (t1 is terminal,
# nothing ever jumps to "ask"). Pre-flight must still reject: reachability
# analysis is explicitly out of scope for this check (see _require_llm_if_needed).
TOOL_ONLY_WITH_DEAD_LLM_STEP = {
    "name": "dead_llm_step",
    "steps": [
        {"id": "t1", "type": "tool", "tool": "spy", "is_terminal": True},
        {"id": "ask", "type": "llm", "prompt": "hi", "output_key": "x", "is_terminal": True},
    ],
}


def make_spy() -> tuple[list[dict], object]:
    calls: list[dict] = []

    def spy(**kwargs: object) -> str:
        calls.append(kwargs)
        return "ok"

    return calls, spy


# ---------------------------------------------------------------------------
# Happy path: tool-only programs run fine with llm=None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_only_program_runs_without_llm_adapter():
    calls, spy = make_spy()
    vm = ExecutionVM(llm=None, tools={"spy": spy})
    trace = await vm.run(Program.from_dict(TOOL_ONLY))
    assert trace.status == TraceStatus.SUCCESS
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_llm_program_runs_fine_when_adapter_is_configured():
    """Control: the same TOOL_THEN_LLM program succeeds end-to-end once an
    adapter is actually provided -- proves the pre-flight check isn't just
    rejecting unconditionally."""
    calls, spy = make_spy()
    vm = ExecutionVM(llm=MockLLMAdapter("done"), tools={"spy": spy})
    trace = await vm.run(Program.from_dict(TOOL_THEN_LLM))
    assert trace.status == TraceStatus.SUCCESS
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Q2 — pre-flight rejection, zero side effects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_program_without_adapter_rejected_before_first_step():
    calls, spy = make_spy()
    vm = ExecutionVM(llm=None, tools={"spy": spy})

    with pytest.raises(VMError) as exc_info:
        await vm.run(Program.from_dict(TOOL_THEN_LLM))

    assert "ask" in str(exc_info.value)
    assert calls == [], "tool step ran before the llm-adapter check rejected the program"


@pytest.mark.asyncio
async def test_unreachable_llm_step_still_rejected_preflight():
    """Deliberate over-approximation: an llm step nobody can ever reach
    still blocks the whole program when llm=None. Reachability analysis is
    ProgramValidator's job (unreachable_steps), not this check's."""
    calls, spy = make_spy()
    vm = ExecutionVM(llm=None, tools={"spy": spy})

    with pytest.raises(VMError) as exc_info:
        await vm.run(Program.from_dict(TOOL_ONLY_WITH_DEAD_LLM_STEP))

    assert "ask" in str(exc_info.value)
    assert calls == []


# ---------------------------------------------------------------------------
# Backstop: the in-loop check in _execute_llm still holds on its own
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_llm_backstop_raises_without_preflight():
    """Direct unit test on the inner guard, independent of the pre-flight
    gate in run()/resume_with_program() -- in case some future call path
    ever reaches _execute_llm without going through either of those first.
    """
    vm = ExecutionVM(llm=None)
    step = Step(id="ask", type=StepType.LLM, prompt="hi", output_key="x")

    with pytest.raises(VMError, match="ask"):
        await vm._execute_llm(step, StateContext())
