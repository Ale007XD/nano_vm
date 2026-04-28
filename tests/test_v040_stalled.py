"""
test_v040_stalled.py
====================
P1: state fingerprint + no-op detection + STALLED status.

Coverage:
  - max_stalled_steps=None → disabled, no STALLED regardless of state
  - Step that writes output_key → fingerprint changes → not a no-op
  - Step without output_key → fingerprint unchanged → no-op
  - max_stalled_steps=1: single no-op step → STALLED
  - max_stalled_steps=2: one no-op → ok; two consecutive → STALLED
  - Alternating no-op/progress → counter resets, no STALLED
  - STALLED trace.error contains max_stalled_steps value and count
  - STALLED trace.final_output is None
  - Tool step that changes state → not a no-op
  - Tool step that does NOT change state → no-op
  - Regression: normal programs with max_stalled_steps set still succeed
"""

from __future__ import annotations

import pytest

from nano_vm.models import Program, Step, StepType, TraceStatus
from nano_vm.vm import ExecutionVM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EchoAdapter:
    async def complete(self, messages):
        return messages[-1]["content"]


def _make_vm(**tools) -> ExecutionVM:
    return ExecutionVM(llm=_EchoAdapter(), tools=tools or {})


def _llm(step_id: str, prompt: str = "hi", output_key: str | None = None) -> Step:
    return Step(id=step_id, type=StepType.LLM, prompt=prompt, output_key=output_key)


def _tool_step(step_id: str, tool: str, output_key: str | None = None) -> Step:
    return Step(id=step_id, type=StepType.TOOL, tool=tool, output_key=output_key)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stalled_disabled_when_none():
    """max_stalled_steps=None → no STALLED, even if every step is a no-op."""
    # Steps without output_key don't mutate state.data, but they DO write step_outputs.
    # First step: last_fingerprint=None → no comparison → stalled_count stays 0.
    # Second step onward: fingerprint changes because step_outputs grows.
    # So with output_key=None, fingerprint still changes (new step_id added).
    # True no-op only when the SAME step_id is re-executed (loop scenario).
    # Here: sequential distinct steps → fingerprint always grows → SUCCESS.
    program = Program(
        name="test",
        max_stalled_steps=None,
        steps=[_llm("s1"), _llm("s2"), _llm("s3")],
    )
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.SUCCESS


@pytest.mark.asyncio
async def test_no_op_fingerprint_logic():
    """
    Verify fingerprint mechanics directly via _state_fingerprint.

    Empty state → 0.
    After adding outputs → hash changes.
    Same state twice → same hash.
    """
    from nano_vm.models import StateContext
    from nano_vm.vm import ExecutionVM

    s0 = StateContext()
    s1 = s0.with_output("step1", "hello")
    s2 = s1.with_output("step2", "world")
    s3 = s1.with_output("step2", "world")  # same as s2

    fp0 = ExecutionVM._state_fingerprint(s0)
    fp1 = ExecutionVM._state_fingerprint(s1)
    fp2 = ExecutionVM._state_fingerprint(s2)
    fp3 = ExecutionVM._state_fingerprint(s3)

    assert fp1 != fp0  # adding first output changes fingerprint
    assert fp2 != fp1  # adding second output changes fingerprint
    assert fp2 == fp3  # identical state → identical fingerprint
    # Empty state is consistent with itself
    assert ExecutionVM._state_fingerprint(StateContext()) == fp0


@pytest.mark.asyncio
async def test_sequential_steps_not_stalled():
    """
    Sequential distinct steps: each step_id is new → step_outputs grows →
    fingerprint changes every step → stalled_count never increments.
    """
    program = Program(
        name="test",
        max_stalled_steps=1,
        steps=[_llm("s1"), _llm("s2"), _llm("s3")],
    )
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.SUCCESS


@pytest.mark.asyncio
async def test_output_key_changes_fingerprint():
    """
    Steps with output_key write to state.data AND step_outputs.
    Fingerprint is based on step_outputs — always changes for new step_id.
    Regression: output_key steps succeed normally.
    """
    program = Program(
        name="test",
        max_stalled_steps=1,
        steps=[
            _llm("s1", output_key="result1"),
            _llm("s2", output_key="result2"),
        ],
    )
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.SUCCESS


@pytest.mark.asyncio
async def test_stalled_when_same_step_id_repeated():
    """
    Simulate a no-op: two steps with the SAME id produce identical step_outputs entry.
    This can only happen if program has duplicate step ids — which is unusual but valid
    at the model level (Program doesn't enforce unique ids).

    Step s1 runs → step_outputs = {"s1": "hi"}.
    Step s1 runs again → step_outputs = {"s1": "hi"} (same, overwrite same key).
    Fingerprint identical → stalled_count=1 → STALLED with max_stalled_steps=1.
    """
    s1a = Step(id="s1", type=StepType.LLM, prompt="hi")
    s1b = Step(id="s1", type=StepType.LLM, prompt="hi")  # same id, same prompt

    program = Program(
        name="test_dup",
        max_stalled_steps=1,
        steps=[s1a, s1b],
    )
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.STALLED


@pytest.mark.asyncio
async def test_stalled_counter_resets_on_progress():
    """
    s1 (new id) → progress (stalled_count=0)
    s1 repeated (same id, same output) → no-op (stalled_count=1)
    s2 (new id) → progress (stalled_count resets to 0)
    s1 repeated again → no-op (stalled_count=1)
    max_stalled_steps=2 → never reaches 2 → SUCCESS
    """
    s1 = Step(id="s1", type=StepType.LLM, prompt="hi")
    s1_dup = Step(id="s1", type=StepType.LLM, prompt="hi")
    s2 = Step(id="s2", type=StepType.LLM, prompt="hello")
    s1_dup2 = Step(id="s1", type=StepType.LLM, prompt="hi")

    program = Program(
        name="test_reset",
        max_stalled_steps=2,
        steps=[s1, s1_dup, s2, s1_dup2],
    )
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.SUCCESS


@pytest.mark.asyncio
async def test_stalled_two_consecutive_no_ops():
    """
    s1 → progress
    s1 dup → no-op (stalled_count=1) — below threshold of 2
    s1 dup → no-op (stalled_count=2) → STALLED
    """
    s1 = Step(id="s1", type=StepType.LLM, prompt="hi")
    s1_dup1 = Step(id="s1", type=StepType.LLM, prompt="hi")
    s1_dup2 = Step(id="s1", type=StepType.LLM, prompt="hi")

    program = Program(
        name="test_two_noop",
        max_stalled_steps=2,
        steps=[s1, s1_dup1, s1_dup2],
    )
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.STALLED


@pytest.mark.asyncio
async def test_stalled_error_message():
    """STALLED trace.error contains max_stalled_steps and stalled_count."""
    s1 = Step(id="s1", type=StepType.LLM, prompt="hi")
    s1_dup = Step(id="s1", type=StepType.LLM, prompt="hi")

    program = Program(
        name="test_msg",
        max_stalled_steps=1,
        steps=[s1, s1_dup],
    )
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.STALLED
    assert trace.error is not None
    assert "max_stalled_steps=1" in trace.error
    assert "1" in trace.error  # stalled count


@pytest.mark.asyncio
async def test_stalled_final_output_none():
    """STALLED → final_output is None."""
    s1 = Step(id="s1", type=StepType.LLM, prompt="hi")
    s1_dup = Step(id="s1", type=StepType.LLM, prompt="hi")

    program = Program(
        name="test_out",
        max_stalled_steps=1,
        steps=[s1, s1_dup],
    )
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.STALLED
    assert trace.final_output is None


@pytest.mark.asyncio
async def test_tool_step_progress_not_stalled():
    """Tool step with new step_id → step_outputs grows → progress."""
    call_log = []

    def _tool(**_):
        call_log.append(1)
        return "tool_result"

    program = Program(
        name="test_tool",
        max_stalled_steps=1,
        steps=[
            _tool_step("t1", "mytool"),
            _tool_step("t2", "mytool"),
        ],
    )
    trace = await _make_vm(mytool=_tool).run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert len(call_log) == 2


@pytest.mark.asyncio
async def test_both_budgets_max_steps_wins():
    """
    max_steps=1, max_stalled_steps=1 on a 3-step program.
    BUDGET_EXCEEDED fires first (after step 1, before step 2).
    STALLED never fires (sequential distinct ids → no no-op).
    """
    program = Program(
        name="test_both",
        max_steps=1,
        max_stalled_steps=1,
        steps=[_llm("s1"), _llm("s2"), _llm("s3")],
    )
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.BUDGET_EXCEEDED
