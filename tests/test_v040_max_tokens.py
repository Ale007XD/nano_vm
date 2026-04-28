"""
test_v040_max_tokens.py
=======================
P3: max_tokens budget via LiteLLM usage aggregation.

Semantics: check fires BEFORE next step starts.
If total_tokens >= max_tokens → BUDGET_EXCEEDED; the triggering step is NOT executed.

Coverage:
  - max_tokens=None → no cap, all steps execute (regression)
  - max_tokens > actual total → all steps execute, SUCCESS
  - max_tokens fires after step 1 consumed enough tokens
  - max_tokens=0 → BUDGET_EXCEEDED before any step (0 tokens >= 0)
  - BUDGET_EXCEEDED error contains max_tokens and consumed count
  - BUDGET_EXCEEDED final_output is None
  - Non-LLM steps (tool) have no usage → don't consume token budget
  - Combined with max_steps: whichever fires first wins
  - Steps without usage (usage=None) counted as 0 tokens
  - total_tokens() aggregates across multiple steps correctly
"""

from __future__ import annotations

import pytest

from nano_vm.models import LLMUsage, Program, Step, StepStatus, StepType, TraceStatus
from nano_vm.vm import ExecutionVM

# ---------------------------------------------------------------------------
# Fake adapter that injects token counts
# ---------------------------------------------------------------------------


class _CountingAdapter:
    """Returns prompt text; injects a fixed token count per call."""

    def __init__(self, tokens_per_call: int = 10):
        self.tokens_per_call = tokens_per_call
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        return messages[-1]["content"]


class _CountingVM(ExecutionVM):
    """
    Subclass that patches _run_step to inject LLMUsage into LLM step results.
    This simulates what the real LiteLLM adapter does.
    """

    def __init__(self, tokens_per_call: int = 10, **kwargs):
        super().__init__(llm=_CountingAdapter(tokens_per_call), **kwargs)
        self._tokens_per_call = tokens_per_call

    async def _run_step(self, step, state):
        result, new_state, sub_results = await super()._run_step(step, state)
        # Inject usage for LLM steps only
        if step.type == StepType.LLM and result.status == StepStatus.SUCCESS:
            usage = LLMUsage(
                prompt_tokens=self._tokens_per_call // 2,
                completion_tokens=self._tokens_per_call // 2,
                total_tokens=self._tokens_per_call,
            )
            result = result.model_copy(update={"usage": usage})
        return result, new_state, sub_results


def _llm(step_id: str) -> Step:
    return Step(id=step_id, type=StepType.LLM, prompt="hi")


def _tool_step(step_id: str, tool: str) -> Step:
    return Step(id=step_id, type=StepType.TOOL, tool=tool)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_tokens_none_no_cap():
    """max_tokens=None → all steps execute regardless of token count."""
    program = Program(
        name="t",
        max_tokens=None,
        steps=[_llm("s1"), _llm("s2"), _llm("s3")],
    )
    trace = await _CountingVM(tokens_per_call=100).run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert trace.total_tokens() == 300


@pytest.mark.asyncio
async def test_max_tokens_above_total_succeeds():
    """max_tokens > actual total → all steps run, SUCCESS."""
    program = Program(
        name="t",
        max_tokens=1000,
        steps=[_llm("s1"), _llm("s2")],
    )
    trace = await _CountingVM(tokens_per_call=10).run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert trace.total_tokens() == 20


@pytest.mark.asyncio
async def test_max_tokens_fires_after_first_step():
    """
    10 tokens per step, max_tokens=10.
    After s1: total=10 >= 10 → BUDGET_EXCEEDED before s2.
    """
    program = Program(
        name="t",
        max_tokens=10,
        steps=[_llm("s1"), _llm("s2"), _llm("s3")],
    )
    trace = await _CountingVM(tokens_per_call=10).run(program)
    assert trace.status == TraceStatus.BUDGET_EXCEEDED
    executed = {s.step_id for s in trace.steps}
    assert "s1" in executed
    assert "s2" not in executed
    assert "s3" not in executed


@pytest.mark.asyncio
async def test_max_tokens_fires_after_second_step():
    """
    10 tokens per step, max_tokens=15.
    After s1: total=10 < 15 → continue.
    After s2: total=20 >= 15 → BUDGET_EXCEEDED before s3.
    """
    program = Program(
        name="t",
        max_tokens=15,
        steps=[_llm("s1"), _llm("s2"), _llm("s3")],
    )
    trace = await _CountingVM(tokens_per_call=10).run(program)
    assert trace.status == TraceStatus.BUDGET_EXCEEDED
    executed = {s.step_id for s in trace.steps}
    assert "s1" in executed
    assert "s2" in executed
    assert "s3" not in executed


@pytest.mark.asyncio
async def test_max_tokens_zero_fires_immediately():
    """
    max_tokens=0: 0 tokens consumed >= 0 → BUDGET_EXCEEDED before s1.
    No steps execute.
    """
    program = Program(
        name="t",
        max_tokens=0,
        steps=[_llm("s1"), _llm("s2")],
    )
    trace = await _CountingVM(tokens_per_call=10).run(program)
    assert trace.status == TraceStatus.BUDGET_EXCEEDED
    assert trace.steps == []


@pytest.mark.asyncio
async def test_max_tokens_error_message():
    """BUDGET_EXCEEDED error contains max_tokens and consumed count."""
    program = Program(
        name="t",
        max_tokens=10,
        steps=[_llm("s1"), _llm("s2")],
    )
    trace = await _CountingVM(tokens_per_call=10).run(program)
    assert trace.status == TraceStatus.BUDGET_EXCEEDED
    assert trace.error is not None
    assert "max_tokens=10" in trace.error
    assert "10" in trace.error  # consumed tokens


@pytest.mark.asyncio
async def test_max_tokens_final_output_none():
    """BUDGET_EXCEEDED → final_output is None."""
    program = Program(
        name="t",
        max_tokens=10,
        steps=[_llm("s1"), _llm("s2")],
    )
    trace = await _CountingVM(tokens_per_call=10).run(program)
    assert trace.status == TraceStatus.BUDGET_EXCEEDED
    assert trace.final_output is None


@pytest.mark.asyncio
async def test_tool_steps_no_token_cost():
    """Tool steps produce no usage → don't consume token budget."""
    call_log = []

    def _tool(**_):
        call_log.append(1)
        return "ok"

    # 3 tool steps, then 1 LLM step — budget only exceeded after LLM
    program = Program(
        name="t",
        max_tokens=5,  # below 10 tokens of LLM step
        steps=[
            _tool_step("t1", "mytool"),
            _tool_step("t2", "mytool"),
            _llm("l1"),
        ],
    )
    vm = _CountingVM(tokens_per_call=10, tools={"mytool": _tool})
    trace = await vm.run(program)
    # t1, t2 → 0 tokens each; l1 → 10 tokens → check before l1: 0 < 5 → executes
    # after l1: 10 >= 5 → but there's no next step to block → SUCCESS
    assert trace.status == TraceStatus.SUCCESS
    assert len(call_log) == 2
    assert trace.total_tokens() == 10


@pytest.mark.asyncio
async def test_tool_then_llm_budget_blocks_second_llm():
    """tool (0 tokens) + LLM (10 tokens) + LLM (10 tokens), max_tokens=10."""
    call_log = []

    def _tool(**_):
        call_log.append(1)
        return "ok"

    program = Program(
        name="t",
        max_tokens=10,
        steps=[
            _tool_step("t1", "mytool"),
            _llm("l1"),
            _llm("l2"),
        ],
    )
    vm = _CountingVM(tokens_per_call=10, tools={"mytool": _tool})
    trace = await vm.run(program)
    # before t1: 0 < 10 → ok; before l1: 0 < 10 → ok; l1 costs 10
    # before l2: 10 >= 10 → BUDGET_EXCEEDED
    assert trace.status == TraceStatus.BUDGET_EXCEEDED
    executed = {s.step_id for s in trace.steps}
    assert "t1" in executed
    assert "l1" in executed
    assert "l2" not in executed


@pytest.mark.asyncio
async def test_max_tokens_and_max_steps_max_steps_wins():
    """
    max_steps=1, max_tokens=1000.
    max_steps check fires first (order: max_steps → max_tokens in loop).
    """
    program = Program(
        name="t",
        max_steps=1,
        max_tokens=1000,
        steps=[_llm("s1"), _llm("s2"), _llm("s3")],
    )
    trace = await _CountingVM(tokens_per_call=10).run(program)
    assert trace.status == TraceStatus.BUDGET_EXCEEDED
    executed = {s.step_id for s in trace.steps}
    assert "s1" in executed
    assert "s2" not in executed


@pytest.mark.asyncio
async def test_max_tokens_and_max_steps_tokens_wins():
    """
    max_steps=10, max_tokens=10.
    Token budget exhausted after step 1 → BUDGET_EXCEEDED before step 2.
    max_steps check passes (1 < 10), token check fires.
    """
    program = Program(
        name="t",
        max_steps=10,
        max_tokens=10,
        steps=[_llm("s1"), _llm("s2"), _llm("s3")],
    )
    trace = await _CountingVM(tokens_per_call=10).run(program)
    assert trace.status == TraceStatus.BUDGET_EXCEEDED
    executed = {s.step_id for s in trace.steps}
    assert "s1" in executed
    assert "s2" not in executed


@pytest.mark.asyncio
async def test_total_tokens_aggregates_correctly():
    """total_tokens() = sum of all step usage.total_tokens."""
    program = Program(
        name="t",
        steps=[_llm("s1"), _llm("s2"), _llm("s3")],
    )
    trace = await _CountingVM(tokens_per_call=7).run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert trace.total_tokens() == 21
