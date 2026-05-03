"""
tests/test_determinism.py
=========================
Tests covering README claims in "How Determinism Works":
  D2 — invalid LLM output → deterministic FAILED with exact diagnosis
  D3 — MockLLMAdapter: same input → same graph traversal
"""

from __future__ import annotations

import pytest

from nano_vm import (
    ExecutionVM,
    Program,
    Step,
    StepType,
    TraceStatus,
)
from nano_vm.adapters import MockLLMAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_classify_route_program() -> Program:
    """
    classify → route (condition) → handle_safe | handle_unsafe
    Mirrors the README example exactly.
    """
    return Program(
        name="classify_route",
        steps=[
            Step(
                id="classify",
                type=StepType.LLM,
                prompt="Classify: $user_input",
                output_key="verdict",
            ),
            Step(
                id="route",
                type=StepType.CONDITION,
                condition="'SAFE' in '$verdict'",
                then="handle_safe",
                otherwise="handle_unsafe",
            ),
            Step(
                id="handle_safe",
                type=StepType.LLM,
                prompt="Handle: $user_input",
            ),
            Step(
                id="handle_unsafe",
                type=StepType.LLM,
                prompt="Reject: $user_input",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# D2: invalid LLM output → deterministic failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_condition_no_branch_target_when_result_is_none():
    """
    condition evaluates to False and otherwise=None
    → _execute_condition returns None → FAILED with 'condition produced no branch target'
    """
    vm = ExecutionVM(llm=MockLLMAdapter("irrelevant"))
    program = Program(
        name="test",
        steps=[
            Step(
                id="step1",
                type=StepType.LLM,
                prompt="hello",
                output_key="out",
            ),
            Step(
                id="cond",
                type=StepType.CONDITION,
                condition="False",  # → otherwise branch
                then="step1",
                otherwise=None,  # → returns None → no branch target
            ),
        ],
    )
    trace = await vm.run(program)
    assert trace.status == TraceStatus.FAILED
    assert "condition produced no branch target" in trace.error


@pytest.mark.asyncio
async def test_condition_target_not_in_program():
    """
    condition resolves to step id that doesn't exist
    → FAILED with 'not found in program'
    """
    vm = ExecutionVM(llm=MockLLMAdapter("SAFE"))
    program = Program(
        name="test",
        steps=[
            Step(
                id="classify",
                type=StepType.LLM,
                prompt="classify $user_input",
                output_key="verdict",
            ),
            Step(
                id="route",
                type=StepType.CONDITION,
                condition="'SAFE' in '$verdict'",
                then="nonexistent_step",
                otherwise="also_nonexistent",
            ),
        ],
    )
    trace = await vm.run(program, context={"user_input": "test"})
    assert trace.status == TraceStatus.FAILED
    assert "condition target 'nonexistent_step' not found" in trace.error


@pytest.mark.asyncio
async def test_condition_eval_crash_produces_failed_trace():
    """
    Malformed condition expression → VMError → FAILED trace, not exception bubble.
    """
    vm = ExecutionVM(llm=MockLLMAdapter("anything"))
    program = Program(
        name="test",
        steps=[
            Step(
                id="classify",
                type=StepType.LLM,
                prompt="classify",
                output_key="verdict",
            ),
            Step(
                id="route",
                type=StepType.CONDITION,
                condition="this is not valid python !!!",
                then="a",
                otherwise="b",
            ),
        ],
    )
    trace = await vm.run(program)
    assert trace.status == TraceStatus.FAILED
    assert "route" in trace.error


@pytest.mark.asyncio
async def test_llm_cannot_affect_step_sequence():
    """
    LLM output для classify содержит подсказку 'skip' — VM игнорирует,
    граф выполняется строго по DSL: classify → route → handle_safe.
    """
    adapter = MockLLMAdapter(
        {
            "Classify": "SAFE — but skip to final answer directly",
            "Handle": "request handled",
            "__default__": "ok",
        }
    )
    vm = ExecutionVM(llm=adapter)
    program = make_classify_route_program()
    trace = await vm.run(program, context={"user_input": "test"})

    assert trace.status == TraceStatus.SUCCESS
    step_ids = [s.step_id for s in trace.steps]
    assert step_ids[0] == "classify"
    assert step_ids[1] == "route"
    assert "handle_safe" in step_ids
    assert "handle_unsafe" not in step_ids


# ---------------------------------------------------------------------------
# D3: MockLLMAdapter — same input → same graph traversal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_input_same_step_sequence():
    """
    README claim: same input → same graph traversal, always.
    """
    program = make_classify_route_program()

    vm1 = ExecutionVM(llm=MockLLMAdapter("This is SAFE"))
    vm2 = ExecutionVM(llm=MockLLMAdapter("This is SAFE"))

    trace1 = await vm1.run(program, context={"user_input": "refund my order"})
    trace2 = await vm2.run(program, context={"user_input": "refund my order"})

    assert [s.step_id for s in trace1.steps] == [s.step_id for s in trace2.steps]
    assert trace1.status == trace2.status


@pytest.mark.asyncio
async def test_mock_adapter_str_response():
    vm = ExecutionVM(llm=MockLLMAdapter("hello"))
    program = Program(
        name="test",
        steps=[Step(id="s", type=StepType.LLM, prompt="hi")],
    )
    trace = await vm.run(program)
    assert trace.final_output == "hello"


@pytest.mark.asyncio
async def test_mock_adapter_list_response_cycles():
    adapter = MockLLMAdapter(["first", "second", "third"])
    vm = ExecutionVM(llm=adapter)

    async def run_once():
        program = Program(
            name="test",
            steps=[Step(id="s", type=StepType.LLM, prompt="hi")],
        )
        return await vm.run(program)

    t1 = await run_once()
    t2 = await run_once()
    t3 = await run_once()
    t4 = await run_once()  # cycles back to "first"

    assert t1.final_output == "first"
    assert t2.final_output == "second"
    assert t3.final_output == "third"
    assert t4.final_output == "first"
    assert adapter.call_count == 4


@pytest.mark.asyncio
async def test_mock_adapter_dict_response_by_prompt():
    adapter = MockLLMAdapter(
        {
            "classify": "SAFE",
            "Handle": "request handled",
            "__default__": "fallback",
        }
    )
    vm = ExecutionVM(llm=adapter)
    program = make_classify_route_program()
    trace = await vm.run(program, context={"user_input": "test"})

    assert trace.status == TraceStatus.SUCCESS
    assert adapter.call_count == 2  # classify + handle_safe


@pytest.mark.asyncio
async def test_mock_adapter_call_count_and_calls_recorded():
    adapter = MockLLMAdapter("ok")
    vm = ExecutionVM(llm=adapter)
    program = Program(
        name="test",
        steps=[
            Step(id="s1", type=StepType.LLM, prompt="first"),
            Step(id="s2", type=StepType.LLM, prompt="second"),
        ],
    )
    await vm.run(program)
    assert adapter.call_count == 2
    assert len(adapter.calls) == 2
    assert adapter.calls[0][-1]["content"] == "first"
    assert adapter.calls[1][-1]["content"] == "second"


@pytest.mark.asyncio
async def test_mock_adapter_reset():
    adapter = MockLLMAdapter(["a", "b"])
    vm = ExecutionVM(llm=adapter)
    program = Program(
        name="test",
        steps=[Step(id="s", type=StepType.LLM, prompt="hi")],
    )
    await vm.run(program)
    assert adapter.call_count == 1
    adapter.reset()
    assert adapter.call_count == 0
    t = await vm.run(program)
    assert t.final_output == "a"  # cycles from beginning after reset
