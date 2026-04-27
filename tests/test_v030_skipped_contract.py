"""
tests/test_v030_skipped_contract.py
====================================
v0.3.0 — partial result contract: SKIPPED sub-step → output=None.

Покрывает:
  S1 — SKIPPED sub-step явно присутствует в outputs dict со значением None
  S2 — $sub_id.output резолвится в строку "None" для downstream шагов
  S3 — все sub-steps SKIPPED → outputs = {id: None, ...}, trace SUCCESS
  S4 — смесь SUCCESS + SKIPPED: SUCCESS-значения не затронуты
  S5 — output_key параллельного блока содержит None для SKIPPED
  S6 — FAILED sub-step с on_error=FAIL не попадает в outputs (trace FAILED)
"""

from __future__ import annotations

import pytest

from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.models import OnError, Step, StepStatus, StepType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLLM:
    def __init__(self, response: str = "llm_ok"):
        self._response = response

    async def complete(self, messages):
        return self._response


def make_vm(tools: dict | None = None) -> ExecutionVM:
    return ExecutionVM(llm=MockLLM(), tools=tools or {})


async def tool_ok() -> str:
    return "value"


async def tool_fail(**_) -> str:
    raise RuntimeError("intentional failure")


# ---------------------------------------------------------------------------
# S1 — SKIPPED явно присутствует в outputs с None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skipped_substep_output_is_none_in_dict():
    vm = make_vm({"ok": tool_ok, "fail": tool_fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                on_error=OnError.SKIP,
                parallel_steps=[
                    Step(id="good", type=StepType.TOOL, tool="ok"),
                    Step(id="bad", type=StepType.TOOL, tool="fail"),
                ],
            )
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    parent = next(s for s in trace.steps if s.step_id == "par")
    # "bad" должен быть в dict, значение None — не absent key
    assert "bad" in parent.output
    assert parent.output["bad"] is None
    assert parent.output["good"] == "value"


# ---------------------------------------------------------------------------
# S2 — $sub_id.output резолвится в downstream шаге
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skipped_output_resolves_in_downstream_prompt():
    """
    Downstream LLM-шаг получает "$bad.output" → должен резолвиться в "None" (str),
    не в нераспознанный $bad.output.
    """
    received_prompts: list[str] = []

    class CaptureLLM:
        async def complete(self, messages):
            received_prompts.append(messages[-1]["content"])
            return "summarized"

    vm = ExecutionVM(llm=CaptureLLM(), tools={"ok": tool_ok, "fail": tool_fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                on_error=OnError.SKIP,
                parallel_steps=[
                    Step(id="good", type=StepType.TOOL, tool="ok"),
                    Step(id="bad", type=StepType.TOOL, tool="fail"),
                ],
            ),
            Step(
                id="summarize",
                type=StepType.LLM,
                prompt="Good: $good.output\nBad: $bad.output",
            ),
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    assert len(received_prompts) == 1
    prompt = received_prompts[0]
    assert "Good: value" in prompt
    # $bad.output должен быть "None", не буквальным "$bad.output"
    assert "Bad: None" in prompt
    assert "$bad.output" not in prompt


# ---------------------------------------------------------------------------
# S3 — все sub-steps SKIPPED → trace SUCCESS, outputs все None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_substeps_skipped_trace_success():
    vm = make_vm({"fail": tool_fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                on_error=OnError.SKIP,
                parallel_steps=[
                    Step(id="s1", type=StepType.TOOL, tool="fail"),
                    Step(id="s2", type=StepType.TOOL, tool="fail"),
                    Step(id="s3", type=StepType.TOOL, tool="fail"),
                ],
            )
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    parent = next(s for s in trace.steps if s.step_id == "par")
    assert parent.status == StepStatus.SUCCESS
    assert parent.output == {"s1": None, "s2": None, "s3": None}


# ---------------------------------------------------------------------------
# S4 — смесь SUCCESS + SKIPPED: SUCCESS-значения не затронуты
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_success_and_skipped_values_intact():
    async def named_ok(name: str) -> str:
        return f"result_{name}"

    vm = make_vm({"named": named_ok, "fail": tool_fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                on_error=OnError.SKIP,
                parallel_steps=[
                    Step(id="a", type=StepType.TOOL, tool="named", args={"name": "a"}),
                    Step(id="b", type=StepType.TOOL, tool="fail"),
                    Step(id="c", type=StepType.TOOL, tool="named", args={"name": "c"}),
                ],
            )
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    parent = next(s for s in trace.steps if s.step_id == "par")
    assert parent.output["a"] == "result_a"
    assert parent.output["b"] is None
    assert parent.output["c"] == "result_c"


# ---------------------------------------------------------------------------
# S5 — output_key параллельного блока содержит None для SKIPPED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_key_dict_contains_none_for_skipped():
    captured = {}

    async def capture(key: str, val: str) -> str:
        captured[key] = val
        return val

    vm = make_vm({"capture": capture, "fail": tool_fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                output_key="par_result",
                on_error=OnError.SKIP,
                parallel_steps=[
                    Step(id="ok1", type=StepType.TOOL, tool="capture", args={"key": "k1", "val": "v1"}),
                    Step(id="bad1", type=StepType.TOOL, tool="fail"),
                ],
            ),
            # Следующий шаг читает output_key через $par_result (dict)
            Step(id="noop", type=StepType.LLM, prompt="result: $par_result"),
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    parent = next(s for s in trace.steps if s.step_id == "par")
    # output_key хранит полный dict включая None
    assert parent.output["ok1"] == "v1"
    assert parent.output["bad1"] is None


# ---------------------------------------------------------------------------
# S6 — on_error=FAIL: FAILED sub-step → trace FAILED, outputs не формируется
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_substep_with_on_error_fail_aborts_trace():
    vm = make_vm({"ok": tool_ok, "fail": tool_fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                on_error=OnError.FAIL,
                parallel_steps=[
                    Step(id="good", type=StepType.TOOL, tool="ok"),
                    Step(id="bad", type=StepType.TOOL, tool="fail"),
                ],
            ),
            Step(id="next", type=StepType.LLM, prompt="should not run"),
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.FAILED
    step_ids = {s.step_id for s in trace.steps}
    assert "next" not in step_ids
