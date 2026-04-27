"""
Tests for StepType.PARALLEL (DSL v0.4)
"""

from __future__ import annotations

import asyncio

import pytest

from nano_vm.models import (
    OnError,
    Program,
    Step,
    StepStatus,
    StepType,
    TraceStatus,
)
from nano_vm.vm import ExecutionVM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLLM:
    async def complete(self, messages):
        return "llm_response"


def make_vm(tools: dict | None = None) -> ExecutionVM:
    return ExecutionVM(llm=MockLLM(), tools=tools or {})


async def tool_echo(value: str) -> str:
    return f"echo:{value}"


async def tool_slow(delay: float = 0.05) -> str:
    await asyncio.sleep(delay)
    return "slow_done"


async def tool_fail(**kwargs) -> str:
    raise RuntimeError("tool_fail intentional")


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def test_parallel_step_requires_parallel_steps():
    with pytest.raises(ValueError, match="requires at least one parallel_steps"):
        Step(id="p", type=StepType.PARALLEL)


def test_parallel_step_rejects_condition_sub():
    with pytest.raises(ValueError, match="cannot be of type"):
        Step(
            id="p",
            type=StepType.PARALLEL,
            parallel_steps=[
                Step(id="c", type=StepType.CONDITION, condition="True", then="x"),
            ],
        )


def test_parallel_step_rejects_nested_parallel():
    with pytest.raises(ValueError, match="cannot be of type"):
        Step(
            id="p",
            type=StepType.PARALLEL,
            parallel_steps=[
                Step(
                    id="inner",
                    type=StepType.PARALLEL,
                    parallel_steps=[
                        Step(id="t", type=StepType.TOOL, tool="x"),
                    ],
                ),
            ],
        )


def test_parallel_step_valid():
    step = Step(
        id="p",
        type=StepType.PARALLEL,
        parallel_steps=[
            Step(id="t1", type=StepType.TOOL, tool="echo"),
            Step(id="t2", type=StepType.TOOL, tool="echo"),
        ],
    )
    assert len(step.parallel_steps) == 2


# ---------------------------------------------------------------------------
# Execution: happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_two_tools_success():
    vm = make_vm({"echo": tool_echo})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                output_key="results",
                parallel_steps=[
                    Step(id="s1", type=StepType.TOOL, tool="echo", args={"value": "a"}),
                    Step(id="s2", type=StepType.TOOL, tool="echo", args={"value": "b"}),
                ],
            )
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    # parent step result
    parent = next(s for s in trace.steps if s.step_id == "par")
    assert parent.status == StepStatus.SUCCESS
    assert parent.output == {"s1": "echo:a", "s2": "echo:b"}
    # sub-steps in trace
    sub_ids = {s.step_id for s in trace.steps}
    assert "s1" in sub_ids
    assert "s2" in sub_ids


@pytest.mark.asyncio
async def test_parallel_output_key_stored_in_state():
    """output_key stores the full dict; sub-step ids stored individually."""
    captured = {}

    async def tool_capture(key: str, value: str) -> str:
        captured[key] = value
        return value

    vm = make_vm({"capture": tool_capture})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                output_key="par_out",
                parallel_steps=[
                    Step(
                        id="sa", type=StepType.TOOL, tool="capture", args={"key": "a", "value": "A"}
                    ),
                    Step(
                        id="sb", type=StepType.TOOL, tool="capture", args={"key": "b", "value": "B"}
                    ),
                ],
            ),
            Step(
                id="use",
                type=StepType.TOOL,
                tool="capture",
                args={"key": "used", "value": "$sa.output"},
            ),
        ],
    )
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert captured["used"] == "A"


@pytest.mark.asyncio
async def test_parallel_concurrency():
    """Two slow tools finish faster than sequential would."""
    import time

    vm = make_vm({"slow": tool_slow})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                parallel_steps=[
                    Step(id="sl1", type=StepType.TOOL, tool="slow", args={"delay": 0.1}),
                    Step(id="sl2", type=StepType.TOOL, tool="slow", args={"delay": 0.1}),
                ],
            )
        ],
    )
    t0 = time.perf_counter()
    trace = await vm.run(program)
    elapsed = time.perf_counter() - t0

    assert trace.status == TraceStatus.SUCCESS
    # Sequential would take >=0.2s; parallel should be <0.18s
    assert elapsed < 0.18, f"Expected concurrent execution, got {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_parallel_with_llm_sub_steps():
    vm = make_vm()
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                parallel_steps=[
                    Step(id="l1", type=StepType.LLM, prompt="hello"),
                    Step(id="l2", type=StepType.LLM, prompt="world"),
                ],
            )
        ],
    )
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    parent = next(s for s in trace.steps if s.step_id == "par")
    assert parent.output == {"l1": "llm_response", "l2": "llm_response"}


# ---------------------------------------------------------------------------
# Execution: on_error=FAIL (default)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_fail_on_sub_error():
    vm = make_vm({"echo": tool_echo, "fail": tool_fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                on_error=OnError.FAIL,
                parallel_steps=[
                    Step(id="ok", type=StepType.TOOL, tool="echo", args={"value": "x"}),
                    Step(id="bad", type=StepType.TOOL, tool="fail"),
                ],
            )
        ],
    )
    trace = await vm.run(program)
    assert trace.status == TraceStatus.FAILED
    assert "bad" in trace.error


@pytest.mark.asyncio
async def test_parallel_fail_propagates_to_trace():
    vm = make_vm({"fail": tool_fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                parallel_steps=[
                    Step(id="bad", type=StepType.TOOL, tool="fail"),
                ],
            ),
            Step(id="next", type=StepType.TOOL, tool="fail"),
        ],
    )
    trace = await vm.run(program)
    assert trace.status == TraceStatus.FAILED
    step_ids = [s.step_id for s in trace.steps]
    assert "next" not in step_ids


# ---------------------------------------------------------------------------
# Execution: on_error=SKIP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_skip_on_sub_error():
    vm = make_vm({"echo": tool_echo, "fail": tool_fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                on_error=OnError.SKIP,
                parallel_steps=[
                    Step(id="ok", type=StepType.TOOL, tool="echo", args={"value": "good"}),
                    Step(id="bad", type=StepType.TOOL, tool="fail"),
                ],
            )
        ],
    )
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    parent = next(s for s in trace.steps if s.step_id == "par")
    assert parent.status == StepStatus.SUCCESS
    assert parent.output == {"ok": "echo:good"}

    bad_result = next(s for s in trace.steps if s.step_id == "bad")
    assert bad_result.status == StepStatus.SKIPPED


# ---------------------------------------------------------------------------
# Trace structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_sub_steps_appear_before_parent_in_trace():
    vm = make_vm({"echo": tool_echo})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                parallel_steps=[
                    Step(id="x1", type=StepType.TOOL, tool="echo", args={"value": "1"}),
                    Step(id="x2", type=StepType.TOOL, tool="echo", args={"value": "2"}),
                ],
            )
        ],
    )
    trace = await vm.run(program)
    ids = [s.step_id for s in trace.steps]
    par_idx = ids.index("par")
    x1_idx = ids.index("x1")
    x2_idx = ids.index("x2")
    assert x1_idx < par_idx
    assert x2_idx < par_idx


@pytest.mark.asyncio
async def test_parallel_mixed_with_sequential():
    vm = make_vm({"echo": tool_echo})
    program = Program(
        name="test",
        steps=[
            Step(id="before", type=StepType.TOOL, tool="echo", args={"value": "pre"}),
            Step(
                id="par",
                type=StepType.PARALLEL,
                parallel_steps=[
                    Step(id="p1", type=StepType.TOOL, tool="echo", args={"value": "1"}),
                    Step(id="p2", type=StepType.TOOL, tool="echo", args={"value": "2"}),
                ],
            ),
            Step(id="after", type=StepType.TOOL, tool="echo", args={"value": "post"}),
        ],
    )
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    step_ids = [s.step_id for s in trace.steps]
    assert step_ids[0] == "before"
    assert "p1" in step_ids
    assert "p2" in step_ids
    assert step_ids[-1] == "after"
