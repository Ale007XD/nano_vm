"""
tests/test_v030_concurrency.py
==============================
v0.3.0 — max_concurrency per parallel block.

Покрывает:
  C1 — max_concurrency=None (default): все sub-steps стартуют одновременно
  C2 — max_concurrency=1: sub-steps выполняются строго по одному
  C3 — max_concurrency=N < len(parallel_steps): одновременно не более N
  C4 — max_concurrency >= len(parallel_steps): поведение идентично None
  C5 — max_concurrency=1 с on_error=SKIP: skip работает корректно
  C6 — DSL round-trip: max_concurrency сериализуется / десериализуется через Program.from_dict
"""

from __future__ import annotations

import asyncio
import time

import pytest

from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.models import OnError, Step, StepStatus, StepType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLLM:
    async def complete(self, messages):
        return "ok"


def make_vm(tools: dict | None = None) -> ExecutionVM:
    return ExecutionVM(llm=MockLLM(), tools=tools or {})


# ---------------------------------------------------------------------------
# C1 — max_concurrency=None: все стартуют одновременно
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_concurrency_cap_all_start_simultaneously():
    """Без ограничений 5 slow-tools выполняются параллельно за ~одно время."""
    DELAY = 0.05

    async def slow(**_):
        await asyncio.sleep(DELAY)
        return "done"

    vm = make_vm({"slow": slow})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                # max_concurrency не задан → None
                parallel_steps=[
                    Step(id=f"s{i}", type=StepType.TOOL, tool="slow")
                    for i in range(5)
                ],
            )
        ],
    )
    t0 = time.perf_counter()
    trace = await vm.run(program)
    elapsed = time.perf_counter() - t0

    assert trace.status == TraceStatus.SUCCESS
    # Параллельно: ~DELAY, не 5*DELAY
    assert elapsed < DELAY * 2.5


# ---------------------------------------------------------------------------
# C2 — max_concurrency=1: строго последовательно
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_concurrency_1_sequential_order():
    """max_concurrency=1 → sub-steps не пересекаются по времени."""
    active: list[int] = []
    max_active = [0]

    async def tracked(idx: int) -> str:
        active.append(idx)
        max_active[0] = max(max_active[0], len(active))
        await asyncio.sleep(0.02)
        active.remove(idx)
        return f"done_{idx}"

    async def make_tracked(i: int):
        return await tracked(i)

    vm = make_vm({f"t{i}": (lambda i=i: make_tracked(i)) for i in range(4)})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                max_concurrency=1,
                parallel_steps=[
                    Step(id=f"s{i}", type=StepType.TOOL, tool=f"t{i}")
                    for i in range(4)
                ],
            )
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    # В любой момент времени активен не более 1 sub-step
    assert max_active[0] == 1


# ---------------------------------------------------------------------------
# C3 — max_concurrency=2: одновременно не более 2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_concurrency_2_limits_active():
    active: list[int] = []
    max_active = [0]

    async def tracked(idx: int) -> str:
        active.append(idx)
        max_active[0] = max(max_active[0], len(active))
        await asyncio.sleep(0.03)
        active.remove(idx)
        return f"r{idx}"

    N = 6
    vm = make_vm({f"t{i}": (lambda i=i: tracked(i)) for i in range(N)})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                max_concurrency=2,
                parallel_steps=[
                    Step(id=f"s{i}", type=StepType.TOOL, tool=f"t{i}")
                    for i in range(N)
                ],
            )
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    assert max_active[0] <= 2


# ---------------------------------------------------------------------------
# C4 — max_concurrency >= len(parallel_steps): поведение как без ограничений
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_concurrency_exceeds_steps_count():
    """max_concurrency=100 при 3 sub-steps — все стартуют сразу."""
    DELAY = 0.05

    async def slow():
        await asyncio.sleep(DELAY)
        return "ok"

    vm = make_vm({"slow": slow})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                max_concurrency=100,
                parallel_steps=[
                    Step(id=f"s{i}", type=StepType.TOOL, tool="slow")
                    for i in range(3)
                ],
            )
        ],
    )
    t0 = time.perf_counter()
    trace = await vm.run(program)
    elapsed = time.perf_counter() - t0

    assert trace.status == TraceStatus.SUCCESS
    assert elapsed < DELAY * 2.5


# ---------------------------------------------------------------------------
# C5 — max_concurrency=1 + on_error=SKIP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_concurrency_1_with_skip():
    async def ok() -> str:
        return "good"

    async def fail() -> str:
        raise RuntimeError("intentional")

    vm = make_vm({"ok": ok, "fail": fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                max_concurrency=1,
                on_error=OnError.SKIP,
                parallel_steps=[
                    Step(id="s1", type=StepType.TOOL, tool="ok"),
                    Step(id="s2", type=StepType.TOOL, tool="fail"),
                    Step(id="s3", type=StepType.TOOL, tool="ok"),
                ],
            )
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    parent = next(s for s in trace.steps if s.step_id == "par")
    assert parent.output["s1"] == "good"
    assert parent.output["s2"] is None  # SKIPPED → None
    assert parent.output["s3"] == "good"

    s2 = next(s for s in trace.steps if s.step_id == "s2")
    assert s2.status == StepStatus.SKIPPED


# ---------------------------------------------------------------------------
# C6 — DSL round-trip через Program.from_dict
# ---------------------------------------------------------------------------


def test_max_concurrency_from_dict():
    program = Program.from_dict(
        {
            "name": "test",
            "steps": [
                {
                    "id": "par",
                    "type": "parallel",
                    "max_concurrency": 3,
                    "parallel_steps": [
                        {"id": "s1", "type": "tool", "tool": "x"},
                        {"id": "s2", "type": "tool", "tool": "x"},
                    ],
                }
            ],
        }
    )
    par_step = program.steps[0]
    assert par_step.max_concurrency == 3


def test_max_concurrency_default_is_none():
    step = Step(
        id="par",
        type=StepType.PARALLEL,
        parallel_steps=[Step(id="s1", type=StepType.TOOL, tool="x")],
    )
    assert step.max_concurrency is None


def test_max_concurrency_none_from_dict():
    program = Program.from_dict(
        {
            "name": "test",
            "steps": [
                {
                    "id": "par",
                    "type": "parallel",
                    # max_concurrency не задан
                    "parallel_steps": [
                        {"id": "s1", "type": "tool", "tool": "x"},
                    ],
                }
            ],
        }
    )
    assert program.steps[0].max_concurrency is None
