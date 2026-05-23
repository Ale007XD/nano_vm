"""
tests/test_v081.py
==================
v0.8.1 bugfix regression tests.

BF-01  async tool корректно определяется через inspect.iscoroutinefunction
       (замена deprecated asyncio.iscoroutinefunction, Python 3.14+/3.16)
BF-02  sync tool по-прежнему работает корректно (regression)
BF-03  mixed: async + sync tools в одной программе
"""

from __future__ import annotations

import pytest

from nano_vm.adapters.mock_adapter import MockLLMAdapter
from nano_vm.models import Program, TraceStatus
from nano_vm.vm import ExecutionVM

_llm = MockLLMAdapter("ok")  # required positional arg; not used in tool-only programs


# ---------------------------------------------------------------------------
# BF-01: async tool корректно определяется и await-ится
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bf_01_async_tool_awaited_correctly() -> None:
    called: list[bool] = []

    async def async_tool(**kwargs: object) -> str:
        called.append(True)
        return "async_result"

    vm = ExecutionVM(llm=_llm, tools={"async_tool": async_tool})
    program = Program.from_dict({
        "name": "test",
        "steps": [{"id": "s1", "type": "tool", "tool": "async_tool"}],
    })
    trace = await vm.run(program, context={})

    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "async_result"
    assert called, "async tool was never called"


# ---------------------------------------------------------------------------
# BF-02: sync tool по-прежнему работает (regression)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bf_02_sync_tool_regression() -> None:
    def sync_tool(**kwargs: object) -> str:
        return "sync_result"

    vm = ExecutionVM(llm=_llm, tools={"sync_tool": sync_tool})
    program = Program.from_dict({
        "name": "test",
        "steps": [{"id": "s1", "type": "tool", "tool": "sync_tool"}],
    })
    trace = await vm.run(program, context={})

    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "sync_result"


# ---------------------------------------------------------------------------
# BF-03: async + sync tools в одной программе
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bf_03_mixed_async_sync_tools() -> None:
    async def fetch(**kwargs: object) -> str:
        return "fetched"

    def process(**kwargs: object) -> str:
        return "processed"

    vm = ExecutionVM(llm=_llm, tools={"fetch": fetch, "process": process})
    program = Program.from_dict({
        "name": "test",
        "steps": [
            {"id": "s1", "type": "tool", "tool": "fetch"},
            {"id": "s2", "type": "tool", "tool": "process"},
        ],
    })
    trace = await vm.run(program, context={})

    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "fetched"
    assert trace.steps[1].output == "processed"
