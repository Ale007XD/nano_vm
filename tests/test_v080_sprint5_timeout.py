"""
tests/test_v080_sprint5_timeout.py
===================================
Sprint 5, Task 3: Step.timeout_seconds — LLM step timeout (v0.8.0).

TO-01  timeout_seconds не задан → SUCCESS, regression
TO-02  timeout_seconds задан, LLM отвечает вовремя → SUCCESS
TO-03  LLM зависает + on_timeout=fail (default) → FAILED, VMError содержит step.id и таймаут
TO-04  LLM зависает + on_timeout=fallback + allowed_outputs задан → output=allowed_outputs[0]
TO-05  LLM зависает + on_timeout=fallback + allowed_outputs=None → output=''
TO-06  timeout + on_timeout=fallback → FSM продолжает выполнение следующего шага
TO-07  regression: allowed_outputs + timeout вместе — оба контракта соблюдаются
"""

from __future__ import annotations

import asyncio

import pytest

from nano_vm.adapters.mock_adapter import MockLLMAdapter
from nano_vm.models import Program, TraceStatus
from nano_vm.vm import ExecutionVM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SlowLLMAdapter:
    """Имитирует зависший LLM."""

    async def complete(self, messages: list[dict]) -> str:
        await asyncio.sleep(60)
        return "never"


def _prog(steps: list[dict]) -> dict:
    return {"name": "test", "steps": steps}


# ---------------------------------------------------------------------------
# TO-01: timeout_seconds не задан → regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_01_no_timeout_regression():
    vm = ExecutionVM(llm=MockLLMAdapter("ok"))
    program = Program.from_dict(
        _prog(
            [
                {"id": "s1", "type": "llm", "prompt": "go", "output_key": "out"},
            ]
        )
    )
    trace = await vm.run(program, context={})

    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "ok"


# ---------------------------------------------------------------------------
# TO-02: timeout задан, LLM отвечает вовремя → SUCCESS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_02_timeout_not_triggered():
    vm = ExecutionVM(llm=MockLLMAdapter("refund"))
    program = Program.from_dict(
        _prog(
            [
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "classify $input",
                    "output_key": "decision",
                    "allowed_outputs": ["refund", "query", "other"],
                    "timeout_seconds": 5.0,
                },
            ]
        )
    )
    trace = await vm.run(program, context={"input": "test"})

    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "refund"


# ---------------------------------------------------------------------------
# TO-03: LLM зависает + on_timeout=fail → FAILED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_03_timeout_fail():
    vm = ExecutionVM(llm=_SlowLLMAdapter())
    program = Program.from_dict(
        _prog(
            [
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "classify $input",
                    "output_key": "decision",
                    "timeout_seconds": 0.05,
                    "on_timeout": "fail",
                },
            ]
        )
    )
    trace = await vm.run(program, context={"input": "test"})

    assert trace.status == TraceStatus.FAILED
    assert trace.error is not None
    assert "classify" in trace.error
    assert "timed out" in trace.error
    assert "0.05" in trace.error


# ---------------------------------------------------------------------------
# TO-04: LLM зависает + on_timeout=fallback + allowed_outputs → output=allowed_outputs[0]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_04_timeout_fallback_with_allowed_outputs():
    vm = ExecutionVM(llm=_SlowLLMAdapter())
    program = Program.from_dict(
        _prog(
            [
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "classify $input",
                    "output_key": "decision",
                    "allowed_outputs": ["other", "refund", "query"],
                    "timeout_seconds": 0.05,
                    "on_timeout": "fallback",
                },
            ]
        )
    )
    trace = await vm.run(program, context={"input": "test"})

    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "other"  # allowed_outputs[0]


# ---------------------------------------------------------------------------
# TO-05: LLM зависает + on_timeout=fallback + allowed_outputs=None → output=''
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_05_timeout_fallback_no_allowed_outputs():
    vm = ExecutionVM(llm=_SlowLLMAdapter())
    program = Program.from_dict(
        _prog(
            [
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "classify $input",
                    "output_key": "decision",
                    "timeout_seconds": 0.05,
                    "on_timeout": "fallback",
                },
            ]
        )
    )
    trace = await vm.run(program, context={"input": "test"})

    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == ""


# ---------------------------------------------------------------------------
# TO-06: timeout + fallback → FSM продолжает следующий шаг
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_06_timeout_fallback_fsm_continues():
    vm = ExecutionVM(
        llm=_SlowLLMAdapter(),
        tools={"act": lambda **kw: "done"},
    )
    program = Program.from_dict(
        _prog(
            [
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "classify $input",
                    "output_key": "decision",
                    "allowed_outputs": ["other", "refund"],
                    "timeout_seconds": 0.05,
                    "on_timeout": "fallback",
                },
                {"id": "act", "type": "tool", "tool": "act"},
            ]
        )
    )
    trace = await vm.run(program, context={"input": "test"})

    assert trace.status == TraceStatus.SUCCESS
    assert len(trace.steps) == 2
    assert trace.steps[0].output == "other"
    assert trace.steps[1].output == "done"


# ---------------------------------------------------------------------------
# TO-07: allowed_outputs + timeout вместе — оба контракта соблюдаются
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_to_07_allowed_outputs_and_timeout_together():
    """LLM отвечает вовремя, но вывод не в allowed_outputs → on_error=skip → fallback."""
    vm = ExecutionVM(llm=MockLLMAdapter("UNEXPECTED"))
    program = Program.from_dict(
        _prog(
            [
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "classify $input",
                    "output_key": "decision",
                    "allowed_outputs": ["refund", "query", "other"],
                    "on_error": "skip",
                    "timeout_seconds": 5.0,
                    "on_timeout": "fail",
                },
            ]
        )
    )
    trace = await vm.run(program, context={"input": "test"})

    # timeout не сработал (LLM ответил быстро)
    # allowed_outputs нарушен + on_error=skip → allowed_outputs[0]
    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "refund"
