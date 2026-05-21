"""
tests/test_v080_sprint5_allowed.py
====================================
Sprint 5, Task 2: Step.allowed_outputs — LLM output validation (v0.8.0).

Контракт репо:
  - allowed_outputs: list[str] | None = None
  - on_error=fail   → VMError с именем шага и реальным выводом
  - on_error=skip   → output = allowed_outputs[0]  (первый элемент = fallback sentinel)
  - on_error=retry  → retry loop до max_retries; VMError если исчерпаны
  - allowed_outputs=None → валидация отсутствует (любой вывод OK)
  - allowed_outputs=[]   → ValidationError при построении Program (запрещено)
  - allowed_outputs только для llm-шагов (ValidationError на tool/condition)

AO-01  output в allowed_outputs → SUCCESS, output stripped и сохранён
AO-02  output НЕ в allowed_outputs + on_error=fail → FAILED, VMError содержит step.id и вывод
AO-03  output НЕ в allowed_outputs + on_error=skip → output=allowed_outputs[0], FSM SUCCESS
AO-04  allowed_outputs=None → любой вывод принимается, regression
AO-05  allowed_outputs=[] → ValidationError при построении Program
AO-06  allowed_outputs на tool-шаге → ValidationError при построении Program
AO-07  on_error=retry, LLM всегда неверный → FAILED после max_retries
AO-08  regression: шаг без allowed_outputs рядом с шагом с allowed_outputs — оба корректны
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nano_vm.adapters.mock_adapter import MockLLMAdapter
from nano_vm.models import Program, TraceStatus
from nano_vm.vm import ExecutionVM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(steps: list[dict]) -> dict:
    return {"name": "test", "steps": steps}


# ---------------------------------------------------------------------------
# AO-01: output в allowed_outputs → SUCCESS, output stripped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ao_01_output_in_allowed():
    vm = ExecutionVM(llm=MockLLMAdapter("refund"))
    program = Program.from_dict(
        _prog(
            [
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "classify: $input",
                    "output_key": "decision",
                    "allowed_outputs": ["refund", "query", "other"],
                },
            ]
        )
    )
    trace = await vm.run(program, context={"input": "I want a refund"})

    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "refund"
    assert trace.steps[0].error is None


# ---------------------------------------------------------------------------
# AO-02: output НЕ в allowed_outputs + on_error=fail → FAILED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ao_02_not_in_allowed_fail():
    vm = ExecutionVM(llm=MockLLMAdapter("REFUND"))  # wrong case
    program = Program.from_dict(
        _prog(
            [
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "classify: $input",
                    "output_key": "decision",
                    "allowed_outputs": ["refund", "query", "other"],
                    "on_error": "fail",
                },
            ]
        )
    )
    trace = await vm.run(program, context={"input": "test"})

    assert trace.status == TraceStatus.FAILED
    assert trace.error is not None
    assert "classify" in trace.error
    assert "REFUND" in trace.error


# ---------------------------------------------------------------------------
# AO-03: output НЕ в allowed_outputs + on_error=skip → output=allowed_outputs[0]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ao_03_not_in_allowed_skip_returns_first():
    vm = ExecutionVM(llm=MockLLMAdapter("UNEXPECTED"))
    program = Program.from_dict(
        _prog(
            [
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "classify: $input",
                    "output_key": "decision",
                    "allowed_outputs": ["other", "refund", "query"],
                    "on_error": "skip",
                },
            ]
        )
    )
    trace = await vm.run(program, context={"input": "test"})

    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "other"  # allowed_outputs[0]


# ---------------------------------------------------------------------------
# AO-04: allowed_outputs=None → любой вывод принимается, regression
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ao_04_allowed_outputs_none_regression():
    vm = ExecutionVM(llm=MockLLMAdapter("anything goes 123"))
    program = Program.from_dict(
        _prog(
            [
                {"id": "s1", "type": "llm", "prompt": "go", "output_key": "out"},
            ]
        )
    )
    trace = await vm.run(program, context={})

    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "anything goes 123"


# ---------------------------------------------------------------------------
# AO-05: allowed_outputs=[] → ValidationError при построении Program
# ---------------------------------------------------------------------------


def test_ao_05_empty_allowed_outputs_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        Program.from_dict(
            _prog(
                [
                    {
                        "id": "classify",
                        "type": "llm",
                        "prompt": "go",
                        "allowed_outputs": [],
                    },
                ]
            )
        )
    assert "allowed_outputs" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# AO-06: allowed_outputs на tool-шаге → ValidationError
# ---------------------------------------------------------------------------


def test_ao_06_allowed_outputs_on_tool_step_validation_error():
    with pytest.raises(ValidationError) as exc_info:
        Program.from_dict(
            _prog(
                [
                    {
                        "id": "act",
                        "type": "tool",
                        "tool": "some_tool",
                        "allowed_outputs": ["ok", "fail"],
                    },
                ]
            )
        )
    assert "allowed_outputs" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# AO-07: on_error=retry, LLM всегда возвращает неверный вывод → FAILED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ao_07_retry_exhausted():
    vm = ExecutionVM(llm=MockLLMAdapter("WRONG"))
    program = Program.from_dict(
        _prog(
            [
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "classify: $input",
                    "output_key": "decision",
                    "allowed_outputs": ["refund", "query", "other"],
                    "on_error": "retry",
                    "max_retries": 2,
                },
            ]
        )
    )
    trace = await vm.run(program, context={"input": "test"})

    assert trace.status == TraceStatus.FAILED
    assert trace.error is not None
    assert "classify" in trace.error


# ---------------------------------------------------------------------------
# AO-08: regression — шаг без allowed_outputs рядом с шагом с allowed_outputs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ao_08_mixed_steps_regression():
    vm = ExecutionVM(
        llm=MockLLMAdapter(["free text", "refund"]),
        tools={"act": lambda **kw: "done"},
    )
    program = Program.from_dict(
        _prog(
            [
                # шаг без allowed_outputs — любой вывод OK
                {
                    "id": "summarize",
                    "type": "llm",
                    "prompt": "summarize $input",
                    "output_key": "summary",
                },
                # шаг с allowed_outputs
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "classify $input",
                    "output_key": "decision",
                    "allowed_outputs": ["refund", "query", "other"],
                },
                {"id": "act", "type": "tool", "tool": "act"},
            ]
        )
    )
    trace = await vm.run(program, context={"input": "I want refund"})

    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "free text"
    assert trace.steps[1].output == "refund"
    assert trace.steps[2].output == "done"
