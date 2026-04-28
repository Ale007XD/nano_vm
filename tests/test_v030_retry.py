"""
tests/test_v030_retry.py
========================
v0.3.0 — retry policy: exponential backoff, max_retries default, exhaustion.

Покрывает:
  R1 — max_retries default изменён на 3 (1 initial + 2 retry)
  R2 — retry: успех на N-й попытке → StepResult.retries == N-1
  R3 — retry исчерпан → TraceStatus.FAILED
  R4 — exponential backoff: задержки растут (0s, 1s, 2s, …)
  R5 — on_error=FAIL (default) — retry не применяется
  R6 — on_error=SKIP — retry не применяется, шаг SKIPPED
"""

from __future__ import annotations

import asyncio

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


def make_retry_program(tool_name: str, max_retries: int | None = None) -> Program:
    step_kwargs: dict = {
        "id": "s1",
        "type": StepType.TOOL,
        "tool": tool_name,
        "args": {},
        "on_error": OnError.RETRY,
    }
    if max_retries is not None:
        step_kwargs["max_retries"] = max_retries
    return Program(name="retry_test", steps=[Step(**step_kwargs)])


# ---------------------------------------------------------------------------
# R1 — max_retries default == 3
# ---------------------------------------------------------------------------


def test_max_retries_default_is_3():
    step = Step(id="s", type=StepType.LLM, prompt="hi")
    assert step.max_retries == 3


# ---------------------------------------------------------------------------
# R2 — успех на 3-й попытке → retries == 2
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_success_on_third_attempt():
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("not yet")
        return "finally"

    vm = make_vm({"flaky": flaky})
    program = make_retry_program("flaky", max_retries=3)
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    assert trace.final_output == "finally"
    assert call_count == 3
    assert trace.steps[0].retries == 2


# ---------------------------------------------------------------------------
# R3 — retry исчерпан → FAILED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_exhausted_produces_failed_trace():
    call_count = 0

    async def always_fail():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("permanent failure")

    vm = make_vm({"fail": always_fail})
    program = make_retry_program("fail", max_retries=3)
    trace = await vm.run(program)

    assert trace.status == TraceStatus.FAILED
    assert call_count == 3  # 1 initial + 2 retries
    assert "permanent failure" in trace.error


# ---------------------------------------------------------------------------
# R4 — exponential backoff: задержки 1s → 2s → 4s (проверяем через mock sleep)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_exponential_backoff_delays(monkeypatch):
    """Проверяет, что asyncio.sleep вызывается с экспоненциально растущими задержками."""
    sleep_calls: list[float] = []

    async def mock_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    call_count = 0

    async def fail_twice():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("retry me")
        return "done"

    vm = make_vm({"fail_twice": fail_twice})
    program = make_retry_program("fail_twice", max_retries=3)
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    # attempt=1 → sleep(min(2^0, 30)) = 1.0
    # attempt=2 → sleep(min(2^1, 30)) = 2.0
    assert sleep_calls == [1.0, 2.0]


# ---------------------------------------------------------------------------
# R4b — backoff cap at 30s
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_backoff_capped_at_30(monkeypatch):
    """При attempt=6 backoff должен быть 30, не 64."""
    sleep_calls: list[float] = []

    async def mock_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    call_count = 0

    async def fail_many():
        nonlocal call_count
        call_count += 1
        if call_count < 7:
            raise RuntimeError("fail")
        return "done"

    vm = make_vm({"fail_many": fail_many})
    program = make_retry_program("fail_many", max_retries=7)
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    # attempt 5 → min(2^4, 30)=16; attempt 6 → min(2^5, 30)=30
    assert all(d <= 30.0 for d in sleep_calls)
    assert sleep_calls[-1] == 30.0


# ---------------------------------------------------------------------------
# R5 — on_error=FAIL не делает retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_retry_on_error_fail():
    call_count = 0

    async def fail():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("fail")

    vm = make_vm({"fail": fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="s1",
                type=StepType.TOOL,
                tool="fail",
                args={},
                on_error=OnError.FAIL,
                max_retries=3,
            )
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.FAILED
    assert call_count == 1  # ровно одна попытка


# ---------------------------------------------------------------------------
# R6 — on_error=SKIP не делает retry, шаг SKIPPED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_retry_on_error_skip():
    call_count = 0

    async def fail():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("fail")

    vm = make_vm({"fail": fail})
    program = Program(
        name="test",
        steps=[
            Step(
                id="s1",
                type=StepType.TOOL,
                tool="fail",
                args={},
                on_error=OnError.SKIP,
                max_retries=3,
            ),
            Step(id="s2", type=StepType.LLM, prompt="continue"),
        ],
    )
    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUCCESS
    assert call_count == 1  # SKIP: одна попытка, без retry
    assert trace.steps[0].status == StepStatus.SKIPPED


# ---------------------------------------------------------------------------
# R7 — retry счётчик сбрасывается между разными запусками программы
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_counter_resets_between_runs():
    """Каждый vm.run — независимый запуск, счётчик attempt начинается с 0."""
    run_calls: list[int] = []

    async def succeed_immediately():
        run_calls.append(1)
        return "ok"

    vm = make_vm({"ok": succeed_immediately})
    program = make_retry_program("ok", max_retries=3)

    for _ in range(3):
        trace = await vm.run(program)
        assert trace.status == TraceStatus.SUCCESS

    # Каждый запуск — ровно 1 вызов (нет накопленных attempt из прошлых runs)
    assert len(run_calls) == 3
