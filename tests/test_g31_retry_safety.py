"""Tests for G-31 retry safety: parallel retry validator ERROR, read-only runtime
context ($__execution_id__, $__step_id__, $__attempt__) via ContextVar, and
Step.idempotent validation/warning.
"""

from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nano_vm import (
    ExecutionVM,
    InMemoryCursorRepository,
    IssueKind,
    IssueSeverity,
    Program,
    ProgramValidationError,
    ProgramValidator,
    StateContext,
    Step,
    StepResult,
    StepType,
    Trace,
    TraceAnalyzer,
    TraceStatus,
    VMError,
    WebhookEvent,
)
from nano_vm.adapters import MockLLMAdapter
from nano_vm.vm import _CURRENT_TRACE_ID, _RUNTIME_CONTEXT_KEYS


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    real_sleep = asyncio.sleep

    async def _fast_sleep(_delay: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr("nano_vm.vm.asyncio.sleep", _fast_sleep)


# ---------------------------------------------------------------------------
# 1. Validator: ERROR for on_error=retry on parallel
# ---------------------------------------------------------------------------


def test_validator_rejects_retry_on_parallel() -> None:
    program = Program.from_dict(
        {
            "name": "par_retry",
            "steps": [
                {
                    "id": "fanout",
                    "type": "parallel",
                    "on_error": "retry",
                    "parallel_steps": [
                        {"id": "a", "type": "tool", "tool": "ta"},
                        {"id": "b", "type": "tool", "tool": "tb"},
                    ],
                }
            ],
        }
    )
    report = ProgramValidator(program).validate()
    assert not report.is_valid()
    issues = report.by_kind(IssueKind.RETRY_ON_PARALLEL)
    assert len(issues) == 1
    assert issues[0].step_id == "fanout"
    assert issues[0].severity == IssueSeverity.ERROR


@pytest.mark.asyncio
async def test_vm_run_with_validate_rejects_retry_on_parallel() -> None:
    program = Program.from_dict(
        {
            "name": "par_retry_vm",
            "steps": [
                {
                    "id": "fanout",
                    "type": "parallel",
                    "on_error": "retry",
                    "parallel_steps": [
                        {"id": "a", "type": "tool", "tool": "ta"},
                    ],
                }
            ],
        }
    )
    called = False

    def ta() -> str:
        nonlocal called
        called = True
        return "ok"

    vm = ExecutionVM(tools={"ta": ta})
    with pytest.raises(ProgramValidationError) as exc_info:
        await vm.run(program, validate=True)

    assert not called
    assert exc_info.value.report.by_kind(IssueKind.RETRY_ON_PARALLEL)


# ---------------------------------------------------------------------------
# 2. Read-only runtime context: $__execution_id__, $__step_id__, $__attempt__
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_context_passed_via_dollar_and_increments_on_retry(
    no_sleep: None,
) -> None:
    seen: list[dict[str, Any]] = []

    def flaky(
        exec_id: str,
        step_id: str,
        attempt: int,
        idem_key: str,
    ) -> str:
        seen.append(
            {
                "exec_id": exec_id,
                "step_id": step_id,
                "attempt": attempt,
                "idem_key": idem_key,
            }
        )
        if attempt < 3:
            raise RuntimeError("transient")
        return "done"

    program = Program.from_dict(
        {
            "name": "ctx_retry",
            "steps": [
                {
                    "id": "charge",
                    "type": "tool",
                    "tool": "flaky",
                    "idempotent": True,
                    "on_error": "retry",
                    "max_retries": 3,
                    "args": {
                        "exec_id": "$__execution_id__",
                        "step_id": "$__step_id__",
                        "attempt": "$__attempt__",
                        "idem_key": "$__execution_id__:$__step_id__",
                    },
                }
            ],
        }
    )

    trace = await ExecutionVM(tools={"flaky": flaky}).run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert [row["attempt"] for row in seen] == [1, 2, 3]
    assert all(isinstance(row["attempt"], int) for row in seen)
    assert all(row["exec_id"] == trace.trace_id for row in seen)
    assert all(row["step_id"] == "charge" for row in seen)
    assert all(row["idem_key"] == f"{trace.trace_id}:charge" for row in seen)


@pytest.mark.asyncio
async def test_runtime_context_no_implicit_kwargs() -> None:
    def strict_tool(x: int) -> int:
        return x + 1

    program = Program.from_dict(
        {
            "name": "strict_sig",
            "steps": [
                {"id": "inc", "type": "tool", "tool": "strict_tool", "args": {"x": 5}},
            ],
        }
    )
    trace = await ExecutionVM(tools={"strict_tool": strict_tool}).run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert trace.final_output == 6


@pytest.mark.asyncio
async def test_runtime_context_is_read_only_against_user_context() -> None:
    observed: dict[str, Any] = {}

    def inspect_ctx(exec_id: str, step_id: str, attempt: int) -> str:
        observed["exec_id"] = exec_id
        observed["step_id"] = step_id
        observed["attempt"] = attempt
        return "ok"

    program = Program.from_dict(
        {
            "name": "ro_ctx",
            "steps": [
                {
                    "id": "real_step",
                    "type": "tool",
                    "tool": "inspect_ctx",
                    "args": {
                        "exec_id": "$__execution_id__",
                        "step_id": "$__step_id__",
                        "attempt": "$__attempt__",
                    },
                }
            ],
        }
    )

    trace = await ExecutionVM(tools={"inspect_ctx": inspect_ctx}).run(
        program,
        context={
            "__execution_id__": "spoofed-trace",
            "__step_id__": "spoofed-step",
            "__attempt__": 999,
        },
    )
    assert trace.status == TraceStatus.SUCCESS
    assert observed == {
        "exec_id": trace.trace_id,
        "step_id": "real_step",
        "attempt": 1,
    }


@pytest.mark.asyncio
async def test_runtime_context_not_shadowed_by_step_outputs() -> None:
    observed: dict[str, Any] = {}

    def produce_shadow() -> str:
        return "shadow"

    def consume(exec_id: str, step_id: str, attempt: int, composite: str) -> str:
        observed["exec_id"] = exec_id
        observed["step_id"] = step_id
        observed["attempt"] = attempt
        observed["composite"] = composite
        return "ok"

    program = Program.from_dict(
        {
            "name": "ro_ctx_step_outputs",
            "steps": [
                {
                    "id": "__step_id__",
                    "type": "tool",
                    "tool": "produce_shadow",
                    "output_key": "__step_id__",
                },
                {
                    "id": "__attempt__",
                    "type": "tool",
                    "tool": "produce_shadow",
                    "output_key": "__attempt__",
                },
                {
                    "id": "__execution_id__",
                    "type": "tool",
                    "tool": "produce_shadow",
                    "output_key": "__execution_id__",
                },
                {
                    "id": "consumer",
                    "type": "tool",
                    "tool": "consume",
                    "args": {
                        "exec_id": "$__execution_id__",
                        "step_id": "$__step_id__",
                        "attempt": "$__attempt__",
                        "composite": "$__execution_id__:$__step_id__:$__attempt__",
                    },
                },
            ],
        }
    )

    vm = ExecutionVM(tools={"produce_shadow": produce_shadow, "consume": consume})
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert observed == {
        "exec_id": trace.trace_id,
        "step_id": "consumer",
        "attempt": 1,
        "composite": f"{trace.trace_id}:consumer:1",
    }


@pytest.mark.asyncio
async def test_runtime_context_excluded_from_fingerprint_and_trace_hash(
    no_sleep: None,
) -> None:
    attempts = 0

    def sometimes_fails(idem_key: str, attempt: int) -> dict[str, str]:
        nonlocal attempts
        attempts += 1
        assert idem_key  # consumed, not echoed into output
        if attempts == 1:
            raise RuntimeError("boom once")
        return {"status": "ok"}

    def always_ok(idem_key: str, attempt: int) -> dict[str, str]:
        assert idem_key and attempt == 1
        return {"status": "ok"}

    program = Program.from_dict(
        {
            "name": "hash_stability",
            "steps": [
                {
                    "id": "work",
                    "type": "tool",
                    "tool": "work_fn",
                    "idempotent": True,
                    "on_error": "retry",
                    "max_retries": 3,
                    "args": {
                        "idem_key": "$__execution_id__:$__step_id__",
                        "attempt": "$__attempt__",
                    },
                }
            ],
        }
    )

    trace_retried = await ExecutionVM(tools={"work_fn": sometimes_fails}).run(program)
    trace_clean = await ExecutionVM(tools={"work_fn": always_ok}).run(program)

    assert trace_retried.trace_id != trace_clean.trace_id
    assert trace_retried.steps[0].retries == 1
    assert trace_clean.steps[0].retries == 0
    assert trace_retried.state_snapshots == trace_clean.state_snapshots
    assert (
        TraceAnalyzer(trace_retried).receipt().trace_hash
        == TraceAnalyzer(trace_clean).receipt().trace_hash
    )


@pytest.mark.asyncio
async def test_runtime_context_preserved_across_suspend_resume_and_parallel() -> None:
    seen: list[tuple[str, str, int, str | None]] = []

    def pause(exec_id: str, step_id: str, attempt: int) -> str:
        seen.append((exec_id, step_id, attempt, _CURRENT_TRACE_ID.get()))
        return "PENDING"

    async def branch(exec_id: str, step_id: str, attempt: int) -> str:
        await asyncio.sleep(0)
        seen.append((exec_id, step_id, attempt, _CURRENT_TRACE_ID.get()))
        return "ok"

    program = Program.from_dict(
        {
            "name": "suspend_par_ctx",
            "steps": [
                {
                    "id": "wait_hook",
                    "type": "tool",
                    "tool": "pause",
                    "args": {
                        "exec_id": "$__execution_id__",
                        "step_id": "$__step_id__",
                        "attempt": "$__attempt__",
                    },
                },
                {
                    "id": "fanout",
                    "type": "parallel",
                    "parallel_steps": [
                        {
                            "id": "b1",
                            "type": "tool",
                            "tool": "branch",
                            "args": {
                                "exec_id": "$__execution_id__",
                                "step_id": "$__step_id__",
                                "attempt": "$__attempt__",
                            },
                        },
                        {
                            "id": "b2",
                            "type": "tool",
                            "tool": "branch",
                            "args": {
                                "exec_id": "$__execution_id__",
                                "step_id": "$__step_id__",
                                "attempt": "$__attempt__",
                            },
                        },
                    ],
                },
            ],
        }
    )

    vm = ExecutionVM(tools={"pause": pause, "branch": branch})
    suspended = await vm.run(program)
    assert suspended.status == TraceStatus.SUSPENDED

    resumed = await vm.resume_with_program(
        WebhookEvent(trace_id=suspended.trace_id, payload={"ok": True}),
        program,
    )
    assert resumed.status == TraceStatus.SUCCESS
    assert seen[0] == (suspended.trace_id, "wait_hook", 1, suspended.trace_id)
    assert set(seen[1:]) == {
        (suspended.trace_id, "b1", 1, suspended.trace_id),
        (suspended.trace_id, "b2", 1, suspended.trace_id),
    }


@pytest.mark.asyncio
async def test_runtime_context_available_in_llm_prompt_and_condition() -> None:
    adapter = MockLLMAdapter("ok")
    program = Program.from_dict(
        {
            "name": "llm_cond_ctx",
            "steps": [
                {
                    "id": "ask",
                    "type": "llm",
                    "prompt": "step=$__step_id__ attempt=$__attempt__",
                    "output_key": "ans",
                },
                {
                    "id": "check",
                    "type": "condition",
                    "condition": "$__attempt__ == 1 and $__step_id__ == 'check'",
                    "then": "done",
                    "otherwise": "fail_step",
                },
                {"id": "done", "type": "tool", "tool": "ok_tool", "is_terminal": True},
                {"id": "fail_step", "type": "tool", "tool": "bad_tool", "is_terminal": True},
            ],
        }
    )
    trace = await ExecutionVM(
        llm=adapter,
        tools={"ok_tool": lambda: "good", "bad_tool": lambda: "bad"},
    ).run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert trace.final_output == "good"
    assert adapter.calls[0][-1]["content"] == "step=ask attempt=1"


# ---------------------------------------------------------------------------
# 3. ContextVar isolation, _run_step signature, reset & non-leakage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_step_subclass_hook_signature_preserved() -> None:
    """Subclasses overriding _run_step(self, step, state) must keep working."""

    class _CountingVM(ExecutionVM):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.counted: list[str] = []

        async def _run_step(
            self,
            step: Step,
            state: StateContext,
        ) -> tuple[StepResult, StateContext, list[StepResult]]:
            self.counted.append(step.id)
            return await super()._run_step(step, state)

    observed: list[tuple[str, str, int]] = []

    def record(exec_id: str, step_id: str, attempt: int) -> str:
        observed.append((exec_id, step_id, attempt))
        return "ok"

    program = Program.from_dict(
        {
            "name": "subclass_hook",
            "steps": [
                {
                    "id": "cond",
                    "type": "condition",
                    "condition": "$flag == True",
                    "then": "leaf",
                    "otherwise": "leaf",
                },
                {
                    "id": "leaf",
                    "type": "tool",
                    "tool": "record",
                    "args": {
                        "exec_id": "$__execution_id__",
                        "step_id": "$__step_id__",
                        "attempt": "$__attempt__",
                    },
                },
            ],
        }
    )

    vm = _CountingVM(tools={"record": record})
    trace = await vm.run(program, context={"flag": True})
    assert trace.status == TraceStatus.SUCCESS
    assert vm.counted == ["cond", "leaf"]
    assert observed == [(trace.trace_id, "leaf", 1)]


@pytest.mark.asyncio
async def test_concurrent_runs_on_same_vm_isolate_trace_id() -> None:
    started = 0
    both_started = asyncio.Event()
    observed: dict[str, list[tuple[str, str, int]]] = {"a": [], "b": []}

    async def slow_tool(tag: str, exec_id: str, step_id: str, attempt: int) -> str:
        nonlocal started
        observed[tag].append((exec_id, step_id, attempt))
        started += 1
        if started == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2.0)
        return f"{tag}:{step_id}"

    async def second_tool(tag: str, exec_id: str, step_id: str, attempt: int) -> str:
        observed[tag].append((exec_id, step_id, attempt))
        return f"{tag}:done"

    program = Program.from_dict(
        {
            "name": "concurrent_isolation",
            "steps": [
                {
                    "id": "s1",
                    "type": "tool",
                    "tool": "slow_tool",
                    "args": {
                        "tag": "$tag",
                        "exec_id": "$__execution_id__",
                        "step_id": "$__step_id__",
                        "attempt": "$__attempt__",
                    },
                },
                {
                    "id": "s2",
                    "type": "tool",
                    "tool": "second_tool",
                    "args": {
                        "tag": "$tag",
                        "exec_id": "$__execution_id__",
                        "step_id": "$__step_id__",
                        "attempt": "$__attempt__",
                    },
                },
            ],
        }
    )

    vm = ExecutionVM(tools={"slow_tool": slow_tool, "second_tool": second_tool})
    trace_a, trace_b = await asyncio.gather(
        vm.run(program, context={"tag": "a"}),
        vm.run(program, context={"tag": "b"}),
    )

    assert trace_a.status == TraceStatus.SUCCESS
    assert trace_b.status == TraceStatus.SUCCESS
    assert trace_a.trace_id != trace_b.trace_id
    assert observed["a"] == [
        (trace_a.trace_id, "s1", 1),
        (trace_a.trace_id, "s2", 1),
    ]
    assert observed["b"] == [
        (trace_b.trace_id, "s1", 1),
        (trace_b.trace_id, "s2", 1),
    ]


@pytest.mark.asyncio
async def test_direct_run_step_rejects_missing_or_empty_trace_id() -> None:
    def echo(k: str) -> str:
        return k

    step = Step(
        id="direct",
        type=StepType.TOOL,
        tool="echo",
        args={"k": "$__execution_id__"},
    )
    vm = ExecutionVM(tools={"echo": echo})

    assert _CURRENT_TRACE_ID.get() is None
    with pytest.raises(VMError, match="trace_id"):
        await vm._run_step(step, StateContext())

    tok = _CURRENT_TRACE_ID.set("")
    try:
        with pytest.raises(VMError, match="trace_id"):
            await vm._run_step(step, StateContext())
    finally:
        _CURRENT_TRACE_ID.reset(tok)


@pytest.mark.asyncio
async def test_trace_id_contextvar_reset_after_run_recursion_error_and_cancel() -> None:
    assert _CURRENT_TRACE_ID.get() is None

    # 1. Normal run + CONDITION -> CONDITION recursion
    seen: list[tuple[str, str]] = []

    def leaf(exec_id: str, step_id: str) -> str:
        seen.append((exec_id, step_id))
        return "reached"

    nested_prog = Program.from_dict(
        {
            "name": "nested_cond",
            "steps": [
                {
                    "id": "c1",
                    "type": "condition",
                    "condition": "$a == 1",
                    "then": "c2",
                    "otherwise": "leaf",
                },
                {
                    "id": "c2",
                    "type": "condition",
                    "condition": "$b == 2 and $__step_id__ == 'c2'",
                    "then": "leaf",
                    "otherwise": "other",
                },
                {
                    "id": "leaf",
                    "type": "tool",
                    "tool": "leaf",
                    "args": {"exec_id": "$__execution_id__", "step_id": "$__step_id__"},
                    "is_terminal": True,
                },
                {"id": "other", "type": "tool", "tool": "leaf", "is_terminal": True},
            ],
        }
    )
    vm = ExecutionVM(tools={"leaf": leaf})
    trace = await vm.run(nested_prog, context={"a": 1, "b": 2})
    assert trace.status == TraceStatus.SUCCESS
    assert seen == [(trace.trace_id, "leaf")]
    assert _CURRENT_TRACE_ID.get() is None

    # 2. Exception raised inside _execute_loop (verified inside same task)
    empty_trace = Trace(program_name="bad", trace_id="")
    simple_prog = Program.from_dict(
        {"name": "bad", "steps": [{"id": "s", "type": "tool", "tool": "leaf"}]}
    )
    observed_after_error: list[str | None] = []

    async def error_runner() -> None:
        try:
            await vm._execute_loop(simple_prog, StateContext(), empty_trace, start_step_id=None)
        finally:
            observed_after_error.append(_CURRENT_TRACE_ID.get())

    error_task = asyncio.create_task(error_runner())
    with pytest.raises(VMError):
        await error_task
    assert observed_after_error == [None]

    # 3. External cancellation (verified INSIDE the cancelled task's own context)
    entered = asyncio.Event()
    observed_after_cancel: list[str | None] = []

    async def hang() -> str:
        entered.set()
        await asyncio.sleep(60)
        return "never"

    cancel_vm = ExecutionVM(tools={"hang": hang})
    cancel_prog = Program.from_dict(
        {"name": "cancel_me", "steps": [{"id": "h", "type": "tool", "tool": "hang"}]}
    )

    async def cancel_runner() -> None:
        try:
            await cancel_vm.run(cancel_prog)
        finally:
            observed_after_cancel.append(_CURRENT_TRACE_ID.get())

    task = asyncio.create_task(cancel_runner())
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancel_vm.crash_trace is not None
    assert cancel_vm.crash_trace.status == TraceStatus.CANCELLED
    assert observed_after_cancel == [None]


@pytest.mark.asyncio
async def test_trace_id_contextvar_reset_on_cancel_inside_same_task() -> None:
    """Dedicated cancellation-path guard: verifies _CURRENT_TRACE_ID is reset
    inside the cancelled task itself (not merely in the caller's copied context).
    """
    entered = asyncio.Event()
    observed_in_task: list[str | None] = []

    async def hang() -> str:
        entered.set()
        await asyncio.sleep(60)
        return "never"

    cancel_vm = ExecutionVM(tools={"hang": hang})
    cancel_prog = Program.from_dict(
        {"name": "cancel_only", "steps": [{"id": "h", "type": "tool", "tool": "hang"}]}
    )

    async def runner() -> None:
        try:
            await cancel_vm.run(cancel_prog)
        finally:
            observed_in_task.append(_CURRENT_TRACE_ID.get())

    task = asyncio.create_task(runner())
    await asyncio.wait_for(entered.wait(), timeout=2.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancel_vm.crash_trace is not None
    assert cancel_vm.crash_trace.status == TraceStatus.CANCELLED
    assert observed_in_task == [None]


@pytest.mark.asyncio
async def test_runtime_context_does_not_leak_into_cursor_or_snapshots(
    no_sleep: None,
) -> None:
    retry_attempts = 0

    def flaky_step(idem_key: str, attempt: int) -> dict[str, str]:
        nonlocal retry_attempts
        retry_attempts += 1
        assert idem_key
        if retry_attempts == 1:
            raise RuntimeError("transient")
        return {"step1": "ok"}

    def clean_step(idem_key: str, attempt: int) -> dict[str, str]:
        assert idem_key and attempt == 1
        return {"step1": "ok"}

    def par_branch(step_id: str) -> str:
        return f"done:{step_id}"

    def suspend_tool(exec_id: str, step_id: str, attempt: int) -> str:
        assert exec_id and step_id == "wait" and attempt == 1
        return "PENDING"

    program = Program.from_dict(
        {
            "name": "cursor_non_leak",
            "steps": [
                {
                    "id": "s_retry",
                    "type": "tool",
                    "tool": "step1",
                    "idempotent": True,
                    "on_error": "retry",
                    "max_retries": 3,
                    "output_key": "s_retry_out",
                    "args": {
                        "idem_key": "$__execution_id__:$__step_id__",
                        "attempt": "$__attempt__",
                    },
                },
                {
                    "id": "s_par",
                    "type": "parallel",
                    "output_key": "s_par_out",
                    "parallel_steps": [
                        {
                            "id": "p1",
                            "type": "tool",
                            "tool": "par_branch",
                            "args": {"step_id": "$__step_id__"},
                        },
                        {
                            "id": "p2",
                            "type": "tool",
                            "tool": "par_branch",
                            "args": {"step_id": "$__step_id__"},
                        },
                    ],
                },
                {
                    "id": "wait",
                    "type": "tool",
                    "tool": "suspend_tool",
                    "args": {
                        "exec_id": "$__execution_id__",
                        "step_id": "$__step_id__",
                        "attempt": "$__attempt__",
                    },
                },
            ],
        }
    )

    repo = InMemoryCursorRepository()
    vm_retried = ExecutionVM(
        tools={"step1": flaky_step, "par_branch": par_branch, "suspend_tool": suspend_tool},
        cursor_repository=repo,
    )
    trace_retried = await vm_retried.run(program)
    assert trace_retried.status == TraceStatus.SUSPENDED

    loaded = await repo.load(trace_retried.trace_id)
    assert loaded is not None
    _step_id, saved_state, _saved_trace = loaded
    assert _RUNTIME_CONTEXT_KEYS.isdisjoint(saved_state.data.keys())
    assert _RUNTIME_CONTEXT_KEYS.isdisjoint(saved_state.step_outputs.keys())

    # Compare snapshots against a clean (no-retry) run with a different trace_id
    vm_clean = ExecutionVM(
        tools={"step1": clean_step, "par_branch": par_branch, "suspend_tool": suspend_tool},
    )
    trace_clean = await vm_clean.run(program)
    assert trace_clean.status == TraceStatus.SUSPENDED
    assert trace_retried.trace_id != trace_clean.trace_id
    assert trace_retried.state_snapshots == trace_clean.state_snapshots
    assert (
        TraceAnalyzer(trace_retried).receipt().trace_hash
        == TraceAnalyzer(trace_clean).receipt().trace_hash
    )


# ---------------------------------------------------------------------------
# 4. Step.idempotent, explicit False, roundtrip & validator WARNING
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "step_dict",
    [
        {"id": "l", "type": "llm", "prompt": "hi", "idempotent": True},
        {
            "id": "c",
            "type": "condition",
            "condition": "$x == 1",
            "then": "t",
            "idempotent": True,
        },
        {
            "id": "p",
            "type": "parallel",
            "parallel_steps": [{"id": "s", "type": "tool", "tool": "t"}],
            "idempotent": True,
        },
        {
            "id": "p_sub",
            "type": "parallel",
            "parallel_steps": [
                {"id": "s_llm", "type": "llm", "prompt": "hi", "idempotent": True},
            ],
        },
    ],
)
def test_idempotent_rejected_on_non_tool_steps(step_dict: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Program.from_dict({"name": "bad_idempotent", "steps": [step_dict]})


@pytest.mark.parametrize(
    "step_dict",
    [
        {"id": "l", "type": "llm", "prompt": "hi", "idempotent": False},
        {
            "id": "c",
            "type": "condition",
            "condition": "$x == 1",
            "then": "t",
            "idempotent": False,
        },
        {
            "id": "p",
            "type": "parallel",
            "parallel_steps": [{"id": "s", "type": "tool", "tool": "t"}],
            "idempotent": False,
        },
        {
            "id": "p_sub",
            "type": "parallel",
            "parallel_steps": [
                {"id": "s_llm", "type": "llm", "prompt": "hi", "idempotent": False},
            ],
        },
    ],
)
def test_idempotent_false_explicit_accepted_on_non_tool_steps(
    step_dict: dict[str, Any],
) -> None:
    program = Program.from_dict({"name": "explicit_false", "steps": [step_dict]})
    assert program.steps[0].idempotent is False


def test_idempotent_program_model_dump_roundtrip() -> None:
    original = Program.from_dict(
        {
            "name": "roundtrip_idempotent",
            "steps": [
                {"id": "classify", "type": "llm", "prompt": "hi", "output_key": "c"},
                {
                    "id": "charge",
                    "type": "tool",
                    "tool": "charge_card",
                    "idempotent": True,
                    "on_error": "retry",
                    "args": {"key": "$__execution_id__:$__step_id__"},
                },
            ],
        }
    )
    reloaded = Program.from_dict(original.model_dump())
    charge_step = reloaded.get_step("charge")
    assert charge_step is not None
    assert charge_step.idempotent is True
    report = ProgramValidator(reloaded).validate()
    assert report.is_valid()
    assert report.by_kind(IssueKind.NON_IDEMPOTENT_TOOL_RETRY) == []


def test_validator_warns_on_non_idempotent_tool_retry_and_clears_when_declared() -> None:
    prog_warn = Program.from_dict(
        {
            "name": "tool_retry_warn",
            "steps": [
                {"id": "pay", "type": "tool", "tool": "pay", "on_error": "retry"},
            ],
        }
    )
    report_warn = ProgramValidator(prog_warn).validate()
    assert report_warn.is_valid()  # WARNING is non-breaking
    warnings = report_warn.by_kind(IssueKind.NON_IDEMPOTENT_TOOL_RETRY)
    assert len(warnings) == 1
    assert warnings[0].step_id == "pay"
    assert warnings[0].severity == IssueSeverity.WARNING

    prog_ok = Program.from_dict(
        {
            "name": "tool_retry_ok",
            "steps": [
                {
                    "id": "pay",
                    "type": "tool",
                    "tool": "pay",
                    "on_error": "retry",
                    "idempotent": True,
                }
            ],
        }
    )
    report_ok = ProgramValidator(prog_ok).validate()
    assert report_ok.is_valid()
    assert report_ok.by_kind(IssueKind.NON_IDEMPOTENT_TOOL_RETRY) == []


# ---------------------------------------------------------------------------
# 5. README example & current-behavior characterizations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readme_idempotent_charge_example(no_sleep: None) -> None:
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")

    blocks = [
        m.group(1)
        for m in re.finditer(r"```python\n(.*?)```", readme_text, re.DOTALL)
        if '"idempotent_charge"' in m.group(1)
    ]
    assert len(blocks) == 1, "Expected exactly one idempotent_charge example in README.md"

    tree = ast.parse(blocks[0])
    prog_dict: dict[str, Any] | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_dict"
            and node.args
        ):
            prog_dict = ast.literal_eval(node.args[0])
            break
    assert prog_dict is not None

    program = Program.from_dict(prog_dict)
    report = ProgramValidator(program).validate()
    assert report.is_valid()
    assert report.by_kind(IssueKind.NON_IDEMPOTENT_TOOL_RETRY) == []

    calls: list[tuple[str, int]] = []

    def charge_card(idempotency_key: str, attempt: int) -> str:
        calls.append((idempotency_key, attempt))
        if attempt == 1:
            raise RuntimeError("network blip")
        return "charged"

    trace = await ExecutionVM(tools={"charge_card": charge_card}).run(program, validate=True)
    assert trace.status == TraceStatus.SUCCESS
    assert trace.final_output == "charged"
    assert calls == [
        (f"{trace.trace_id}:charge", 1),
        (f"{trace.trace_id}:charge", 2),
    ]


def test_parallel_tool_substep_retry_rejected_by_model() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Program.from_dict(
            {
                "name": "par_tool_sub_retry",
                "steps": [
                    {
                        "id": "fanout",
                        "type": "parallel",
                        "parallel_steps": [
                            {
                                "id": "branch_tool",
                                "type": "tool",
                                "tool": "do_work",
                                "on_error": "retry",
                                "max_retries": 3,
                            },
                            {"id": "s2", "type": "tool", "tool": "ok_tool"},
                        ],
                    }
                ],
            }
        )
    msg = str(exc_info.value)
    assert "fanout" in msg
    assert "branch_tool" in msg
    assert "retry" in msg.lower()


def test_parallel_llm_substep_retry_rejected_by_model() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Program.from_dict(
            {
                "name": "par_llm_sub_retry",
                "steps": [
                    {
                        "id": "fanout_llm",
                        "type": "parallel",
                        "parallel_steps": [
                            {
                                "id": "branch_llm",
                                "type": "llm",
                                "prompt": "hi",
                                "on_error": "retry",
                            }
                        ],
                    }
                ],
            }
        )
    msg = str(exc_info.value)
    assert "fanout_llm" in msg
    assert "branch_llm" in msg


def test_parallel_substeps_fail_skip_and_model_dump_roundtrip_accepted() -> None:
    program = Program.from_dict(
        {
            "name": "par_sub_valid",
            "steps": [
                {
                    "id": "fanout",
                    "type": "parallel",
                    "parallel_steps": [
                        {
                            "id": "sub_fail",
                            "type": "tool",
                            "tool": "t1",
                            "on_error": "fail",
                            "max_retries": 5,
                        },
                        {
                            "id": "sub_skip",
                            "type": "tool",
                            "tool": "t2",
                            "on_error": "skip",
                        },
                        {
                            "id": "sub_llm_skip",
                            "type": "llm",
                            "prompt": "classify",
                            "allowed_outputs": ["yes", "no"],
                            "on_error": "skip",
                        },
                    ],
                }
            ],
        }
    )
    reloaded = Program.from_dict(program.model_dump())
    fanout = reloaded.get_step("fanout")
    assert fanout is not None
    assert [s.id for s in fanout.parallel_steps] == ["sub_fail", "sub_skip", "sub_llm_skip"]
    assert ProgramValidator(reloaded).validate().is_valid()


@pytest.mark.asyncio
async def test_characterization_parallel_retry_without_validate_reruns_all_branches(
    no_sleep: None,
) -> None:
    """Characterization: without validate=True, on_error='retry' on a parallel
    step retains legacy runtime behavior (all branches re-execute on retry).
    """
    a_calls = 0
    b_calls = 0

    def branch_a() -> str:
        nonlocal a_calls
        a_calls += 1
        return f"a-{a_calls}"

    def branch_b() -> str:
        nonlocal b_calls
        b_calls += 1
        if b_calls == 1:
            raise RuntimeError("transient b failure")
        return "b-ok"

    program = Program.from_dict(
        {
            "name": "par_retry_unvalidated",
            "steps": [
                {
                    "id": "fanout",
                    "type": "parallel",
                    "on_error": "retry",
                    "max_retries": 2,
                    "parallel_steps": [
                        {"id": "a", "type": "tool", "tool": "branch_a"},
                        {"id": "b", "type": "tool", "tool": "branch_b"},
                    ],
                }
            ],
        }
    )

    vm = ExecutionVM(tools={"branch_a": branch_a, "branch_b": branch_b})
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert a_calls == 2
    assert b_calls == 2
