"""
tests/test_cancellation.py
===========================
Cancellability of ExecutionVM.run() from an external asyncio caller.

A cycle in the transition graph (see ProgramValidator.cycle_detection) run
through purely synchronous tool functions never suspends internally on its
own. Before this fix, asyncio.wait_for(vm.run(...), timeout=N) could not
deliver CancelledError until the coroutine returned by itself -- i.e. never,
for a genuine cycle. _execute_loop now awaits asyncio.sleep(0) once per
iteration as a cooperative checkpoint.

Non-goal: this does NOT make a single blocking tool call interruptible
mid-call. Only the transition between steps becomes a cancellation point.
A tool that itself never returns (e.g. `while True: pass` with no await)
still cannot be cancelled -- that requires running sync tools off-loop
(run_in_executor), which is out of scope here.
"""

from __future__ import annotations

import asyncio

import pytest

from nano_vm import ExecutionVM, Program
from nano_vm.validator import IssueKind, ProgramValidator


class MockLLM:
    async def complete(self, messages):
        return "ok"


def make_vm(tools: dict | None = None) -> ExecutionVM:
    return ExecutionVM(llm=MockLLM(), tools=tools or {})


def make_cycle_program() -> Program:
    """a -> b -> a: a genuine graph cycle. ProgramValidator flags it, but
    ExecutionVM.run() does not call the validator itself (validation stays
    opt-in by design, see DECISIONS.md 2026-06-28) -- so this program runs
    forever unless the caller enforces max_steps or cancels externally.
    """
    return Program.from_dict(
        {
            "name": "cycle_test",
            "steps": [
                {"id": "a", "type": "tool", "tool": "noop", "next_step": "b"},
                {"id": "b", "type": "tool", "tool": "noop", "next_step": "a"},
            ],
        }
    )


def noop(**kwargs) -> str:
    return "ok"


@pytest.mark.asyncio
async def test_validator_flags_the_cycle_used_below():
    """Sanity check on the fixture itself: confirm this is a real cycle,
    not an accidentally-terminating program that would make the tests
    below pass for the wrong reason."""
    report = ProgramValidator(make_cycle_program()).validate()
    assert any(issue.kind == IssueKind.CYCLE_DETECTED for issue in report.issues)
    assert not report.is_valid()


@pytest.mark.asyncio
async def test_wait_for_cancels_a_cyclic_run():
    """asyncio.wait_for must be able to cancel a run stuck in a cycle of
    synchronous tool calls, instead of hanging until killed externally.
    """
    vm = make_vm({"noop": noop})
    program = make_cycle_program()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(vm.run(program), timeout=0.05)


@pytest.mark.asyncio
async def test_task_cancel_actually_stops_a_cyclic_run():
    """Task.cancel() (what wait_for uses internally) must really reach the
    coroutine and stop it -- not just let the caller give up while the
    task keeps burning CPU in the background forever.
    """
    vm = make_vm({"noop": noop})
    program = make_cycle_program()

    task = asyncio.ensure_future(vm.run(program))
    await asyncio.sleep(0.01)
    assert not task.done(), "cycle finished on its own -- fixture is not a real cycle"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()