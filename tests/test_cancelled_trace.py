"""
tests/test_cancelled_trace.py
=============================
Q3 (owner-confirmed 2026-09-18): trace finalization on external cancellation.

Before: asyncio.wait_for()/Task.cancel() aborted ExecutionVM.run() with a bare
CancelledError -- the partial Trace was garbage-collected, and nothing recorded
that a run had been interrupted or how far it got.

Now: _execute_loop terminalizes the last consistent Trace as
TraceStatus.CANCELLED, exposes it as ExecutionVM.last_trace, and re-raises
CancelledError unchanged.

Cancellation points covered (CN = cancelled-trace):
  CN-01  sleep(0) checkpoint between steps (cyclic sync tools)
  CN-02  same, delivered through asyncio.wait_for timeout
  CN-03  retry-backoff sleep inside _run_step
  CN-04  llm adapter await inside _run_step
  CN-05  TraceAnalyzer.receipt()/report() on a CANCELLED trace do not crash
  CN-06  CONDITION->CONDITION recursion: innermost frame's trace wins
  CN-07  resume_with_program() entry point (run() / resume divergence precedent)
  CN-08  last_trace reset on entry: stale value never survives a later call
  CN-09  reset precedes the llm pre-flight (VMError leaves last_trace None)

Non-goals, asserted by absence rather than by test:
  - a single blocking sync tool call is still not interruptible mid-call
    (see tests/test_cancellation.py);
  - the in-flight step is NOT recorded. A cancel during an llm await is
    crash-equivalent: the provider request may already have been sent (and
    billed), and the Trace cannot tell "never sent" from "sent, no reply".
    CN-04 pins this: the in-flight step is absent from Trace.steps.
"""

from __future__ import annotations

import asyncio

import pytest

from nano_vm import ExecutionVM, Program, TraceAnalyzer
from nano_vm.models import StepStatus, Trace, TraceStatus
from nano_vm.vm import VMError, WebhookEvent

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def noop(**kwargs: object) -> str:
    return "ok"


def pend(**kwargs: object) -> str:
    return "PENDING"  # reserved suspend sentinel


def make_vm(tools: dict | None = None) -> ExecutionVM:
    return ExecutionVM(llm=None, tools=tools or {"noop": noop})


CYCLE = {
    "name": "cycle_cancel",
    "steps": [
        {"id": "a", "type": "tool", "tool": "noop", "next_step": "b"},
        {"id": "b", "type": "tool", "tool": "noop", "next_step": "a"},
    ],
}


async def cancel_running(coro) -> None:
    """Start `coro` as a task, let it spin, cancel it, and assert the
    CancelledError really propagates (i.e. is re-raised, not swallowed)."""
    task = asyncio.ensure_future(coro)
    await asyncio.sleep(0.01)
    assert not task.done(), "run finished on its own -- fixture is not a real cycle"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()


async def cancelled_cycle_trace(vm: ExecutionVM) -> Trace:
    await cancel_running(vm.run(Program.from_dict(CYCLE)))
    assert vm.last_trace is not None
    return vm.last_trace


# ---------------------------------------------------------------------------
# CN-01 / CN-02: sleep(0) checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cn01_cancel_at_checkpoint_finalizes_trace():
    vm = make_vm()
    assert vm.last_trace is None

    t = await cancelled_cycle_trace(vm)

    assert t.status == TraceStatus.CANCELLED
    assert len(t.steps) > 0
    assert all(s.status == StepStatus.SUCCESS for s in t.steps)
    assert {s.step_id for s in t.steps} == {"a", "b"}
    assert t.finished_at is not None and t.duration_ms is not None
    assert t.final_output is None
    assert t.error is not None and t.error.startswith("cancelled:")
    assert f"{len(t.steps)} recorded step(s)" in t.error


@pytest.mark.asyncio
async def test_cn02_wait_for_timeout_finalizes_trace():
    vm = make_vm()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(vm.run(Program.from_dict(CYCLE)), timeout=0.05)

    assert vm.last_trace is not None
    assert vm.last_trace.status == TraceStatus.CANCELLED
    assert len(vm.last_trace.steps) > 0


# ---------------------------------------------------------------------------
# CN-03: retry backoff (cancel lands INSIDE a step, after one failed attempt)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cn03_cancel_in_retry_backoff_excludes_inflight_step(monkeypatch):
    real_sleep = asyncio.sleep
    reached = asyncio.Event()
    never = asyncio.Event()

    async def gated_sleep(delay, *args, **kwargs):
        if delay > 0:  # retry backoff -- park here until cancelled
            reached.set()
            await never.wait()
        else:  # sleep(0) checkpoint keeps its real behaviour
            await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", gated_sleep)

    boom_calls: list[int] = []

    def boom(**kwargs: object) -> str:
        boom_calls.append(1)
        raise RuntimeError("transient")

    vm = make_vm({"noop": noop, "boom": boom})
    program = Program.from_dict(
        {
            "name": "backoff_cancel",
            "steps": [
                {"id": "first", "type": "tool", "tool": "noop"},
                {
                    "id": "flaky",
                    "type": "tool",
                    "tool": "boom",
                    "on_error": "retry",
                    "max_retries": 3,
                },
            ],
        }
    )

    task = asyncio.ensure_future(vm.run(program))
    await asyncio.wait_for(reached.wait(), timeout=5)
    assert boom_calls == [1], "cancel must land after the first failed attempt"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()

    t = vm.last_trace
    assert t is not None
    assert t.status == TraceStatus.CANCELLED
    # 'flaky' was in flight: no StepResult was ever produced for it.
    assert [s.step_id for s in t.steps] == ["first"]
    # ...and therefore it is not miscounted as a failed step.
    assert TraceAnalyzer(t).receipt().failed_steps == 0


# ---------------------------------------------------------------------------
# CN-04: llm adapter await (crash-equivalent -- see module docstring)
# ---------------------------------------------------------------------------


class HangingLLM:
    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self._never = asyncio.Event()

    async def complete(self, messages):
        self.reached.set()
        await self._never.wait()
        return "unreachable"


@pytest.mark.asyncio
async def test_cn04_cancel_in_llm_await_excludes_inflight_step():
    llm = HangingLLM()
    vm = ExecutionVM(llm=llm, tools={"noop": noop})
    program = Program.from_dict(
        {
            "name": "llm_cancel",
            "steps": [
                {"id": "first", "type": "tool", "tool": "noop"},
                {"id": "ask", "type": "llm", "prompt": "hi"},
            ],
        }
    )

    task = asyncio.ensure_future(vm.run(program))
    await asyncio.wait_for(llm.reached.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()

    t = vm.last_trace
    assert t is not None
    assert t.status == TraceStatus.CANCELLED
    assert [s.step_id for s in t.steps] == ["first"]  # 'ask' absent: crash-equivalent
    assert t.step_metrics.llm_calls == 0  # metric is recorded on step completion only


# ---------------------------------------------------------------------------
# CN-05: receipt / health report on a CANCELLED trace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cn05_receipt_and_report_from_cancelled_trace():
    t = await cancelled_cycle_trace(make_vm())

    analyzer = TraceAnalyzer(t)
    receipt = analyzer.receipt()
    analyzer.report()  # must not raise on the new status

    assert receipt.trace_id == t.trace_id
    assert receipt.final_status == TraceStatus.CANCELLED
    assert receipt.resumable is False  # no cursor was saved for a cancelled run
    assert receipt.replayable is True  # terminal, recomputable from the trace
    assert receipt.failed_steps == 0
    assert receipt.rejected_transitions == ()
    assert receipt.trace_hash == TraceAnalyzer(t).receipt().trace_hash  # recomputable


# ---------------------------------------------------------------------------
# CN-06: recursion guard (CONDITION -> CONDITION re-enters _execute_loop)
# ---------------------------------------------------------------------------

NESTED_CONDITION_THEN_SPIN = {
    "name": "nested_cond_cancel",
    "steps": [
        {
            "id": "c1",
            "type": "condition",
            "condition": "'x' == 'y'",
            "then": "leaf_dead",
            "otherwise": "c2",
        },
        {
            "id": "c2",
            "type": "condition",
            "condition": "'a' == 'a'",
            "then": "spin_a",
            "otherwise": "leaf_dead",
        },
        {"id": "leaf_dead", "type": "tool", "tool": "noop", "is_terminal": True},
        {"id": "spin_a", "type": "tool", "tool": "noop", "next_step": "spin_b"},
        {"id": "spin_b", "type": "tool", "tool": "noop", "next_step": "spin_a"},
    ],
}


@pytest.mark.asyncio
async def test_cn06_innermost_frame_trace_is_not_overwritten_by_outer_frame():
    """c1 -> c2 (condition) recurses into _execute_loop(start_step_id='spin_a'),
    which then spins. The cancel is delivered in the INNER frame; the outer
    frame's local `trace` is stale (c1, c2 only). Without the recursion guard
    the outer handler overwrites last_trace with that stale two-step trace."""
    vm = make_vm()
    await cancel_running(vm.run(Program.from_dict(NESTED_CONDITION_THEN_SPIN)))

    t = vm.last_trace
    assert t is not None
    assert t.status == TraceStatus.CANCELLED
    ids = [s.step_id for s in t.steps]
    assert ids[:2] == ["c1", "c2"]
    assert len(ids) > 4, f"stale outer-frame trace won: {ids}"
    assert ids[-1] in {"spin_a", "spin_b"}


# ---------------------------------------------------------------------------
# CN-07: resume_with_program() is a separate entry point into _execute_loop
# ---------------------------------------------------------------------------

SUSPEND_THEN_SPIN = {
    "name": "resume_cancel",
    "steps": [
        {"id": "gate", "type": "tool", "tool": "pend"},
        {"id": "spin_a", "type": "tool", "tool": "noop", "next_step": "spin_b"},
        {"id": "spin_b", "type": "tool", "tool": "noop", "next_step": "spin_a"},
    ],
}


@pytest.mark.asyncio
async def test_cn07_cancel_during_resume_finalizes_trace():
    vm = make_vm({"noop": noop, "pend": pend})
    program = Program.from_dict(SUSPEND_THEN_SPIN)

    suspended = await vm.run(program)
    assert suspended.status == TraceStatus.SUSPENDED
    assert vm.last_trace is None  # suspension is not a cancellation

    event = WebhookEvent(trace_id=suspended.trace_id, payload={})
    await cancel_running(vm.resume_with_program(event, program))

    t = vm.last_trace
    assert t is not None
    assert t.status == TraceStatus.CANCELLED
    assert t.trace_id == suspended.trace_id  # same trace, continued then interrupted
    ids = [s.step_id for s in t.steps]
    assert ids[0] == "gate"
    assert len(ids) > 2  # steps executed after the resume are recorded
    assert TraceAnalyzer(t).receipt().resumable is False


# ---------------------------------------------------------------------------
# CN-08 / CN-09: reset-on-entry semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cn08_normal_run_resets_stale_last_trace():
    vm = make_vm()
    await cancelled_cycle_trace(vm)
    assert vm.last_trace is not None

    ok = await vm.run(
        Program.from_dict(
            {"name": "quick", "steps": [{"id": "only", "type": "tool", "tool": "noop"}]}
        )
    )

    assert ok.status == TraceStatus.SUCCESS
    assert vm.last_trace is None  # last_trace describes only the most recent call


@pytest.mark.asyncio
async def test_cn09_reset_precedes_llm_preflight():
    vm = make_vm()  # llm=None
    await cancelled_cycle_trace(vm)
    assert vm.last_trace is not None

    needs_llm = Program.from_dict(
        {"name": "needs_llm", "steps": [{"id": "ask", "type": "llm", "prompt": "hi"}]}
    )
    with pytest.raises(VMError):
        await vm.run(needs_llm)  # rejected pre-flight, before any trace exists

    assert vm.last_trace is None  # a stale CANCELLED trace must not survive
