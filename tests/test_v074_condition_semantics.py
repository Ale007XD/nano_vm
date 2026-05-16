"""
tests/test_v074_condition_semantics.py
=======================================
v0.7.4 — Condition branch semantics, Step.is_terminal, Step.next_step,
          _execute_condition ctx wrapping, _resolve typed return.

Coverage:
  CB-01  condition then → terminal branch: s3 executed, s4 skipped
  CB-02  condition otherwise → terminal branch: s4 executed, s3 skipped
  CB-03  condition → inline branch (next_step): branch target + continuation run
  CB-04  condition chain condition→condition→terminal: correct leaf reached
  CB-05  condition chain otherwise path: correct leaf
  CB-06  is_terminal=True on non-branch step: FSM halts after it
  CB-07  next_step invalid id: FAILED with descriptive error
  CB-08  condition → next_step → further steps: full linear continuation
  CB-09  MoMo-style pipeline: guard(inline)→create→poll→check→notify_success
  CB-10  MoMo-style pipeline: guard(inline)→create→poll→check→notify_pending

  CTX-01 _execute_condition: $step_id.output scalar resolved correctly
  CTX-02 _execute_condition: $step_id.output.field dict resolved correctly
  CTX-03 _execute_condition: output_key flat alias accessible in condition
  CTX-04 _execute_condition: missing step → condition False, no crash
  CTX-05 _execute_condition: $step_id.output.nested.field two-level

  RES-01 _resolve: single $var returns typed int (not str)
  RES-02 _resolve: single $var returns dict
  RES-03 _resolve: interpolation stringifies correctly
  RES-04 _resolve: $step_id.output scalar typed return
  RES-05 _resolve: $step_id.output.field typed return from dict output
  RES-06 _resolve: missing key → original string unchanged
"""

from __future__ import annotations

import asyncio

import pytest

from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.adapters import MockLLMAdapter
from nano_vm.models import StateContext, Step, StepType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_vm(tools: dict | None = None) -> ExecutionVM:
    return ExecutionVM(llm=MockLLMAdapter("ok"), tools=tools or {})


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.get_event_loop().run_until_complete(coro)


ECHO = lambda: "done"  # noqa: E731


# ---------------------------------------------------------------------------
# CB: Condition Branch semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cb01_then_terminal():
    """then branch is terminal — s3 runs, s4 skipped."""
    vm = make_vm({"echo": ECHO})
    prog = Program.from_dict({
        "name": "cb01",
        "steps": [
            {"id": "check", "type": "condition",
             "condition": "'yes' == 'yes'", "then": "s3", "otherwise": "s4"},
            {"id": "s3", "type": "tool", "tool": "echo", "is_terminal": True},
            {"id": "s4", "type": "tool", "tool": "echo", "is_terminal": True},
        ],
    })
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert "s3" in ids
    assert "s4" not in ids


@pytest.mark.asyncio
async def test_cb02_otherwise_terminal():
    """otherwise branch is terminal — s4 runs, s3 skipped."""
    vm = make_vm({"echo": ECHO})
    prog = Program.from_dict({
        "name": "cb02",
        "steps": [
            {"id": "check", "type": "condition",
             "condition": "'yes' == 'no'", "then": "s3", "otherwise": "s4"},
            {"id": "s3", "type": "tool", "tool": "echo", "is_terminal": True},
            {"id": "s4", "type": "tool", "tool": "echo", "is_terminal": True},
        ],
    })
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert "s4" in ids
    assert "s3" not in ids


@pytest.mark.asyncio
async def test_cb03_inline_branch_next_step():
    """next_step causes inline continuation after branch target."""
    vm = make_vm({"echo": ECHO})
    prog = Program.from_dict({
        "name": "cb03",
        "steps": [
            {"id": "check", "type": "condition",
             "condition": "'yes' == 'yes'", "then": "step_a", "otherwise": "dead"},
            {"id": "dead",   "type": "tool", "tool": "echo", "is_terminal": True},
            {"id": "step_a", "type": "tool", "tool": "echo",
             "next_step": "step_b"},  # inline — continues to step_b
            {"id": "step_b", "type": "tool", "tool": "echo"},
        ],
    })
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["check", "step_a", "step_b"]


@pytest.mark.asyncio
async def test_cb04_condition_chain_then():
    """condition→condition→terminal: then path."""
    vm = make_vm({"echo": ECHO})
    prog = Program.from_dict({
        "name": "cb04",
        "steps": [
            {"id": "c1", "type": "condition",
             "condition": "'a' == 'a'", "then": "c2", "otherwise": "leaf_no"},
            {"id": "c2", "type": "condition",
             "condition": "'b' == 'b'", "then": "leaf_yes", "otherwise": "leaf_no"},
            {"id": "leaf_yes", "type": "tool", "tool": "echo", "is_terminal": True},
            {"id": "leaf_no",  "type": "tool", "tool": "echo", "is_terminal": True},
        ],
    })
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["c1", "c2", "leaf_yes"]


@pytest.mark.asyncio
async def test_cb05_condition_chain_otherwise():
    """condition→condition→terminal: otherwise path."""
    vm = make_vm({"echo": ECHO})
    prog = Program.from_dict({
        "name": "cb05",
        "steps": [
            {"id": "c1", "type": "condition",
             "condition": "'a' == 'b'", "then": "leaf_yes", "otherwise": "c2"},
            {"id": "c2", "type": "condition",
             "condition": "'x' == 'y'", "then": "leaf_yes", "otherwise": "leaf_no"},
            {"id": "leaf_yes", "type": "tool", "tool": "echo", "is_terminal": True},
            {"id": "leaf_no",  "type": "tool", "tool": "echo", "is_terminal": True},
        ],
    })
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["c1", "c2", "leaf_no"]


@pytest.mark.asyncio
async def test_cb06_is_terminal_on_linear_step():
    """is_terminal on a linear (non-branch) step: FSM halts, next steps skipped."""
    vm = make_vm({"echo": ECHO})
    prog = Program.from_dict({
        "name": "cb06",
        "steps": [
            {"id": "s1", "type": "tool", "tool": "echo", "is_terminal": True},
            {"id": "s2", "type": "tool", "tool": "echo"},  # never reached
        ],
    })
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["s1"]


@pytest.mark.asyncio
async def test_cb07_next_step_invalid_id():
    """next_step pointing to nonexistent step_id → FAILED."""
    vm = make_vm({"echo": ECHO})
    prog = Program.from_dict({
        "name": "cb07",
        "steps": [
            {"id": "check", "type": "condition",
             "condition": "'yes' == 'yes'", "then": "target", "otherwise": "dead"},
            {"id": "target", "type": "tool", "tool": "echo",
             "next_step": "nonexistent"},
            {"id": "dead", "type": "tool", "tool": "echo", "is_terminal": True},
        ],
    })
    trace = await vm.run(prog)
    assert trace.status == TraceStatus.FAILED
    assert "nonexistent" in trace.error


@pytest.mark.asyncio
async def test_cb08_next_step_continues_full_chain():
    """next_step supports multi-hop inline continuation."""
    order = []

    async def record(name: str) -> str:
        order.append(name)
        return name

    vm = ExecutionVM(llm=MockLLMAdapter("ok"), tools={
        "a": lambda: order.append("a") or "a",
        "b": lambda: order.append("b") or "b",
        "c": lambda: order.append("c") or "c",
    })
    prog = Program.from_dict({
        "name": "cb08",
        "steps": [
            {"id": "check", "type": "condition",
             "condition": "'go' == 'go'", "then": "step_a", "otherwise": "dead"},
            {"id": "dead",   "type": "tool", "tool": "a", "is_terminal": True},
            {"id": "step_a", "type": "tool", "tool": "a", "next_step": "step_b"},
            {"id": "step_b", "type": "tool", "tool": "b", "next_step": "step_c"},
            {"id": "step_c", "type": "tool", "tool": "c"},
        ],
    })
    trace = await vm.run(prog)
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["check", "step_a", "step_b", "step_c"]
    assert order == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_cb09_momo_style_success():
    """MoMo-style inline pipeline: guard→create(next_step=poll)→poll→check→notify_success."""
    calls: list[str] = []

    async def create(**_): calls.append("create"); return {"ok": True}
    async def poll(**_):   calls.append("poll");   return {"payment_status": "SUCCESS"}
    async def notify(**_): calls.append("notify"); return {"sent": True}

    vm = ExecutionVM(llm=MockLLMAdapter("ok"), tools={
        "create": create, "poll": poll, "notify": notify,
    })
    prog = Program.from_dict({
        "name": "momo_success",
        "steps": [
            {"id": "guard", "type": "condition",
             "condition": "'ok' == 'ok'", "then": "create", "otherwise": "reject"},
            {"id": "create", "type": "tool", "tool": "create",
             "output_key": "create_result", "next_step": "poll"},
            {"id": "poll",   "type": "tool", "tool": "poll",
             "output_key": "poll_result"},
            {"id": "check",  "type": "condition",
             "condition": "$poll_result.payment_status == 'SUCCESS'",
             "then": "notify_ok", "otherwise": "notify_pending"},
            {"id": "notify_ok",      "type": "tool", "tool": "notify",
             "is_terminal": True},
            {"id": "notify_pending", "type": "tool", "tool": "notify",
             "is_terminal": True},
            {"id": "reject",         "type": "tool", "tool": "notify",
             "is_terminal": True},
        ],
    })
    trace = await vm.run(prog, context={"amount": 50000})
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["guard", "create", "poll", "check", "notify_ok"]
    assert calls == ["create", "poll", "notify"]


@pytest.mark.asyncio
async def test_cb10_momo_style_pending():
    """MoMo-style inline pipeline: poll PENDING → notify_pending."""
    calls: list[str] = []

    async def create(**_): calls.append("create"); return {"ok": True}
    async def poll(**_):   calls.append("poll");   return {"payment_status": "PENDING"}
    async def notify(**_): calls.append("notify"); return {"sent": True}

    vm = ExecutionVM(llm=MockLLMAdapter("ok"), tools={
        "create": create, "poll": poll, "notify": notify,
    })
    prog = Program.from_dict({
        "name": "momo_pending",
        "steps": [
            {"id": "guard", "type": "condition",
             "condition": "'ok' == 'ok'", "then": "create", "otherwise": "reject"},
            {"id": "create", "type": "tool", "tool": "create",
             "next_step": "poll"},
            {"id": "poll",   "type": "tool", "tool": "poll",
             "output_key": "poll_result"},
            {"id": "check",  "type": "condition",
             "condition": "$poll_result.payment_status == 'SUCCESS'",
             "then": "notify_ok", "otherwise": "notify_pending"},
            {"id": "notify_ok",      "type": "tool", "tool": "notify",
             "is_terminal": True},
            {"id": "notify_pending", "type": "tool", "tool": "notify",
             "is_terminal": True},
            {"id": "reject",         "type": "tool", "tool": "notify",
             "is_terminal": True},
        ],
    })
    trace = await vm.run(prog, context={"amount": 50000})
    ids = [s.step_id for s in trace.steps]
    assert trace.status == TraceStatus.SUCCESS
    assert ids == ["guard", "create", "poll", "check", "notify_pending"]


# ---------------------------------------------------------------------------
# CTX: _execute_condition context building
# ---------------------------------------------------------------------------


def test_ctx01_scalar_step_output():
    """$step_id.output resolves scalar step output."""
    vm = make_vm()
    state = StateContext(
        data={},
        step_outputs={"validate": "OK"},
    )
    step = Step(id="chk", type=StepType.CONDITION,
                condition="$validate.output == 'OK'",
                then="yes", otherwise="no")
    assert vm._execute_condition(step, state) == "yes"


def test_ctx02_dict_step_output_field():
    """$step_id.output.field resolves dict output field."""
    vm = make_vm()
    state = StateContext(
        data={},
        step_outputs={"poll": {"payment_status": "SUCCESS", "amount": 50000}},
    )
    step = Step(id="chk", type=StepType.CONDITION,
                condition="$poll.output.payment_status == 'SUCCESS'",
                then="yes", otherwise="no")
    assert vm._execute_condition(step, state) == "yes"


def test_ctx03_output_key_flat_alias():
    """output_key alias ($validation) accessible in condition."""
    vm = make_vm()
    state = StateContext(
        data={"validation": "OK"},      # output_key="validation"
        step_outputs={"validate": "OK"},
    )
    step = Step(id="chk", type=StepType.CONDITION,
                condition="$validation == 'OK'",
                then="yes", otherwise="no")
    assert vm._execute_condition(step, state) == "yes"


def test_ctx04_missing_step_output_condition_false():
    """Missing step in step_outputs → condition evaluates to False, no crash."""
    vm = make_vm()
    state = StateContext(data={}, step_outputs={})
    step = Step(id="chk", type=StepType.CONDITION,
                condition="$nonexistent.output == 'OK'",
                then="yes", otherwise="no")
    assert vm._execute_condition(step, state) == "no"


def test_ctx05_two_level_nested_field():
    """$step_id.output.nested.field two-level dict traversal."""
    vm = make_vm()
    state = StateContext(
        data={},
        step_outputs={"create": {"data": {"order_id": "ORD-123"}}},
    )
    step = Step(id="chk", type=StepType.CONDITION,
                condition="$create.output.data.order_id == 'ORD-123'",
                then="yes", otherwise="no")
    assert vm._execute_condition(step, state) == "yes"


# ---------------------------------------------------------------------------
# RES: _resolve typed return
# ---------------------------------------------------------------------------


def test_res01_single_var_int_typed():
    """Single $var returns original int, not str."""
    vm = make_vm()
    state = StateContext(data={"amount": 50000}, step_outputs={})
    result = vm._resolve("$amount", state)
    assert result == 50000
    assert isinstance(result, int)


def test_res02_single_var_dict_typed():
    """Single $var returns original dict."""
    vm = make_vm()
    state = StateContext(data={"config": {"key": "val"}}, step_outputs={})
    result = vm._resolve("$config", state)
    assert result == {"key": "val"}
    assert isinstance(result, dict)


def test_res03_interpolation_stringifies():
    """$var embedded in larger string → stringified."""
    vm = make_vm()
    state = StateContext(data={"order_id": "ORD-123"}, step_outputs={})
    result = vm._resolve("order-$order_id-suffix", state)
    assert result == "order-ORD-123-suffix"
    assert isinstance(result, str)


def test_res04_step_output_scalar_typed():
    """$step_id.output returns scalar step output with original type."""
    vm = make_vm()
    state = StateContext(data={}, step_outputs={"validate": "OK"})
    result = vm._resolve("$validate.output", state)
    assert result == "OK"
    assert isinstance(result, str)


def test_res05_step_output_field_typed():
    """$step_id.output.field returns typed value from dict output."""
    vm = make_vm()
    state = StateContext(
        data={},
        step_outputs={"refund": {"result_code": 0, "message": "Successful."}},
    )
    result = vm._resolve("$refund.output.result_code", state)
    assert result == 0
    assert isinstance(result, int)


def test_res06_missing_key_unchanged():
    """Missing $key → original string returned unchanged."""
    vm = make_vm()
    state = StateContext(data={}, step_outputs={})
    result = vm._resolve("$nonexistent", state)
    assert result == "$nonexistent"


def test_res07_multi_segment_nested():
    """$step_id.output.a.b multi-segment traversal."""
    vm = make_vm()
    state = StateContext(
        data={},
        step_outputs={"generate": {"order_id": "ORD-001", "request_id": "REQ-001"}},
    )
    result = vm._resolve("$generate.output.order_id", state)
    assert result == "ORD-001"
