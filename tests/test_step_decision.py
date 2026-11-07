"""SD-01..: governance decisions declared on steps, counted by the receipt (gap G-09)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nano_vm import (
    DecisionKind,
    ExecutionVM,
    Program,
    StepDecision,
    TraceAnalyzer,
    TraceStatus,
)
from nano_vm.adapters import MockLLMAdapter
from nano_vm.models import StepResult, StepStatus, Trace

SUPPORT = {
    "name": "support",
    "steps": [
        {"id": "classify", "type": "tool", "tool": "classify"},
        {
            "id": "route",
            "type": "condition",
            "condition": "$topic == 'refund'",
            "then": "escalate",
            "otherwise": "deny",
        },
        {
            "id": "escalate",
            "type": "tool",
            "tool": "notify",
            "decision": {"kind": "escalate", "rule_id": "R-12", "reason": "refund > limit"},
            "is_terminal": True,
        },
        {"id": "deny", "type": "tool", "tool": "refuse", "decision": "block", "is_terminal": True},
    ],
}


def _tools(topic: str) -> dict[str, object]:
    return {
        "classify": lambda **kw: topic,
        "notify": lambda **kw: "sent",
        "refuse": lambda **kw: "refused",
    }


async def _run(data: dict, topic: str, ctx: dict | None = None):
    vm = ExecutionVM(tools=_tools(topic))  # type: ignore[arg-type]
    return await vm.run(Program.from_dict(data), ctx if ctx is not None else {"topic": topic})


# SD-01: model, shorthand and validation
def test_sd01_decision_model_and_shorthand() -> None:
    assert StepDecision.model_validate("escalate").kind is DecisionKind.ESCALATE
    d = StepDecision.model_validate({"kind": "block", "rule_id": "R-1", "reason": "x"})
    assert (d.kind, d.rule_id, d.reason) == (DecisionKind.BLOCK, "R-1", "x")
    with pytest.raises(ValidationError):
        StepDecision.model_validate("allow")  # absence of a decision means allow
    with pytest.raises(ValidationError):
        d.rule_id = "mutated"  # type: ignore[misc]  # frozen


# SD-02: where a decision may be declared
def test_sd02_decision_placement_rules() -> None:
    ok = {"name": "p", "steps": [{"id": "a", "type": "tool", "tool": "t", "decision": "block"}]}
    assert Program.from_dict(ok).steps[0].decision.kind is DecisionKind.BLOCK  # type: ignore[union-attr]

    on_condition = {
        "name": "p",
        "steps": [
            {
                "id": "c",
                "type": "condition",
                "condition": "$a",
                "then": "x",
                "decision": "escalate",
            },
            {"id": "x", "type": "tool", "tool": "t"},
        ],
    }
    with pytest.raises(ValidationError, match="condition step"):
        Program.from_dict(on_condition)

    on_sub = {
        "name": "p",
        "steps": [
            {
                "id": "par",
                "type": "parallel",
                "parallel_steps": [{"id": "s1", "type": "tool", "tool": "t", "decision": "block"}],
            }
        ],
    }
    with pytest.raises(ValidationError, match="parallel sub-step"):
        Program.from_dict(on_sub)

    on_parallel_itself = {
        "name": "p",
        "steps": [
            {
                "id": "par",
                "type": "parallel",
                "decision": "escalate",
                "parallel_steps": [{"id": "s1", "type": "tool", "tool": "t"}],
            }
        ],
    }
    assert Program.from_dict(on_parallel_itself).steps[0].decision is not None


# SD-03: the VM copies the declared decision into StepResult, with metadata
@pytest.mark.asyncio
async def test_sd03_vm_copies_decision_to_step_result() -> None:
    trace = await _run(SUPPORT, "refund")
    by_id = {s.step_id: s for s in trace.steps}
    esc = by_id["escalate"]
    assert esc.status == StepStatus.SUCCESS and esc.decision is not None
    assert (esc.decision.kind, esc.decision.rule_id) == (DecisionKind.ESCALATE, "R-12")
    assert by_id["classify"].decision is None and by_id["route"].decision is None
    assert "deny" not in by_id  # not executed: its decision must not appear anywhere


@pytest.mark.asyncio
async def test_sd03b_llm_step_carries_decision() -> None:
    prog = Program.from_dict(
        {
            "name": "p",
            "steps": [
                {"id": "ask", "type": "llm", "prompt": "hi", "decision": "escalate"},
            ],
        }
    )
    trace = await ExecutionVM(llm=MockLLMAdapter("ok")).run(prog)
    assert trace.steps[0].decision is not None
    assert trace.steps[0].decision.kind is DecisionKind.ESCALATE


# SD-04: only a completed step carries its decision
@pytest.mark.asyncio
async def test_sd04_failed_skipped_pending_steps_carry_no_decision() -> None:
    def boom(**kw: object) -> str:
        raise RuntimeError("down")

    failed = Program.from_dict(
        {"name": "f", "steps": [{"id": "a", "type": "tool", "tool": "boom", "decision": "block"}]}
    )
    t = await ExecutionVM(tools={"boom": boom}).run(failed)
    assert t.status == TraceStatus.FAILED
    assert t.steps[0].status == StepStatus.FAILED and t.steps[0].decision is None

    skipped = Program.from_dict(
        {
            "name": "s",
            "steps": [
                {
                    "id": "a",
                    "type": "tool",
                    "tool": "boom",
                    "decision": "block",
                    "on_error": "skip",
                },
                {"id": "b", "type": "tool", "tool": "ok", "is_terminal": True},
            ],
        }
    )
    t = await ExecutionVM(tools={"boom": boom, "ok": lambda **kw: 1}).run(skipped)
    assert t.steps[0].status == StepStatus.SKIPPED and t.steps[0].decision is None

    pending = Program.from_dict(
        {
            "name": "p",
            "steps": [{"id": "a", "type": "tool", "tool": "pend", "decision": "escalate"}],
        }
    )
    t = await ExecutionVM(tools={"pend": lambda **kw: "PENDING"}).run(pending)
    assert t.status == TraceStatus.SUSPENDED and t.steps[0].decision is None
    assert TraceAnalyzer(t).receipt().escalations == 0


# SD-05: the receipt counts exactly the decisions on the path that ran
@pytest.mark.asyncio
async def test_sd05_receipt_counts_executed_path_only() -> None:
    esc = TraceAnalyzer(await _run(SUPPORT, "refund")).receipt()
    assert (esc.escalations, esc.blocked_actions) == (1, 0)
    den = TraceAnalyzer(await _run(SUPPORT, "other")).receipt()
    assert (den.escalations, den.blocked_actions) == (0, 1)
    assert esc.trace_hash != den.trace_hash  # different paths, different outputs


# SD-06: a block is a successful deliberate denial, not a rejected transition
@pytest.mark.asyncio
async def test_sd06_block_is_not_a_rejected_transition() -> None:
    receipt = TraceAnalyzer(await _run(SUPPORT, "other")).receipt()
    assert receipt.blocked_actions == 1
    assert receipt.rejected_transitions == () and receipt.failed_steps == 0


# SD-07: counts accumulate over several decision steps
@pytest.mark.asyncio
async def test_sd07_multiple_decisions_accumulate() -> None:
    prog = {
        "name": "multi",
        "steps": [
            {"id": "a", "type": "tool", "tool": "t", "decision": "escalate"},
            {"id": "b", "type": "tool", "tool": "t", "decision": "block"},
            {"id": "c", "type": "tool", "tool": "t", "decision": "escalate"},
            {"id": "d", "type": "tool", "tool": "t", "is_terminal": True},
        ],
    }
    t = await ExecutionVM(tools={"t": lambda **kw: 1}).run(Program.from_dict(prog))
    r = TraceAnalyzer(t).receipt()
    assert (r.escalations, r.blocked_actions) == (2, 1)


# SD-08: analyzer rule on non-SUCCESS results (guards the status check independently of the VM)
def test_sd08_receipt_ignores_decisions_on_non_success_results() -> None:
    dec = StepDecision(kind=DecisionKind.ESCALATE)
    t = Trace(program_name="p", status=TraceStatus.SUCCESS)
    for sid, st in (
        ("ok", StepStatus.SUCCESS),
        ("bad", StepStatus.FAILED),
        ("skip", StepStatus.SKIPPED),
        ("wait", StepStatus.PENDING),
    ):
        t = t.add_step(StepResult(step_id=sid, status=st, decision=dec))
    assert TraceAnalyzer(t).receipt().escalations == 1


# SD-09: documented property — decisions are not part of the state fingerprint
@pytest.mark.asyncio
async def test_sd09_decision_does_not_change_trace_hash() -> None:
    def prog(decision: str | None) -> Program:
        step = {"id": "a", "type": "tool", "tool": "t", "is_terminal": True}
        if decision:
            step["decision"] = decision
        return Program.from_dict({"name": "p", "steps": [step]})

    vm_tools = {"t": lambda **kw: "same"}
    a = await ExecutionVM(tools=vm_tools).run(prog(None))
    b = await ExecutionVM(tools=vm_tools).run(prog("escalate"))
    ra, rb = TraceAnalyzer(a).receipt(), TraceAnalyzer(b).receipt()
    assert ra.trace_hash == rb.trace_hash
    assert (ra.escalations, rb.escalations) == (0, 1)


# SD-10: serialisation compatibility (old traces load, new ones round-trip)
def test_sd10_serialisation_compat() -> None:
    old = {"step_id": "a", "status": "success", "output": 1}
    assert StepResult.model_validate(old).decision is None
    r = StepResult(
        step_id="a",
        status=StepStatus.SUCCESS,
        decision=StepDecision(kind=DecisionKind.BLOCK, rule_id="R-1"),
    )
    back = StepResult.model_validate(json.loads(r.model_dump_json()))
    assert back.decision == r.decision


# SD-11: programs without decisions are unchanged (receipt counters stay 0)
@pytest.mark.asyncio
async def test_sd11_no_decisions_no_counts() -> None:
    prog = {"name": "p", "steps": [{"id": "a", "type": "tool", "tool": "t", "is_terminal": True}]}
    t = await ExecutionVM(tools={"t": lambda **kw: 1}).run(Program.from_dict(prog))
    r = TraceAnalyzer(t).receipt()
    assert (r.escalations, r.blocked_actions) == (0, 0)
