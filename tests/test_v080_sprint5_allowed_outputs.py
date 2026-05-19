"""
test_v080_sprint5_allowed_outputs.py
=====================================
Sprint 5 task 2 — Step.allowed_outputs

AO-01  llm step, output in allowed_outputs → SUCCESS, output unchanged
AO-02  llm step, output NOT in allowed_outputs, on_error=fail → VMError raised
AO-03  llm step, output NOT in allowed_outputs, on_error=skip → fallback (allowed_outputs[0])
AO-04  llm step, output NOT in allowed_outputs, on_error=retry → retried; exhausted → VMError
AO-05  allowed_outputs=None (not set) → no validation, any output passes
AO-06  allowed_outputs on non-llm step (tool) → ValidationError at Program construction
AO-07  allowed_outputs=[] (empty) → ValidationError at Step construction
AO-08  output matches after strip() — leading/trailing whitespace ignored
AO-09  multi-step program: first step allowed, second step allowed → both succeed
AO-10  allowed_outputs match is case-sensitive ('Yes' != 'yes')
"""

import pytest

from nano_vm.models import OnError, Program, Step, StepType, TraceStatus
from nano_vm.adapters import MockLLMAdapter
from nano_vm.vm import ExecutionVM, VMError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _program(llm_output: str, allowed: list[str] | None, on_error: str = "fail") -> Program:
    return Program.from_dict({
        "name": "test_allowed_outputs",
        "steps": [
            {
                "id": "classify",
                "type": "llm",
                "prompt": "Classify the request: $input",
                "output_key": "decision",
                **({"allowed_outputs": allowed} if allowed is not None else {}),
                "on_error": on_error,
            }
        ],
    })


# ---------------------------------------------------------------------------
# AO-01  match → SUCCESS, output unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ao_01_match_success():
    program = _program("refund", ["refund", "query", "other"], on_error="fail")
    vm = ExecutionVM(llm=MockLLMAdapter("refund"))
    trace = await vm.run(program, context={"input": "I want a refund"})
    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "refund"


# ---------------------------------------------------------------------------
# AO-02  no match, on_error=fail → VMError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ao_02_no_match_fail():
    program = _program("dunno", ["refund", "query", "other"], on_error="fail")
    vm = ExecutionVM(llm=MockLLMAdapter("dunno"))
    with pytest.raises(VMError, match="allowed_outputs"):
        await vm.run(program, context={"input": "..."})


# ---------------------------------------------------------------------------
# AO-03  no match, on_error=skip → fallback = allowed_outputs[0]
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ao_03_no_match_skip_fallback():
    program = _program("dunno", ["other", "refund", "query"], on_error="skip")
    vm = ExecutionVM(llm=MockLLMAdapter("dunno"))
    trace = await vm.run(program, context={"input": "..."})
    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "other"  # allowed_outputs[0]


# ---------------------------------------------------------------------------
# AO-04  no match, on_error=retry → retried; still wrong → VMError
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ao_04_no_match_retry_exhausted():
    program = Program.from_dict({
        "name": "retry_test",
        "steps": [{
            "id": "classify",
            "type": "llm",
            "prompt": "classify: $input",
            "output_key": "decision",
            "allowed_outputs": ["refund", "query"],
            "on_error": "retry",
            "max_retries": 2,
        }],
    })
    # MockLLMAdapter always returns "dunno" — all retries fail
    vm = ExecutionVM(llm=MockLLMAdapter("dunno"))
    with pytest.raises(VMError, match="allowed_outputs"):
        await vm.run(program, context={"input": "..."})


# ---------------------------------------------------------------------------
# AO-05  allowed_outputs=None → no validation, any output passes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ao_05_none_no_validation():
    program = _program("anything goes", allowed=None)
    vm = ExecutionVM(llm=MockLLMAdapter("anything goes"))
    trace = await vm.run(program, context={"input": "x"})
    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "anything goes"


# ---------------------------------------------------------------------------
# AO-06  allowed_outputs on tool step → ValidationError
# ---------------------------------------------------------------------------

def test_ao_06_allowed_outputs_non_llm_raises():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="allowed_outputs"):
        Program.from_dict({
            "name": "bad",
            "steps": [{
                "id": "pay",
                "type": "tool",
                "tool": "charge",
                "allowed_outputs": ["OK", "FAIL"],
            }],
        })


# ---------------------------------------------------------------------------
# AO-07  allowed_outputs=[] → ValidationError
# ---------------------------------------------------------------------------

def test_ao_07_empty_allowed_outputs_raises():
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="allowed_outputs"):
        Program.from_dict({
            "name": "bad",
            "steps": [{
                "id": "classify",
                "type": "llm",
                "prompt": "classify: $x",
                "allowed_outputs": [],
            }],
        })


# ---------------------------------------------------------------------------
# AO-08  strip() — whitespace ignored
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ao_08_strip_whitespace():
    program = _program("  refund  ", ["refund", "query"], on_error="fail")
    vm = ExecutionVM(llm=MockLLMAdapter("  refund  "))
    trace = await vm.run(program, context={"input": "x"})
    assert trace.status == TraceStatus.SUCCESS
    # output stored as stripped value
    assert trace.steps[0].output == "refund"


# ---------------------------------------------------------------------------
# AO-09  multi-step, both allowed → both succeed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ao_09_multi_step_both_allowed():
    program = Program.from_dict({
        "name": "multi",
        "steps": [
            {
                "id": "s1",
                "type": "llm",
                "prompt": "step1: $x",
                "output_key": "r1",
                "allowed_outputs": ["yes", "no"],
            },
            {
                "id": "s2",
                "type": "llm",
                "prompt": "step2: $x",
                "output_key": "r2",
                "allowed_outputs": ["approve", "reject"],
            },
        ],
    })
    vm = ExecutionVM(llm=MockLLMAdapter(["yes", "approve"]))
    trace = await vm.run(program, context={"x": "test"})
    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].output == "yes"
    assert trace.steps[1].output == "approve"


# ---------------------------------------------------------------------------
# AO-10  case-sensitive: 'Yes' != 'yes' → fail
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ao_10_case_sensitive():
    program = _program("Yes", ["yes", "no"], on_error="fail")
    vm = ExecutionVM(llm=MockLLMAdapter("Yes"))
    with pytest.raises(VMError, match="allowed_outputs"):
        await vm.run(program, context={"input": "x"})
