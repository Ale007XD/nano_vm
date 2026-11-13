"""SF-01..: VM output substitution is visible in StepResult and Receipt (gap G-32)."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from pydantic import ValidationError

from nano_vm import ExecutionVM, Program, TraceAnalyzer, TraceStatus
from nano_vm.adapters import MockLLMAdapter
from nano_vm.models import (
    FallbackReason,
    StepFallback,
    StepResult,
    StepStatus,
    Trace,
)
from nano_vm.vm import _check_allowed_outputs


def _prog(**extra: object) -> Program:
    step: dict[str, object] = {
        "id": "cls",
        "type": "llm",
        "prompt": "x",
        "output_key": "d",
        "allowed_outputs": ["yes", "no"],
        "is_terminal": True,
    }
    step.update(extra)
    return Program.from_dict({"name": "c", "steps": [step]})


class _Slow:
    async def complete(self, messages: list[dict[str, str]], **kw: object) -> str:
        await asyncio.sleep(1.0)
        return "no"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# SF-01: model is frozen, reason is an enum
def test_sf01_model() -> None:
    f = StepFallback(reason="timeout")  # type: ignore[arg-type]
    assert f.reason is FallbackReason.TIMEOUT and f.raw_output_sha256 is None
    with pytest.raises(ValidationError):
        f.raw_output_sha256 = "x"  # type: ignore[misc]  # frozen
    with pytest.raises(ValidationError):
        StepFallback(reason="whatever")  # type: ignore[arg-type]


# SF-02: on_error=skip substitution carries a marker with the hash of the raw answer
@pytest.mark.asyncio
async def test_sf02_skip_substitution_marked() -> None:
    raw = "  GARBAGE \n"
    trace = await ExecutionVM(llm=MockLLMAdapter(raw)).run(_prog(on_error="skip"))
    s = trace.steps[0]
    assert s.status == StepStatus.SUCCESS and s.output == "yes"
    assert s.error is None and s.retries == 0  # the marker is not an error
    assert s.fallback is not None
    assert s.fallback.reason is FallbackReason.ALLOWED_OUTPUTS_MISMATCH
    assert s.fallback.raw_output_sha256 == _sha(raw)  # raw, before strip


# SF-03: a genuine answer (also with whitespace around) carries no marker
@pytest.mark.asyncio
async def test_sf03_genuine_answer_not_marked() -> None:
    for answer in ("yes", "  yes\n"):
        trace = await ExecutionVM(llm=MockLLMAdapter(answer)).run(_prog(on_error="skip"))
        assert trace.steps[0].output == "yes" and trace.steps[0].fallback is None


# SF-04: on_timeout=fallback is marked; no answer exists, so no hash
@pytest.mark.asyncio
async def test_sf04_timeout_fallback_marked() -> None:
    trace = await ExecutionVM(llm=_Slow()).run(_prog(timeout_seconds=0.1, on_timeout="fallback"))
    s = trace.steps[0]
    assert s.status == StepStatus.SUCCESS and s.output == "yes" and s.error is None
    assert s.fallback == StepFallback(reason=FallbackReason.TIMEOUT)


@pytest.mark.asyncio
async def test_sf04b_timeout_fallback_without_allowed_outputs_marked() -> None:
    prog = Program.from_dict(
        {
            "name": "c",
            "steps": [
                {
                    "id": "ask",
                    "type": "llm",
                    "prompt": "x",
                    "timeout_seconds": 0.1,
                    "on_timeout": "fallback",
                    "is_terminal": True,
                }
            ],
        }
    )
    trace = await ExecutionVM(llm=_Slow()).run(prog)
    s = trace.steps[0]
    assert s.output == "" and s.fallback is not None
    assert s.fallback.reason is FallbackReason.TIMEOUT


# SF-05: no substitution -> no marker (fail mode, retry that ends with a genuine answer)
@pytest.mark.asyncio
async def test_sf05_no_marker_without_substitution() -> None:
    failed = await ExecutionVM(llm=MockLLMAdapter("GARBAGE")).run(_prog())
    assert failed.status == TraceStatus.FAILED
    assert failed.steps[0].status == StepStatus.FAILED and failed.steps[0].fallback is None

    retried = await ExecutionVM(llm=MockLLMAdapter(["GARBAGE", "no"])).run(
        _prog(on_error="retry", max_retries=3)
    )
    s = retried.steps[0]
    assert s.status == StepStatus.SUCCESS and s.output == "no" and s.retries == 1
    assert s.fallback is None  # earlier rejected attempt must not leave a marker


# SF-06: receipt counts only completed substituted steps
@pytest.mark.asyncio
async def test_sf06_receipt_counter() -> None:
    genuine = await ExecutionVM(llm=MockLLMAdapter("yes")).run(_prog(on_error="skip"))
    subst = await ExecutionVM(llm=MockLLMAdapter("GARBAGE")).run(_prog(on_error="skip"))
    assert TraceAnalyzer(genuine).receipt().fallbacks_used == 0
    assert TraceAnalyzer(subst).receipt().fallbacks_used == 1


def test_sf06b_receipt_ignores_marker_on_non_success_results() -> None:
    fb = StepFallback(reason=FallbackReason.TIMEOUT)
    t = Trace(program_name="p", status=TraceStatus.SUCCESS)
    for sid, st in (
        ("ok", StepStatus.SUCCESS),
        ("bad", StepStatus.FAILED),
        ("skip", StepStatus.SKIPPED),
        ("wait", StepStatus.PENDING),
    ):
        t = t.add_step(StepResult(step_id=sid, status=st, fallback=fb))
    assert TraceAnalyzer(t).receipt().fallbacks_used == 1


# SF-07: documented property — the marker is not part of the state fingerprint
@pytest.mark.asyncio
async def test_sf07_marker_does_not_change_trace_hash() -> None:
    genuine = await ExecutionVM(llm=MockLLMAdapter("yes")).run(_prog(on_error="skip"))
    subst = await ExecutionVM(llm=MockLLMAdapter("GARBAGE")).run(_prog(on_error="skip"))
    rg, rs = TraceAnalyzer(genuine).receipt(), TraceAnalyzer(subst).receipt()
    assert rg.trace_hash == rs.trace_hash
    assert (rg.fallbacks_used, rs.fallbacks_used) == (0, 1)


# SF-08: parallel llm sub-steps are marked too (they are separate StepResults)
@pytest.mark.asyncio
async def test_sf08_parallel_sub_step_marked() -> None:
    prog = Program.from_dict(
        {
            "name": "par",
            "steps": [
                {
                    "id": "par",
                    "type": "parallel",
                    "is_terminal": True,
                    "parallel_steps": [
                        {
                            "id": "a",
                            "type": "llm",
                            "prompt": "x",
                            "allowed_outputs": ["yes", "no"],
                            "on_error": "skip",
                        },
                        {"id": "b", "type": "llm", "prompt": "y"},
                    ],
                }
            ],
        }
    )
    trace = await ExecutionVM(llm=MockLLMAdapter("GARBAGE")).run(prog)
    by_id = {s.step_id: s for s in trace.steps}
    assert by_id["a"].fallback is not None and by_id["a"].output == "yes"
    assert by_id["b"].fallback is None and by_id["b"].output == "GARBAGE"
    assert by_id["par"].fallback is None
    assert TraceAnalyzer(trace).receipt().fallbacks_used == 1


# SF-09: serialisation compatibility (old traces load, new ones round-trip)
def test_sf09_serialisation_compat() -> None:
    old = {"step_id": "a", "status": "success", "output": 1}
    assert StepResult.model_validate(old).fallback is None
    r = StepResult(
        step_id="a",
        status=StepStatus.SUCCESS,
        fallback=StepFallback(
            reason=FallbackReason.ALLOWED_OUTPUTS_MISMATCH, raw_output_sha256="h"
        ),
    )
    back = StepResult.model_validate(json.loads(r.model_dump_json()))
    assert back.fallback == r.fallback


# SF-10: helper keeps its old call shape (no sink argument)
def test_sf10_helper_backward_compatible() -> None:
    step = _prog(on_error="skip").steps[0]
    assert _check_allowed_outputs(step, "GARBAGE") == "yes"
    assert _check_allowed_outputs(step, " no ") == "no"


# SF-11: programs without substitution are unchanged
@pytest.mark.asyncio
async def test_sf11_plain_program_counts_zero() -> None:
    prog = Program.from_dict(
        {"name": "p", "steps": [{"id": "a", "type": "tool", "tool": "t", "is_terminal": True}]}
    )
    t = await ExecutionVM(tools={"t": lambda **kw: 1}).run(prog)
    assert TraceAnalyzer(t).receipt().fallbacks_used == 0
