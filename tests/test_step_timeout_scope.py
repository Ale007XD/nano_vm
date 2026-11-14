"""TT-01..: timeout_seconds / on_timeout are accepted only on llm steps (gap G-20)."""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import ValidationError

from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.adapters import MockLLMAdapter

_MSG = "only valid for llm steps"


def _prog(step: dict[str, object]) -> Program:
    return Program.from_dict({"name": "p", "steps": [step]})


# TT-01: a non-llm step with timeout_seconds is refused at parse time, naming the step
@pytest.mark.parametrize(
    "step",
    [
        {"id": "a", "type": "tool", "tool": "t", "timeout_seconds": 0.1},
        {"id": "a", "type": "tool", "tool": "t", "timeout_seconds": 0.1, "on_timeout": "fail"},
        {
            "id": "a",
            "type": "condition",
            "condition": "$x == 1",
            "then": "b",
            "timeout_seconds": 1,
        },
        {
            "id": "a",
            "type": "parallel",
            "timeout_seconds": 1,
            "parallel_steps": [{"id": "s", "type": "tool", "tool": "t"}],
        },
    ],
)
def test_tt01_non_llm_step_with_timeout_refused(step: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match=_MSG) as exc:
        _prog(step)
    assert "Step 'a'" in str(exc.value)
    assert "inside the tool" in str(exc.value)  # tells where the timeout belongs


# TT-02: on_timeout other than the default is refused too; the explicit default is harmless
def test_tt02_on_timeout_on_non_llm_step() -> None:
    with pytest.raises(ValidationError, match=_MSG):
        _prog({"id": "a", "type": "tool", "tool": "t", "on_timeout": "fallback"})
    ok = _prog({"id": "a", "type": "tool", "tool": "t", "on_timeout": "fail", "is_terminal": True})
    assert ok.steps[0].timeout_seconds is None


# TT-03: parallel sub-steps are validated like any step
def test_tt03_parallel_sub_steps() -> None:
    with pytest.raises(ValidationError, match=_MSG):
        _prog(
            {
                "id": "par",
                "type": "parallel",
                "parallel_steps": [{"id": "s", "type": "tool", "tool": "t", "timeout_seconds": 1}],
            }
        )
    ok = _prog(
        {
            "id": "par",
            "type": "parallel",
            "is_terminal": True,
            "parallel_steps": [
                {
                    "id": "s",
                    "type": "llm",
                    "prompt": "x",
                    "timeout_seconds": 5,
                    "on_timeout": "fallback",
                }
            ],
        }
    )
    assert ok.steps[0].parallel_steps[0].timeout_seconds == 5


# TT-04: llm steps keep working: timeout is enforced, fallback is applied
@pytest.mark.asyncio
async def test_tt04_llm_timeout_unchanged() -> None:
    class Slow:
        async def complete(self, messages: list[dict[str, str]], **kw: object) -> str:
            await asyncio.sleep(1.0)
            return "late"

    prog = _prog(
        {"id": "a", "type": "llm", "prompt": "x", "timeout_seconds": 0.1, "is_terminal": True}
    )
    t0 = time.perf_counter()
    trace = await ExecutionVM(llm=Slow()).run(prog)  # type: ignore[arg-type]
    assert trace.status == TraceStatus.FAILED and time.perf_counter() - t0 < 0.6

    fast = _prog(
        {"id": "a", "type": "llm", "prompt": "x", "timeout_seconds": 5, "is_terminal": True}
    )
    done = await ExecutionVM(llm=MockLLMAdapter("ok")).run(fast)
    assert done.status == TraceStatus.SUCCESS and done.steps[0].output == "ok"


# TT-05: programs without the fields are unaffected, and a dumped program reloads
def test_tt05_plain_program_roundtrip() -> None:
    prog = Program.from_dict(
        {
            "name": "p",
            "steps": [
                {"id": "a", "type": "tool", "tool": "t"},
                {
                    "id": "b",
                    "type": "llm",
                    "prompt": "x",
                    "timeout_seconds": 2,
                    "is_terminal": True,
                },
            ],
        }
    )
    assert Program.model_validate(prog.model_dump()).steps[1].timeout_seconds == 2
