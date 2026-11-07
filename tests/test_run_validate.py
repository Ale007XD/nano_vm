"""RV-01..: opt-in `validate=True` pre-flight on run()/resume_with_program() (gap G-07)."""

from __future__ import annotations

import pytest

from nano_vm import (
    ExecutionVM,
    InMemoryCursorRepository,
    Program,
    ProgramValidationError,
    TraceStatus,
    VMError,
    WebhookEvent,
)
from nano_vm.validator import IssueKind

# 'audit' is declared after a branch without next_step: unreachable (error)
INVALID = {
    "name": "invalid",
    "max_steps": 20,
    "steps": [
        {"id": "chk", "type": "tool", "tool": "spy"},
        {
            "id": "gate",
            "type": "condition",
            "condition": "$c > 0",
            "then": "approve",
            "otherwise": "deny",
        },
        {"id": "approve", "type": "tool", "tool": "spy"},
        {"id": "deny", "type": "tool", "tool": "spy"},
        {"id": "audit", "type": "tool", "tool": "spy", "is_terminal": True},
    ],
}
VALID = {
    "name": "valid",
    "steps": [
        {"id": "a", "type": "tool", "tool": "spy"},
        {"id": "b", "type": "tool", "tool": "spy", "is_terminal": True},
    ],
}
SUSPEND_THEN_INVALID_TAIL = {
    "name": "suspend_tail",
    "steps": [
        {"id": "p", "type": "tool", "tool": "pend"},
        {"id": "q", "type": "tool", "tool": "spy", "is_terminal": True},
    ],
}


def _spy() -> tuple[list[str], object]:
    calls: list[str] = []

    def spy(**kw: object) -> str:
        calls.append("x")
        return "ok"

    return calls, spy


@pytest.mark.asyncio
async def test_rv01_default_is_opt_out_invalid_program_still_runs() -> None:
    calls, spy = _spy()
    trace = await ExecutionVM(tools={"spy": spy}).run(Program.from_dict(INVALID), {"c": 1})
    assert trace.status == TraceStatus.SUCCESS and calls


@pytest.mark.asyncio
async def test_rv02_validate_rejects_before_any_side_effect() -> None:
    calls, spy = _spy()
    vm = ExecutionVM(tools={"spy": spy})
    with pytest.raises(ProgramValidationError) as ei:
        await vm.run(Program.from_dict(INVALID), {"c": 1}, validate=True)
    assert calls == [], "no step may execute when the program is rejected"
    assert [i.step_id for i in ei.value.report.by_kind(IssueKind.UNREACHABLE_STEP)] == ["audit"]
    assert "audit" in str(ei.value)


@pytest.mark.asyncio
async def test_rv03_valid_program_runs_identically_with_validate() -> None:
    calls1, spy1 = _spy()
    calls2, spy2 = _spy()
    t1 = await ExecutionVM(tools={"spy": spy1}).run(Program.from_dict(VALID))
    t2 = await ExecutionVM(tools={"spy": spy2}).run(Program.from_dict(VALID), validate=True)
    assert t1.status == t2.status == TraceStatus.SUCCESS
    assert [s.step_id for s in t1.steps] == [s.step_id for s in t2.steps]
    assert len(calls1) == len(calls2) == 2


@pytest.mark.asyncio
async def test_rv04_error_type_and_crash_trace() -> None:
    vm = ExecutionVM(tools={"spy": _spy()[1]})
    with pytest.raises(VMError) as ei:  # subclass of VMError
        await vm.run(Program.from_dict(INVALID), validate=True)
    assert isinstance(ei.value, ProgramValidationError)
    assert vm.crash_trace is None


@pytest.mark.asyncio
async def test_rv05_resume_with_program_validates_before_cursor_is_consumed() -> None:
    class CountingRepo(InMemoryCursorRepository):
        def __init__(self) -> None:
            super().__init__()
            self.loads = 0

        async def load(self, trace_id: str):  # type: ignore[override]
            self.loads += 1
            return await super().load(trace_id)

    repo = CountingRepo()
    calls, spy = _spy()
    tools = {"pend": lambda **kw: "PENDING", "spy": spy}
    vm = ExecutionVM(tools=tools, cursor_repository=repo)
    suspended = await vm.run(Program.from_dict(SUSPEND_THEN_INVALID_TAIL))
    assert suspended.status == TraceStatus.SUSPENDED

    event = WebhookEvent(trace_id=suspended.trace_id, payload={})
    with pytest.raises(ProgramValidationError):
        await vm.resume_with_program(event, Program.from_dict(INVALID), validate=True)
    assert repo.loads == 0, "validation must fire before the cursor is loaded"
    assert calls == []

    # cursor survived the rejection: the valid program still resumes
    resumed = await vm.resume_with_program(
        event, Program.from_dict(SUSPEND_THEN_INVALID_TAIL), validate=True
    )
    assert resumed.status == TraceStatus.SUCCESS


@pytest.mark.asyncio
async def test_rv06_validate_is_keyword_only() -> None:
    vm = ExecutionVM(tools={"spy": _spy()[1]})
    with pytest.raises(TypeError):
        await vm.run(Program.from_dict(VALID), {}, True)  # type: ignore[misc]
