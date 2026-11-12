"""CC-01..CC-05 [G-29]: атомарный claim курсора (CursorRepository.delete -> bool).

Свойство: два одновременных resume одного trace_id исполняют шаг после SUSPENDED
ровно один раз; проигравший получает ResumeError. Репозиторий со старым контрактом
(delete -> None) продолжает работать без проверки claim (legacy).
"""

from __future__ import annotations

import asyncio

import pytest

from nano_vm import (
    ExecutionVM,
    InMemoryCursorRepository,
    Program,
    ResumeError,
    Trace,
    TraceStatus,
    WebhookEvent,
)

_PAY = {
    "name": "pay",
    "steps": [
        {"id": "init", "type": "tool", "tool": "initiate"},
        {"id": "cap", "type": "tool", "tool": "capture", "is_terminal": True},
    ],
}


class _IORepo(InMemoryCursorRepository):
    """Репозиторий с реальным I/O: load/delete уступают event loop."""

    async def load(self, trace_id):
        await asyncio.sleep(0.01)
        return await super().load(trace_id)

    async def delete(self, trace_id):
        await asyncio.sleep(0.01)
        return await super().delete(trace_id)


class _LegacyRepo(InMemoryCursorRepository):
    """Старый контракт: delete() ничего не возвращает."""

    async def delete(self, trace_id):
        await super().delete(trace_id)
        return None


class _LostRaceRepo(InMemoryCursorRepository):
    """delete() сообщает, что курсор уже забрал другой resume."""

    async def delete(self, trace_id):
        await super().delete(trace_id)
        return False


def _tools(charges: list[int]):
    async def initiate(**k):
        return "PENDING"

    async def capture(**k):
        charges.append(1)
        await asyncio.sleep(0.05)
        return "captured"

    return {"initiate": initiate, "capture": capture}


async def _suspend(repo, tools):
    prog = Program.from_dict(_PAY)
    trace = await ExecutionVM(tools=tools, cursor_repository=repo).run(prog)
    assert trace.status == TraceStatus.SUSPENDED
    return prog, trace


def _ev(trace) -> WebhookEvent:
    return WebhookEvent(trace_id=trace.trace_id, payload={"type": "payment.confirmed"})


def test_cc01_inmemory_delete_reports_claim():
    """CC-01: первый delete -> True, повторный -> False."""

    async def scenario():
        repo = InMemoryCursorRepository()
        charges: list[int] = []
        _, trace = await _suspend(repo, _tools(charges))
        return await repo.delete(trace.trace_id), await repo.delete(trace.trace_id)

    assert asyncio.run(scenario()) == (True, False)


def test_cc02_concurrent_resumes_execute_step_once():
    """CC-02: N одновременных resume -> шаг один раз, остальные ResumeError."""
    charges: list[int] = []

    async def scenario():
        repo = _IORepo()
        tools = _tools(charges)
        prog, trace = await _suspend(repo, tools)
        return await asyncio.gather(
            *[
                ExecutionVM(tools=tools, cursor_repository=repo).resume_with_program(
                    _ev(trace), prog
                )
                for _ in range(5)
            ],
            return_exceptions=True,
        )

    results = asyncio.run(scenario())
    assert len(charges) == 1, f"capture выполнен {len(charges)} раз(а), ожидалось 1"
    assert sum(isinstance(r, Trace) for r in results) == 1
    assert sum(isinstance(r, ResumeError) for r in results) == 4


def test_cc03_legacy_repository_returning_none_still_resumes():
    """CC-03: delete() -> None (legacy) не ломает resume."""
    charges: list[int] = []

    async def scenario():
        repo = _LegacyRepo()
        tools = _tools(charges)
        prog, trace = await _suspend(repo, tools)
        return await ExecutionVM(tools=tools, cursor_repository=repo).resume_with_program(
            _ev(trace), prog
        )

    assert asyncio.run(scenario()).status == TraceStatus.SUCCESS
    assert len(charges) == 1


def test_cc04_sequential_second_resume_still_rejected():
    """CC-04: последовательный повтор отклоняется (поведение до фикса сохранено)."""

    async def scenario():
        repo = InMemoryCursorRepository()
        tools = _tools([])
        prog, trace = await _suspend(repo, tools)
        await ExecutionVM(tools=tools, cursor_repository=repo).resume_with_program(_ev(trace), prog)
        with pytest.raises(ResumeError):
            await ExecutionVM(tools=tools, cursor_repository=repo).resume_with_program(
                _ev(trace), prog
            )

    asyncio.run(scenario())


def test_cc05_delete_false_raises_resume_error_and_skips_step():
    """CC-05: delete() == False -> ResumeError("already claimed"), шаг не исполнен."""
    charges: list[int] = []

    async def scenario():
        repo = _LostRaceRepo()
        tools = _tools(charges)
        prog, trace = await _suspend(repo, tools)
        with pytest.raises(ResumeError, match="already claimed"):
            await ExecutionVM(tools=tools, cursor_repository=repo).resume_with_program(
                _ev(trace), prog
            )

    asyncio.run(scenario())
    assert charges == []
