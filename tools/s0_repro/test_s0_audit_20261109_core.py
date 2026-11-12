"""S0 repro pack — nano_vm, аудит 2026-11-09, база bc94abd (v0.8.12).

Назначение: verify-first. RED-тесты фиксируют ЖЕЛАЕМОЕ свойство и должны быть
красными на bc94abd; CH-тесты — характеризация текущего поведения (зелёные),
их меняют осознанно вместе с фиксом. Тесты не навязывают конкретный API фикса:
допустимо любое решение, при котором свойство выполняется (в т.ч. отказ на
parse/validate там, где это отмечено в докстринге).

Размещение: tools/s0_repro/ (вне testpaths, не влияет на CI до фикса).
Запуск:  python -m pytest tools/s0_repro/test_s0_audit_20261109_core.py -q -p no:cacheprovider
Ожидание на bc94abd: RED = CL-01, TO-01, FB-01, FB-02, PR-01, RJ-02; остальные GREEN.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from pydantic import ValidationError

from nano_vm import (
    ExecutionVM,
    InMemoryCursorRepository,
    Program,
    ProgramValidationError,
    ResumeError,
    TraceAnalyzer,
    TraceStatus,
    WebhookEvent,
)
from nano_vm.adapters import MockLLMAdapter

_VOLATILE = {"duration_ms", "finished_at", "started_at", "usage"}


def _evidence(trace) -> tuple:
    """Всё, что Trace/Receipt сообщает о шагах, без времени/токенов/ID."""
    steps = [
        {k: v for k, v in s.model_dump(mode="json").items() if k not in _VOLATILE}
        for s in trace.steps
    ]
    r = TraceAnalyzer(trace).receipt()
    counters = tuple(
        repr(getattr(r, n, None))
        for n in (
            "failed_steps",
            "retried_steps",
            "rejected_transitions",
            "escalations",
            "blocked_actions",
        )
    )
    return repr(steps), counters


# ───────────────────────── G-29 · атомарность claim курсора ─────────────────────────


class _IORepo(InMemoryCursorRepository):
    """Любой репозиторий с реальным I/O: load/delete уступают event loop."""

    async def load(self, trace_id):
        await asyncio.sleep(0.01)
        return await super().load(trace_id)

    async def delete(self, trace_id):
        await asyncio.sleep(0.01)
        return await super().delete(trace_id)


_PAY = {
    "name": "pay",
    "steps": [
        {"id": "init", "type": "tool", "tool": "initiate"},
        {"id": "cap", "type": "tool", "tool": "capture", "is_terminal": True},
    ],
}


def test_cl01_concurrent_resume_executes_step_once():
    """CL-01 [G-29, RED]: два одновременных вебхука на один trace_id → шаг после
    SUSPENDED выполняется ровно один раз (второй получает ResumeError)."""
    charges: list[int] = []

    async def initiate(**k):
        return "PENDING"

    async def capture(**k):
        charges.append(1)
        await asyncio.sleep(0.05)
        return "captured"

    async def scenario():
        repo = _IORepo()
        tools = {"initiate": initiate, "capture": capture}
        prog = Program.from_dict(_PAY)
        tr = await ExecutionVM(tools=tools, cursor_repository=repo).run(prog)
        assert tr.status == TraceStatus.SUSPENDED

        def ev() -> WebhookEvent:
            return WebhookEvent(trace_id=tr.trace_id, payload={"type": "payment.confirmed"})

        a = ExecutionVM(tools=tools, cursor_repository=repo)
        b = ExecutionVM(tools=tools, cursor_repository=repo)
        await asyncio.gather(
            a.resume_with_program(ev(), prog),
            b.resume_with_program(ev(), prog),
            return_exceptions=True,
        )

    asyncio.run(scenario())
    assert len(charges) == 1, f"capture выполнен {len(charges)} раз(а), ожидалось 1"


def test_cl02_sequential_second_resume_is_rejected():
    """CL-02 [G-29, CH]: последовательный повторный resume отклоняется (one-shot работает
    без конкуренции) — проблема только в конкурентном окне load→delete."""

    async def initiate(**k):
        return "PENDING"

    async def capture(**k):
        return "ok"

    async def scenario():
        repo = InMemoryCursorRepository()
        tools = {"initiate": initiate, "capture": capture}
        prog = Program.from_dict(_PAY)
        tr = await ExecutionVM(tools=tools, cursor_repository=repo).run(prog)
        ev = WebhookEvent(trace_id=tr.trace_id, payload={})
        await ExecutionVM(tools=tools, cursor_repository=repo).resume_with_program(ev, prog)
        with pytest.raises(ResumeError):
            await ExecutionVM(tools=tools, cursor_repository=repo).resume_with_program(ev, prog)

    asyncio.run(scenario())


# ───────────────────────── G-20 · таймаут TOOL-шага ─────────────────────────


def test_to01_tool_timeout_is_rejected_or_enforced():
    """TO-01 [G-20, RED]: timeout_seconds на TOOL-шаге не может быть принят и молча
    проигнорирован. Допустимо: (A) отказ на parse/validate; (B) реальное прерывание."""

    async def slow(**k):
        await asyncio.sleep(1.0)
        return "done"

    async def scenario():
        try:
            prog = Program.from_dict(
                {
                    "name": "t",
                    "steps": [
                        {
                            "id": "a",
                            "type": "tool",
                            "tool": "slow",
                            "timeout_seconds": 0.1,
                            "on_timeout": "fail",
                            "is_terminal": True,
                        }
                    ],
                }
            )
            t0 = time.perf_counter()
            trace = await ExecutionVM(tools={"slow": slow}).run(prog, validate=True)
            return trace, time.perf_counter() - t0
        except (ValidationError, ProgramValidationError):
            return None, 0.0

    trace, dt = asyncio.run(scenario())
    if trace is None:
        return  # вариант A
    assert trace.status != TraceStatus.SUCCESS and dt < 0.6, (
        f"timeout_seconds=0.1 проигнорирован: status={trace.status.name}, {dt:.2f}s"
    )


# ───────────────────────── G-32 · подмена вывода видна в Trace/Receipt ─────────────────────────


def _classify_prog(**extra):
    step = {
        "id": "cls",
        "type": "llm",
        "prompt": "x",
        "output_key": "d",
        "allowed_outputs": ["yes", "no"],
        "is_terminal": True,
    }
    step.update(extra)
    return Program.from_dict({"name": "c", "steps": [step]})


def test_fb01_skip_substitution_is_distinguishable_from_genuine_output():
    """FB-01 [G-32, RED]: on_error=skip с подстановкой allowed_outputs[0] отличим в
    Trace/Receipt от случая, когда модель сама ответила тем же значением."""

    async def scenario():
        genuine = await ExecutionVM(llm=MockLLMAdapter("yes")).run(_classify_prog(on_error="skip"))
        substituted = await ExecutionVM(llm=MockLLMAdapter("GARBAGE")).run(
            _classify_prog(on_error="skip")
        )
        return genuine, substituted

    genuine, substituted = asyncio.run(scenario())
    assert genuine.steps[0].output == substituted.steps[0].output == "yes"
    assert _evidence(genuine) != _evidence(substituted), "подстановка неотличима от ответа модели"


def test_fb02_timeout_fallback_is_distinguishable_from_genuine_output():
    """FB-02 [G-32, RED]: то же для on_timeout=fallback."""

    class _Slow:
        async def complete(self, messages):
            await asyncio.sleep(1.0)
            return "no"

    async def scenario():
        genuine = await ExecutionVM(llm=MockLLMAdapter("yes")).run(
            _classify_prog(timeout_seconds=0.1, on_timeout="fallback")
        )
        fallback = await ExecutionVM(llm=_Slow()).run(
            _classify_prog(timeout_seconds=0.1, on_timeout="fallback")
        )
        return genuine, fallback

    genuine, fallback = asyncio.run(scenario())
    assert genuine.steps[0].output == fallback.steps[0].output == "yes"
    assert _evidence(genuine) != _evidence(fallback), "fallback неотличим от ответа модели"


# ───────────────────────── G-31 · retry и побочные эффекты ─────────────────────────


def test_pr01_parallel_block_retry_does_not_reexecute_succeeded_branches():
    """PR-01 [G-31, RED]: при retry parallel-блока уже успешные ветки с эффектами не
    выполняются повторно. Допустимо: валидатор запрещает retry на parallel (вариант A)."""
    calls = {"A": 0, "B": 0, "C": 0}

    async def A(**k):
        calls["A"] += 1
        return "a"

    async def B(**k):
        calls["B"] += 1
        if calls["B"] == 1:
            raise RuntimeError("boom")
        return "b"

    async def C(**k):
        calls["C"] += 1
        await asyncio.sleep(0.05)
        return "c"

    prog = Program.from_dict(
        {
            "name": "pr",
            "steps": [
                {
                    "id": "par",
                    "type": "parallel",
                    "on_error": "retry",
                    "max_retries": 3,
                    "parallel_steps": [
                        {"id": "sa", "type": "tool", "tool": "A"},
                        {"id": "sb", "type": "tool", "tool": "B"},
                        {"id": "sc", "type": "tool", "tool": "C"},
                    ],
                    "is_terminal": True,
                }
            ],
        }
    )

    async def scenario():
        try:
            await ExecutionVM(tools={"A": A, "B": B, "C": C}).run(prog, validate=True)
        except ProgramValidationError:
            return "rejected"
        return "ran"

    if asyncio.run(scenario()) == "rejected":
        return
    assert calls["A"] == 1 and calls["C"] == 1, f"эффекты продублированы: {calls}"


def test_pr02_parallel_semantics_characterization():
    """PR-02 [G-31, CH]: при отказе ветки соседи доделываются (не отменяются);
    результаты собираются в порядке объявления независимо от порядка завершения."""
    finished: list[str] = []

    async def slow_ok(**k):
        await asyncio.sleep(0.2)
        finished.append("slow")
        return "S"

    async def fast_ok(**k):
        finished.append("fast")
        return "F"

    async def bad(**k):
        raise RuntimeError("boom")

    async def scenario():
        ok_prog = Program.from_dict(
            {
                "name": "o",
                "steps": [
                    {
                        "id": "p",
                        "type": "parallel",
                        "is_terminal": True,
                        "parallel_steps": [
                            {"id": "first", "type": "tool", "tool": "slow_ok"},
                            {"id": "second", "type": "tool", "tool": "fast_ok"},
                        ],
                    }
                ],
            }
        )
        ok = await ExecutionVM(tools={"slow_ok": slow_ok, "fast_ok": fast_ok}).run(ok_prog)
        fail_prog = Program.from_dict(
            {
                "name": "f",
                "steps": [
                    {
                        "id": "p",
                        "type": "parallel",
                        "is_terminal": True,
                        "parallel_steps": [
                            {"id": "x", "type": "tool", "tool": "bad"},
                            {"id": "y", "type": "tool", "tool": "slow_ok"},
                        ],
                    }
                ],
            }
        )
        finished.clear()
        failed = await ExecutionVM(tools={"bad": bad, "slow_ok": slow_ok}).run(fail_prog)
        return ok, failed

    ok, failed = asyncio.run(scenario())
    assert finished == ["slow"], "соседняя ветка должна доработать до конца"
    assert failed.status == TraceStatus.FAILED
    assert list(ok.final_output.keys()) == ["first", "second"], "порядок = порядок объявления"


def test_id01_tool_retry_gets_identical_kwargs_no_key(monkeypatch):
    """ID-01 [G-31, CH]: при retry tool получает те же kwargs; ключа идемпотентности/номера
    попытки нет. Тест меняют осознанно, когда ключ появится."""
    seen: list[dict] = []
    delays: list[float] = []

    async def fake_sleep(d, *a, **k):
        if d > 0:  # sleep(0) — кооперативный yield, не backoff
            delays.append(d)

    async def pay(**k):
        seen.append(dict(k))
        if len(seen) < 3:
            raise TimeoutError("gateway timeout after charge")
        return "ok"

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    prog = Program.from_dict(
        {
            "name": "r",
            "steps": [
                {
                    "id": "cap",
                    "type": "tool",
                    "tool": "pay",
                    "args": {"amount": 500},
                    "on_error": "retry",
                    "max_retries": 3,
                    "is_terminal": True,
                }
            ],
        }
    )
    trace = asyncio.run(ExecutionVM(tools={"pay": pay}).run(prog))
    assert trace.status == TraceStatus.SUCCESS
    assert seen == [{"amount": 500}] * 3
    assert delays == [1, 2], "backoff 1s, 2s без jitter"  # RJ-01 [G-22, CH]


# ───────────────────────── G-22 · гигиена retry ─────────────────────────


def test_rj02_config_error_is_not_retried_with_backoff(monkeypatch):
    """RJ-02 [G-22, RED]: ошибка конфигурации (tool не зарегистрирован) не ретраится с
    backoff — падает сразу."""
    delays: list[float] = []

    async def fake_sleep(d, *a, **k):
        if d > 0:  # sleep(0) — кооперативный yield, не backoff
            delays.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    prog = Program.from_dict(
        {
            "name": "m",
            "steps": [
                {
                    "id": "a",
                    "type": "tool",
                    "tool": "ghost",
                    "on_error": "retry",
                    "max_retries": 3,
                    "is_terminal": True,
                }
            ],
        }
    )
    trace = asyncio.run(ExecutionVM(tools={}).run(prog))
    assert trace.status == TraceStatus.FAILED
    assert delays == [], f"нет смысла ретраить ошибку конфигурации, backoff={delays}"


# ───────────────────────── G-21 · sync-tool и event loop ─────────────────────────


def _max_heartbeat_gap(tool, seconds: float) -> float:
    beats: list[float] = []

    async def hb():
        while True:
            beats.append(time.perf_counter())
            await asyncio.sleep(0.05)

    async def scenario():
        prog = Program.from_dict(
            {"name": "b", "steps": [{"id": "a", "type": "tool", "tool": "f", "is_terminal": True}]}
        )
        h = asyncio.create_task(hb())
        await asyncio.sleep(0.2)
        await ExecutionVM(tools={"f": tool}).run(prog)
        await asyncio.sleep(0.1)
        h.cancel()

    asyncio.run(scenario())
    return max(b - a for a, b in zip(beats, beats[1:]))


def test_bl01_sync_tool_blocks_event_loop_characterization():
    """BL-01 [G-21, CH]: sync-tool выполняется inline и блокирует loop на всё время вызова;
    async-tool — нет (контрольный замер)."""

    def sync_block(**k):
        time.sleep(0.6)
        return "x"

    async def async_wait(**k):
        await asyncio.sleep(0.6)
        return "x"

    assert _max_heartbeat_gap(sync_block, 0.6) > 0.5
    assert _max_heartbeat_gap(async_wait, 0.6) < 0.3
