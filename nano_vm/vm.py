"""
nano_vm.vm
==========
ExecutionVM: deterministic execution of a Program.

v0.6.0 changes:
  - suspend() / resume() для async webhook-событий (инвариант I5)
  - BudgetInterrupt как отдельный signal (инвариант I7: Budget = Interrupt)
  - _emit_interrupt() hook для vault-layer (escalation routing)
  - CursorRepository protocol для persisted suspend cursor
  - WebhookEvent — типизированный входной контракт resume()

Key properties (unchanged):
  - VM has no knowledge of providers; receives LLMAdapter via __init__
  - Same Program + StateContext + deterministic adapter -> reproducible result
  - Steps execute sequentially; condition steps control branching
  - Parallel steps execute sub-steps concurrently via asyncio.gather
  - Per-step on_error: fail | skip | retry
  - LLM steps populate StepResult.usage (tokens + cost) when available
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

from .adapters.base import LLMAdapter
from .models import (
    InterruptType,
    LLMUsage,
    OnError,
    Program,
    StateContext,
    Step,
    StepResult,
    StepStatus,
    StepType,
    Trace,
    TraceStatus,
)


class VMError(Exception):
    """Program execution error."""


class ResumeError(VMError):
    """Cannot resume: trace not found or not in SUSPENDED state."""


# ---------------------------------------------------------------------------
# v0.6.0: WebhookEvent — typed input contract for resume()
# ---------------------------------------------------------------------------


class WebhookEvent:
    """
    Типизированный входной контракт для VM.resume().

    trace_id: должен совпадать с Trace.trace_id — ключ корреляции.
    payload:  данные от внешней системы (payment confirmed, delivery status, etc.)
    source:   "WEBHOOK" | "OPERATOR" | "TIMER" — для event sourcing (DomainEvent.source)

    Валидация nonce + timestamp + HMAC — ответственность MCPPolicyLayer (P2 roadmap).
    VM.resume() получает уже провалидированный WebhookEvent.
    """

    def __init__(
        self,
        trace_id: str,
        payload: dict[str, Any],
        source: str = "WEBHOOK",
    ) -> None:
        if not trace_id:
            raise ValueError("WebhookEvent.trace_id cannot be empty")
        if source not in ("WEBHOOK", "OPERATOR", "TIMER"):
            raise ValueError(f"WebhookEvent.source must be WEBHOOK|OPERATOR|TIMER, got '{source}'")
        self.trace_id = trace_id
        self.payload = payload
        self.source = source
        self.received_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# v0.6.0: CursorRepository Protocol — persist/load suspend cursor
# ---------------------------------------------------------------------------


class CursorRepository(Protocol):
    """
    Protocol для persisted suspend cursor.

    Реализации:
      SqliteCursorRepository — alpha (infrastructure.db, WAL)
      InMemoryCursorRepository — тесты и dry-run

    Контракт:
      save(trace_id, step_id, state) — сохранить cursor перед suspend
      load(trace_id) — восстановить cursor для resume
      delete(trace_id) — очистить после успешного resume или FAILED terminal
    """

    async def save(
        self,
        trace_id: str,
        step_id: str,
        state: StateContext,
        trace: Trace,
    ) -> None: ...

    async def load(
        self,
        trace_id: str,
    ) -> tuple[str, StateContext, Trace] | None:
        """
        Returns (step_id, state, trace) или None если trace_id не найден.
        """
        ...

    async def delete(self, trace_id: str) -> None: ...


class InMemoryCursorRepository:
    """
    In-memory реализация CursorRepository для тестов и dry-run.
    Не использовать в production: рестарт = потеря курсора = нарушение идемпотентности.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[str, StateContext, Trace]] = {}

    async def save(
        self,
        trace_id: str,
        step_id: str,
        state: StateContext,
        trace: Trace,
    ) -> None:
        self._store[trace_id] = (step_id, state, trace)

    async def load(
        self,
        trace_id: str,
    ) -> tuple[str, StateContext, Trace] | None:
        return self._store.get(trace_id)

    async def delete(self, trace_id: str) -> None:
        self._store.pop(trace_id, None)


# ---------------------------------------------------------------------------
# ExecutionVM
# ---------------------------------------------------------------------------


class ExecutionVM:
    """
    Deterministic virtual machine for executing Programs.

    Args:
        llm:               LLM adapter (any object with async complete() method)
        tools:             tool registry {name: async callable}
        cursor_repository: persisted suspend cursor (default: InMemoryCursorRepository)
                           В production заменить на SqliteCursorRepository (infrastructure.db).

    v0.6.0:
        suspend() / resume() для async webhook-событий.
        _emit_interrupt() для BudgetInterrupt signal (vault escalation hook).

    Example:
        vm = ExecutionVM(
            llm=LiteLLMAdapter("groq/llama-3.3-70b-versatile"),
            tools={"search": my_search_fn},
        )
        trace = await vm.run(program, context={"user_input": "..."})

        # Если шаг вернул PENDING (напр. payment initiation):
        # trace.status == TraceStatus.SUSPENDED
        # trace.trace_id — ключ для resume
        #
        # После получения webhook:
        # event = WebhookEvent(trace_id=trace.trace_id, payload={"status": "confirmed"})
        # trace = await vm.resume(event)
    """

    def __init__(
        self,
        llm: LLMAdapter,
        tools: dict[str, Callable] | None = None,
        cursor_repository: CursorRepository | None = None,
    ) -> None:
        self._llm = llm
        self._tools: dict[str, Callable] = tools or {}
        self._cursor_repo: CursorRepository = cursor_repository or InMemoryCursorRepository()

    def register_tool(self, name: str, fn: Callable) -> None:
        self._tools[name] = fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        program: Program,
        context: dict[str, Any] | None = None,
    ) -> Trace:
        """
        Execute a program.

        Returns Trace with status:
          SUCCESS         — all steps completed
          FAILED          — step failed with on_error=FAIL
          BUDGET_EXCEEDED — max_steps or max_tokens limit reached
          STALLED         — max_stalled_steps consecutive no-ops
          SUSPENDED       — step returned PENDING; call resume() when webhook arrives
        """
        state = StateContext(data=context or {})
        trace = Trace(program_name=program.name)
        return await self._execute_loop(program, state, trace, start_step_id=None)

    async def resume(self, webhook_event: WebhookEvent) -> Trace:
        """
        v0.6.0: Восстанавливает suspended execution по trace_id из WebhookEvent.

        Контракт:
          1. Загружает cursor (step_id, state, trace) из CursorRepository.
          2. Валидирует что trace в SUSPENDED — guard от двойного resume.
          3. Инжектирует webhook payload в StateContext под ключом "__webhook__".
          4. Продолжает execution loop со step после suspended_step_id.
          5. Удаляет cursor после завершения (SUCCESS/FAILED/SUSPENDED снова).

        Raises:
          ResumeError: если trace не найден или уже не в SUSPENDED.
        """
        cursor = await self._cursor_repo.load(webhook_event.trace_id)
        if cursor is None:
            raise ResumeError(
                f"No suspended trace found for trace_id='{webhook_event.trace_id}'. "
                "Possible causes: already resumed, cursor not persisted (production requires "
                "SqliteCursorRepository), or invalid trace_id."
            )

        suspended_step_id, state, trace = cursor

        if trace.status != TraceStatus.SUSPENDED:
            raise ResumeError(
                f"Trace '{webhook_event.trace_id}' is not in SUSPENDED state "
                f"(current: {trace.status}). Cannot resume."
            )

        # Инжектируем webhook payload в context для downstream steps
        state = state.with_data("__webhook__", webhook_event.payload)
        state = state.with_data("__webhook_source__", webhook_event.source)

        # Восстанавливаем программу — нужна для продолжения loop
        # vault-layer передаёт program через resume; здесь используем имя из trace
        # для поиска в registry (P8). Пока: program передаётся явно через resume_with_program.
        # TODO(P8): Blueprint registry -> program lookup by trace.program_name

        await self._cursor_repo.delete(webhook_event.trace_id)

        # Продолжаем loop начиная со следующего шага после suspended_step_id
        # program недоступен без registry — vault вызывает resume_with_program напрямую
        raise ResumeError(
            "resume() requires program context. Use resume_with_program(event, program) "
            "until Blueprint registry (P8) is implemented."
        )

    async def resume_with_program(
        self,
        webhook_event: WebhookEvent,
        program: Program,
    ) -> Trace:
        """
        v0.6.0: resume() с явной передачей Program.

        Используется vault-layer до реализации Blueprint registry (P8).
        После P8: resume(event) найдёт программу через registry по trace.program_name.

        Replay-safe: cursor удаляется перед продолжением.
        Двойной вызов resume_with_program с одним trace_id → ResumeError (cursor уже удалён).
        """
        cursor = await self._cursor_repo.load(webhook_event.trace_id)
        if cursor is None:
            raise ResumeError(
                f"No suspended trace for trace_id='{webhook_event.trace_id}'. "
                "Already resumed or never suspended."
            )

        suspended_step_id, state, trace = cursor

        if trace.status != TraceStatus.SUSPENDED:
            raise ResumeError(
                f"Trace '{webhook_event.trace_id}' is not SUSPENDED (current: {trace.status})."
            )

        # Инжектируем webhook payload
        state = state.with_data("__webhook__", webhook_event.payload)
        state = state.with_data("__webhook_source__", webhook_event.source)

        # Удаляем cursor до продолжения — replay protection
        # Если execution снова вернёт PENDING (multi-step async), cursor будет пересохранён.
        await self._cursor_repo.delete(webhook_event.trace_id)

        # Продолжаем со следующего шага
        return await self._execute_loop(
            program=program,
            state=state,
            trace=trace,
            start_step_id=suspended_step_id,
            resume_after=True,  # пропустить suspended_step_id, начать со следующего
        )

    # ------------------------------------------------------------------
    # Core execution loop
    # ------------------------------------------------------------------

    async def _execute_loop(
        self,
        program: Program,
        state: StateContext,
        trace: Trace,
        start_step_id: str | None,
        resume_after: bool = False,
    ) -> Trace:
        """
        Основной execution loop. Используется и run(), и resume_with_program().

        start_step_id=None: начать с первого шага (run()).
        start_step_id=X, resume_after=True: начать со шага после X (resume).
        """
        step_index = {s.id: i for i, s in enumerate(program.steps)}
        steps = program.steps

        # Определяем стартовый индекс
        if start_step_id is None:
            current_idx = 0
        else:
            if start_step_id not in step_index:
                return trace.finish(
                    TraceStatus.FAILED,
                    error=f"resume: step_id '{start_step_id}' not found in program",
                )
            current_idx = step_index[start_step_id]
            if resume_after:
                current_idx += 1  # продолжаем со следующего шага

        steps_executed = len(trace.steps)  # учитываем уже выполненные шаги при resume
        last_fingerprint: int | None = None
        stalled_count = 0

        while current_idx < len(steps):
            # Budget guards — до исполнения шага (I7: Budget = Interrupt)
            if program.max_steps is not None and steps_executed >= program.max_steps:
                await self._emit_interrupt(InterruptType.BUDGET, trace)
                return trace.finish(
                    TraceStatus.BUDGET_EXCEEDED,
                    error=f"max_steps={program.max_steps} exceeded after {steps_executed} step(s)",
                )

            if program.max_tokens is not None:
                tokens_used = trace.total_tokens()
                if tokens_used >= program.max_tokens:
                    await self._emit_interrupt(InterruptType.BUDGET, trace)
                    return trace.finish(
                        TraceStatus.BUDGET_EXCEEDED,
                        error=(
                            f"max_tokens={program.max_tokens} exceeded: "
                            f"{tokens_used} tokens consumed"
                        ),
                    )

            step = steps[current_idx]
            result, state, sub_results = await self._run_step(step, state)
            steps_executed += 1

            # Fingerprint-based no-op detection
            current_fp = self._state_fingerprint(state)
            if last_fingerprint is not None and current_fp == last_fingerprint:
                stalled_count += 1
            else:
                stalled_count = 0
            last_fingerprint = current_fp

            if program.max_stalled_steps is not None and stalled_count >= program.max_stalled_steps:
                return trace.finish(
                    TraceStatus.STALLED,
                    error=(
                        f"max_stalled_steps={program.max_stalled_steps} exceeded: "
                        f"{stalled_count} consecutive no-op step(s)"
                    ),
                )

            trace = trace.add_snapshot(
                steps_executed - 1,
                self._state_fingerprint_hex(state),
            )

            for sub_result in sub_results:
                trace = trace.add_step(sub_result)
            trace = trace.add_step(result)

            # v0.6.0: PENDING → suspend, вернуть SUSPENDED trace
            if result.status == StepStatus.PENDING:
                trace = await self._suspend(step, state, trace)
                return trace

            if result.status == StepStatus.FAILED:
                return trace.finish(
                    TraceStatus.FAILED,
                    error=f"Step '{step.id}' failed: {result.error}",
                )

            # Condition step: jump to then/otherwise
            if step.type == StepType.CONDITION and result.status == StepStatus.SUCCESS:
                next_id = result.output

                if not next_id:
                    return trace.finish(
                        TraceStatus.FAILED,
                        error=f"Step '{step.id}': condition produced no branch target",
                    )

                if next_id not in step_index:
                    return trace.finish(
                        TraceStatus.FAILED,
                        error=f"Step '{step.id}': condition target '{next_id}' not found",
                    )

                target_step = steps[step_index[next_id]]
                steps_executed += 1
                target_result, state, target_sub = await self._run_step(target_step, state)
                for sub_result in target_sub:
                    trace = trace.add_step(sub_result)
                trace = trace.add_step(target_result)

                if target_result.status == StepStatus.PENDING:
                    trace = await self._suspend(target_step, state, trace)
                    return trace

                if target_result.status == StepStatus.FAILED:
                    return trace.finish(
                        TraceStatus.FAILED,
                        error=f"Step '{target_step.id}' failed: {target_result.error}",
                    )
                return trace.finish(TraceStatus.SUCCESS, final_output=trace.last_output())

            current_idx += 1

        return trace.finish(TraceStatus.SUCCESS, final_output=trace.last_output())

    # ------------------------------------------------------------------
    # v0.6.0: suspend / interrupt hooks
    # ------------------------------------------------------------------

    async def _suspend(
        self,
        step: Step,
        state: StateContext,
        trace: Trace,
    ) -> Trace:
        """
        Переводит VM в режим ожидания webhook.

        1. Обновляет Trace → SUSPENDED (cursor зафиксирован).
        2. Сохраняет cursor в CursorRepository (persisted в production).
        3. Возвращает SUSPENDED Trace вызывающему коду.

        Caller (vault ExecutionVM wrapper) регистрирует webhook handler
        с trace_id для последующего вызова resume_with_program().
        """
        trace = trace.suspend(step.id)
        await self._cursor_repo.save(
            trace_id=trace.trace_id,
            step_id=step.id,
            state=state,
            trace=trace,
        )
        return trace

    async def _emit_interrupt(
        self,
        interrupt_type: InterruptType,
        trace: Trace,
    ) -> None:
        """
        v0.6.0: Эмиттит interrupt signal.

        Инвариант I7: Budget трактуется как Interrupt, не control-flow условие.
        Текущая реализация: no-op hook для vault-layer.
        vault.execution.budget.BudgetInterrupt подписывается на этот hook
        и маршрутизирует в FSM.ESCALATED через SagaCoordinator.

        Сигнатура стабильна — vault может переопределить через subclass или callback.
        """
        # Hook point для vault-layer escalation.
        # В standalone использовании (без vault) — логирование достаточно.
        pass

    # ------------------------------------------------------------------
    # Step execution (без изменений относительно v0.5.0)
    # ------------------------------------------------------------------

    async def _run_step(
        self,
        step: Step,
        state: StateContext,
    ) -> tuple[StepResult, StateContext, list[StepResult]]:
        result = StepResult(step_id=step.id, status=StepStatus.RUNNING)
        attempt = 0

        while True:
            try:
                output, usage, sub_results = await self._execute_step(step, state)

                # v0.6.0: tool возвращает "PENDING" string → PENDING статус
                # Это позволяет tool (через MCP) сигнализировать async ожидание
                if output == "PENDING" and step.type == StepType.TOOL:
                    result = result.model_copy(
                        update={"status": StepStatus.PENDING, "output": output}
                    )
                    return result, state, sub_results

                result = result.finish(output=output, usage=usage)

                if step.output_key:
                    state = state.with_data(step.output_key, output)
                state = state.with_output(step.id, output)

                if step.type == StepType.PARALLEL and isinstance(output, dict):
                    for sub_id, sub_out in output.items():
                        state = state.with_output(sub_id, sub_out)

                return result, state, sub_results

            except Exception as exc:
                attempt += 1
                if step.on_error == OnError.RETRY and attempt < step.max_retries:
                    result = result.model_copy(update={"retries": attempt})
                    backoff = min(2 ** (attempt - 1), 30)
                    await asyncio.sleep(backoff)
                    continue

                error_msg = f"{type(exc).__name__}: {exc}"

                if step.on_error == OnError.SKIP:
                    result = result.model_copy(
                        update={
                            "status": StepStatus.SKIPPED,
                            "error": error_msg,
                        }
                    )
                    return result, state, []

                result = result.finish(error=error_msg)
                return result, state, []

    async def _execute_step(
        self,
        step: Step,
        state: StateContext,
    ) -> tuple[Any, LLMUsage | None, list[StepResult]]:
        if step.type == StepType.LLM:
            output, usage = await self._execute_llm(step, state)
            return output, usage, []
        if step.type == StepType.TOOL:
            output = await self._execute_tool(step, state)
            return output, None, []
        if step.type == StepType.CONDITION:
            output = self._execute_condition(step, state)
            return output, None, []
        if step.type == StepType.PARALLEL:
            output, sub_results = await self._execute_parallel(step, state)
            return output, None, sub_results
        raise VMError(f"Unknown step type: {step.type}")

    # ------------------------------------------------------------------
    # LLM step
    # ------------------------------------------------------------------

    async def _execute_llm(
        self,
        step: Step,
        state: StateContext,
    ) -> tuple[str, LLMUsage | None]:
        prompt = self._resolve(step.prompt, state)
        messages: list[dict[str, str]] = []
        if step.system:
            messages.append({"role": "system", "content": self._resolve(step.system, state)})
        messages.append({"role": "user", "content": prompt})

        result = await self._llm.complete(messages)

        if isinstance(result, tuple):
            text, usage_data = result
            usage = LLMUsage(**usage_data) if usage_data else None
        else:
            text = result
            usage = None

        return text, usage

    # ------------------------------------------------------------------
    # Tool step
    # ------------------------------------------------------------------

    async def _execute_tool(self, step: Step, state: StateContext) -> Any:
        if step.tool not in self._tools:
            raise VMError(
                f"Tool '{step.tool}' not registered. Available: {list(self._tools.keys())}"
            )
        fn = self._tools[step.tool]
        resolved_args = {k: self._resolve(v, state) for k, v in step.args.items()}
        if asyncio.iscoroutinefunction(fn):
            return await fn(**resolved_args)
        result = fn(**resolved_args)
        if asyncio.iscoroutine(result):
            return await result
        return result

    # ------------------------------------------------------------------
    # Condition step
    # ------------------------------------------------------------------

    def _execute_condition(self, step: Step, state: StateContext) -> str | None:
        condition = self._resolve(step.condition, state)
        try:
            result = bool(eval(condition, {"__builtins__": {}}, {}))  # noqa: S307
        except Exception as exc:
            raise VMError(f"Condition eval error '{condition}': {exc}") from exc
        return step.then if result else step.otherwise

    # ------------------------------------------------------------------
    # Parallel step
    # ------------------------------------------------------------------

    async def _execute_parallel(
        self,
        step: Step,
        state: StateContext,
    ) -> tuple[dict[str, Any], list[StepResult]]:
        semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(step.max_concurrency) if step.max_concurrency is not None else None
        )

        async def _run_sub(sub: Step) -> tuple[StepResult, Any]:
            sub_result = StepResult(step_id=sub.id, status=StepStatus.RUNNING)
            try:
                if semaphore is not None:
                    async with semaphore:
                        output, usage = await self._dispatch_leaf(sub, state)
                else:
                    output, usage = await self._dispatch_leaf(sub, state)
                sub_result = sub_result.finish(output=output, usage=usage)
                return sub_result, output
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                if step.on_error == OnError.SKIP:
                    sub_result = sub_result.model_copy(
                        update={"status": StepStatus.SKIPPED, "error": error_msg}
                    )
                    return sub_result, None
                sub_result = sub_result.finish(error=error_msg)
                return sub_result, None

        raw = await asyncio.gather(*[_run_sub(sub) for sub in step.parallel_steps])

        sub_results: list[StepResult] = []
        outputs: dict[str, Any] = {}
        failed: list[str] = []

        for sub_step, (sub_result, output) in zip(step.parallel_steps, raw):
            sub_results.append(sub_result)
            if sub_result.status == StepStatus.FAILED:
                failed.append(sub_step.id)
            elif sub_result.status == StepStatus.SUCCESS:
                outputs[sub_step.id] = output
            else:
                outputs[sub_step.id] = None

        if failed and step.on_error != OnError.SKIP:
            raise VMError(f"Parallel step '{step.id}': sub-steps failed: {failed}")

        return outputs, sub_results

    async def _dispatch_leaf(
        self,
        step: Step,
        state: StateContext,
    ) -> tuple[Any, LLMUsage | None]:
        if step.type == StepType.LLM:
            return await self._execute_llm(step, state)
        if step.type == StepType.TOOL:
            output = await self._execute_tool(step, state)
            return output, None
        raise VMError(f"Sub-step '{step.id}': type '{step.type}' not allowed inside parallel")

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------

    @staticmethod
    def _state_fingerprint(state: StateContext) -> int:
        return hash(frozenset((k, str(v)) for k, v in state.step_outputs.items()))

    @staticmethod
    def _state_fingerprint_hex(state: StateContext) -> str:
        canonical = ",".join(f"{k}={v!r}" for k, v in sorted(state.step_outputs.items()))
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Resolver
    # ------------------------------------------------------------------

    def _resolve(self, value: Any, state: StateContext) -> Any:
        if not isinstance(value, str):
            return value

        _MISSING = object()

        def replace(match: re.Match) -> str:
            expr = match.group(1)

            if "." in expr:
                step_id, field = expr.split(".", 1)
                step_out = state.step_outputs.get(step_id, _MISSING)
                if step_out is _MISSING:
                    return match.group(0)
                if field == "output":
                    return str(step_out)
                if isinstance(step_out, dict):
                    val = step_out.get(field, _MISSING)
                    return str(val) if val is not _MISSING else match.group(0)
                return match.group(0)

            val = state.data.get(expr, _MISSING)
            return str(val) if val is not _MISSING else match.group(0)

        return re.sub(r"\$(\w+(?:\.\w+)?)", replace, value)
