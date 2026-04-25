"""
nano_vm.vm
==========
ExecutionVM — детерминированное исполнение Program.

Ключевые свойства:
- VM не знает про провайдеров, получает LLMAdapter через __init__
- При одинаковом Program + StateContext + детерминированном адаптере → результат воспроизводим
- Шаги исполняются последовательно; condition-шаги управляют переходами
- on_error per-step: fail | skip | retry
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable

from .adapters.base import LLMAdapter
from .models import (
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
    """Ошибка исполнения программы."""


class ExecutionVM:
    """
    Детерминированная виртуальная машина для исполнения Program.

    Args:
        llm:   адаптер языковой модели (любой объект с методом async complete())
        tools: реестр инструментов {имя: async callable}

    Пример:
        vm = ExecutionVM(
            llm=LiteLLMAdapter("groq/llama-3.3-70b-versatile"),
            tools={"search": my_search_fn},
        )
        trace = await vm.run(program, context={"user_input": "..."})
    """

    def __init__(
        self,
        llm: LLMAdapter,
        tools: dict[str, Callable] | None = None,
    ) -> None:
        self._llm = llm
        self._tools: dict[str, Callable] = tools or {}

    def register_tool(self, name: str, fn: Callable) -> None:
        """Зарегистрировать инструмент после создания VM."""
        self._tools[name] = fn

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    async def run(
        self,
        program: Program,
        context: dict[str, Any] | None = None,
    ) -> Trace:
        """
        Исполнить программу.

        Args:
            program: Program с шагами
            context: начальные данные (попадают в StateContext.data)

        Returns:
            Trace — полная история прогона со статусом и результатами шагов.
        """
        state = StateContext(data=context or {})
        trace = Trace(program_name=program.name)

        # Строим индекс шагов для быстрого перехода
        step_index = {s.id: i for i, s in enumerate(program.steps)}
        steps = program.steps
        current_idx = 0

        while current_idx < len(steps):
            step = steps[current_idx]
            result, state = await self._run_step(step, state)
            trace = trace.add_step(result)

            if result.status == StepStatus.FAILED:
                trace = trace.finish(
                    TraceStatus.FAILED,
                    error=f"Шаг '{step.id}' завершился с ошибкой: {result.error}",
                )
                return trace

            # Condition-шаг управляет переходом
            if step.type == StepType.CONDITION and result.status == StepStatus.SUCCESS:
                next_id = result.output  # then/otherwise id
                if next_id and next_id in step_index:
                    current_idx = step_index[next_id]
                    # После condition-перехода исполняем только целевой шаг и выходим
                    target_step = steps[current_idx]
                    target_result, state = await self._run_step(target_step, state)
                    trace = trace.add_step(target_result)
                    if target_result.status == StepStatus.FAILED:
                        trace = trace.finish(
                            TraceStatus.FAILED,
                            error=f"Шаг '{target_step.id}' завершился с ошибкой: {target_result.error}",
                        )
                    else:
                        trace = trace.finish(TraceStatus.SUCCESS, final_output=trace.last_output())
                    return trace

            current_idx += 1

        trace = trace.finish(TraceStatus.SUCCESS, final_output=trace.last_output())
        return trace

    # ------------------------------------------------------------------
    # Исполнение одного шага
    # ------------------------------------------------------------------

    async def _run_step(
        self,
        step: Step,
        state: StateContext,
    ) -> tuple[StepResult, StateContext]:
        """Запустить шаг с учётом on_error и max_retries."""
        result = StepResult(step_id=step.id, status=StepStatus.RUNNING)
        attempt = 0

        while True:
            try:
                output = await self._execute_step(step, state)
                result = result.finish(output=output)

                # Записать output в state если указан output_key
                if step.output_key:
                    state = state.with_data(step.output_key, output)
                state = state.with_output(step.id, output)

                return result, state

            except Exception as exc:
                attempt += 1
                if step.on_error == OnError.RETRY and attempt < step.max_retries:
                    await asyncio.sleep(0.5 * attempt)  # экспоненциальный backoff
                    continue

                error_msg = f"{type(exc).__name__}: {exc}"

                if step.on_error == OnError.SKIP:
                    result = result.model_copy(update={
                        "status": StepStatus.SKIPPED,
                        "error": error_msg,
                    })
                    return result, state

                # on_error == FAIL (default)
                result = result.finish(error=error_msg)
                return result, state

    async def _execute_step(self, step: Step, state: StateContext) -> Any:
        """Диспетчер по типу шага."""
        if step.type == StepType.LLM:
            return await self._execute_llm(step, state)
        if step.type == StepType.TOOL:
            return await self._execute_tool(step, state)
        if step.type == StepType.CONDITION:
            return self._execute_condition(step, state)
        raise VMError(f"Неизвестный тип шага: {step.type}")

    # ------------------------------------------------------------------
    # LLM-шаг
    # ------------------------------------------------------------------

    async def _execute_llm(self, step: Step, state: StateContext) -> str:
        prompt = self._resolve(step.prompt, state)
        messages: list[dict[str, str]] = []
        if step.system:
            messages.append({"role": "system", "content": self._resolve(step.system, state)})
        messages.append({"role": "user", "content": prompt})
        return await self._llm.complete(messages)

    # ------------------------------------------------------------------
    # Tool-шаг
    # ------------------------------------------------------------------

    async def _execute_tool(self, step: Step, state: StateContext) -> Any:
        if step.tool not in self._tools:
            raise VMError(
                f"Инструмент '{step.tool}' не зарегистрирован. "
                f"Доступные: {list(self._tools.keys())}"
            )
        fn = self._tools[step.tool]
        resolved_args = {
            k: self._resolve(v, state) for k, v in step.args.items()
        }
        if asyncio.iscoroutinefunction(fn):
            return await fn(**resolved_args)
        return fn(**resolved_args)

    # ------------------------------------------------------------------
    # Condition-шаг
    # ------------------------------------------------------------------

    def _execute_condition(self, step: Step, state: StateContext) -> str | None:
        """
        Вычислить условие, вернуть id следующего шага (then или otherwise).
        Условие — строка вида "$step_id.output == 'yes'" или "$key == value".
        """
        condition = self._resolve(step.condition, state)

        # Безопасное вычисление булева выражения
        try:
            result = bool(eval(condition, {"__builtins__": {}}, {}))  # noqa: S307
        except Exception as exc:
            raise VMError(f"Ошибка вычисления условия '{condition}': {exc}") from exc

        return step.then if result else step.otherwise

    # ------------------------------------------------------------------
    # Resolver — подстановка $переменных из state
    # ------------------------------------------------------------------

    def _resolve(self, value: Any, state: StateContext) -> Any:
        """
        Подставить значения из state в строку.

        Синтаксис:
            $key               → state.data[key]
            $step_id.output    → state.step_outputs[step_id]
        """
        if not isinstance(value, str):
            return value

        def replace(match: re.Match) -> str:
            expr = match.group(1)  # "key" или "step_id.output"

            if "." in expr:
                step_id, field = expr.split(".", 1)
                step_out = state.step_outputs.get(step_id)
                if step_out is None:
                    return match.group(0)  # оставить как есть
                if field == "output":
                    return str(step_out)
                # Для dict-output: $step_id.field
                if isinstance(step_out, dict):
                    return str(step_out.get(field, match.group(0)))
                return match.group(0)

            # Простая переменная из data
            val = state.data.get(expr)
            return str(val) if val is not None else match.group(0)

        return re.sub(r"\$(\w+(?:\.\w+)?)", replace, value)
