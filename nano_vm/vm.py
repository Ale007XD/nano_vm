"""
nano_vm.vm
==========
ExecutionVM: deterministic execution of a Program.

Key properties:
- VM has no knowledge of providers; receives LLMAdapter via __init__
- Same Program + StateContext + deterministic adapter -> reproducible result
- Steps execute sequentially; condition steps control branching
- Per-step on_error: fail | skip | retry
- LLM steps populate StepResult.usage (tokens + cost) when available
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable

from .adapters.base import LLMAdapter
from .models import (
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


class ExecutionVM:
    """
    Deterministic virtual machine for executing Programs.

    Args:
        llm:   LLM adapter (any object with async complete() method)
        tools: tool registry {name: async callable}

    Example:
        vm = ExecutionVM(
            llm=LiteLLMAdapter("groq/llama-3.3-70b-versatile"),
            tools={"search": my_search_fn},
        )
        trace = await vm.run(program, context={"user_input": "..."})
        print(trace.final_output)
        print(f"Total tokens: {trace.total_tokens()}")
        print(f"Total cost: ${trace.total_cost_usd():.6f}")
    """

    def __init__(
        self,
        llm: LLMAdapter,
        tools: dict[str, Callable] | None = None,
    ) -> None:
        self._llm = llm
        self._tools: dict[str, Callable] = tools or {}

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

        Args:
            program: Program with steps
            context: initial data (stored in StateContext.data)

        Returns:
            Trace: full execution history with status and per-step results.
        """
        state = StateContext(data=context or {})
        trace = Trace(program_name=program.name)
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
                    error=f"Step '{step.id}' failed: {result.error}",
                )
                return trace

            # Condition step: jump to then/otherwise and finish
            if step.type == StepType.CONDITION and result.status == StepStatus.SUCCESS:
                next_id = result.output
                if next_id and next_id in step_index:
                    target_step = steps[step_index[next_id]]
                    target_result, state = await self._run_step(target_step, state)
                    trace = trace.add_step(target_result)
                    if target_result.status == StepStatus.FAILED:
                        trace = trace.finish(
                            TraceStatus.FAILED,
                            error=f"Step '{target_step.id}' failed: {target_result.error}",
                        )
                    else:
                        trace = trace.finish(TraceStatus.SUCCESS, final_output=trace.last_output())
                    return trace

            current_idx += 1

        trace = trace.finish(TraceStatus.SUCCESS, final_output=trace.last_output())
        return trace

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _run_step(
        self,
        step: Step,
        state: StateContext,
    ) -> tuple[StepResult, StateContext]:
        result = StepResult(step_id=step.id, status=StepStatus.RUNNING)
        attempt = 0

        while True:
            try:
                output, usage = await self._execute_step(step, state)
                result = result.finish(output=output, usage=usage)

                if step.output_key:
                    state = state.with_data(step.output_key, output)
                state = state.with_output(step.id, output)

                return result, state

            except Exception as exc:
                attempt += 1
                if step.on_error == OnError.RETRY and attempt < step.max_retries:
                    await asyncio.sleep(0.5 * attempt)
                    continue

                error_msg = f"{type(exc).__name__}: {exc}"

                if step.on_error == OnError.SKIP:
                    result = result.model_copy(update={
                        "status": StepStatus.SKIPPED,
                        "error": error_msg,
                    })
                    return result, state

                result = result.finish(error=error_msg)
                return result, state

    async def _execute_step(
        self,
        step: Step,
        state: StateContext,
    ) -> tuple[Any, LLMUsage | None]:
        """Dispatch by step type. Returns (output, usage)."""
        if step.type == StepType.LLM:
            return await self._execute_llm(step, state)
        if step.type == StepType.TOOL:
            output = await self._execute_tool(step, state)
            return output, None
        if step.type == StepType.CONDITION:
            output = self._execute_condition(step, state)
            return output, None
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

        # Pass raw call to adapter; usage extracted if adapter returns it
        result = await self._llm.complete(messages)

        # LiteLLMAdapter can optionally return (text, usage_dict) tuple
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
                f"Tool '{step.tool}' not registered. "
                f"Available: {list(self._tools.keys())}"
            )
        fn = self._tools[step.tool]
        resolved_args = {k: self._resolve(v, state) for k, v in step.args.items()}
        if asyncio.iscoroutinefunction(fn):
            return await fn(**resolved_args)
        return fn(**resolved_args)

    # ------------------------------------------------------------------
    # Condition step
    # ------------------------------------------------------------------

    def _execute_condition(self, step: Step, state: StateContext) -> str | None:
        """
        Evaluate condition string, return id of next step (then or otherwise).

        Condition syntax examples:
            "'yes' in '$step_id.output'.lower()"
            "$score.output >= '0.8'"

        Note: $variables are resolved to strings before eval.
        Use .lower() in your condition for case-insensitive matching.
        """
        condition = self._resolve(step.condition, state)
        try:
            result = bool(eval(condition, {"__builtins__": {}}, {}))  # noqa: S307
        except Exception as exc:
            raise VMError(f"Condition eval error '{condition}': {exc}") from exc
        return step.then if result else step.otherwise

    # ------------------------------------------------------------------
    # Resolver: substitute $variables from state
    # ------------------------------------------------------------------

    def _resolve(self, value: Any, state: StateContext) -> Any:
        """
        Substitute $variables in a string from state.

        Syntax:
            $key            -> state.data[key]
            $step_id.output -> state.step_outputs[step_id]
        """
        if not isinstance(value, str):
            return value

        def replace(match: re.Match) -> str:
            expr = match.group(1)

            if "." in expr:
                step_id, field = expr.split(".", 1)
                step_out = state.step_outputs.get(step_id)
                if step_out is None:
                    return match.group(0)
                if field == "output":
                    return str(step_out)
                if isinstance(step_out, dict):
                    return str(step_out.get(field, match.group(0)))
                return match.group(0)

            val = state.data.get(expr)
            return str(val) if val is not None else match.group(0)

        return re.sub(r"\$(\w+(?:\.\w+)?)", replace, value)
