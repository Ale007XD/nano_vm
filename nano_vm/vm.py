"""
nano_vm.vm
==========
ExecutionVM: deterministic execution of a Program.

Key properties:
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
from typing import Any

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
        steps_executed = 0
        last_fingerprint: int | None = None
        stalled_count = 0

        while current_idx < len(steps):
            if program.max_steps is not None and steps_executed >= program.max_steps:
                trace = trace.finish(
                    TraceStatus.BUDGET_EXCEEDED,
                    error=f"max_steps={program.max_steps} exceeded after {steps_executed} step(s)",
                )
                return trace

            step = steps[current_idx]
            result, state, sub_results = await self._run_step(step, state)
            steps_executed += 1

            # Fingerprint-based no-op detection (P1)
            current_fp = self._state_fingerprint(state)
            if last_fingerprint is not None and current_fp == last_fingerprint:
                stalled_count += 1
            else:
                stalled_count = 0
            last_fingerprint = current_fp

            if program.max_stalled_steps is not None and stalled_count >= program.max_stalled_steps:
                trace = trace.finish(
                    TraceStatus.STALLED,
                    error=(
                        f"max_stalled_steps={program.max_stalled_steps} exceeded: "
                        f"{stalled_count} consecutive no-op step(s)"
                    ),
                )
                return trace

            # State snapshot (P2): sha256 fingerprint per step
            trace = trace.add_snapshot(
                steps_executed - 1,
                self._state_fingerprint_hex(state),
            )

            # Add sub-step results to trace before the parent step
            for sub_result in sub_results:
                trace = trace.add_step(sub_result)
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

                if not next_id:
                    trace = trace.finish(
                        TraceStatus.FAILED,
                        error=f"Step '{step.id}': condition produced no branch target",
                    )
                    return trace

                if next_id not in step_index:
                    msg = f"Step '{step.id}': condition target '{next_id}' not found in program"
                    trace = trace.finish(TraceStatus.FAILED, error=msg)
                    return trace

                target_step = steps[step_index[next_id]]
                steps_executed += 1
                target_result, state, target_sub = await self._run_step(target_step, state)
                for sub_result in target_sub:
                    trace = trace.add_step(sub_result)
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
    ) -> tuple[StepResult, StateContext, list[StepResult]]:
        """
        Execute a single step.

        Returns:
            (result, state, sub_results)
            sub_results: list of StepResult for parallel sub-steps (empty otherwise)
        """
        result = StepResult(step_id=step.id, status=StepStatus.RUNNING)
        attempt = 0

        while True:
            try:
                output, usage, sub_results = await self._execute_step(step, state)
                result = result.finish(output=output, usage=usage)

                if step.output_key:
                    state = state.with_data(step.output_key, output)
                state = state.with_output(step.id, output)

                # For parallel: also store each sub-step output individually
                if step.type == StepType.PARALLEL and isinstance(output, dict):
                    for sub_id, sub_out in output.items():
                        state = state.with_output(sub_id, sub_out)

                return result, state, sub_results

            except Exception as exc:
                attempt += 1
                if step.on_error == OnError.RETRY and attempt < step.max_retries:
                    result = result.model_copy(update={"retries": attempt})
                    # exponential backoff: 1s, 2s, 4s, … capped at 30s
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
        """
        Dispatch by step type.

        Returns:
            (output, usage, sub_results)
            sub_results: StepResult list for parallel sub-steps; empty for all others
        """
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
    # Parallel step
    # ------------------------------------------------------------------

    async def _execute_parallel(
        self,
        step: Step,
        state: StateContext,
    ) -> tuple[dict[str, Any], list[StepResult]]:
        """
        Execute parallel_steps concurrently via asyncio.gather.

        on_error=FAIL (default): any sub-step failure raises, aborting the parallel step.
        on_error=SKIP: failed sub-steps are recorded with SKIPPED status; others proceed.

        Returns:
            outputs: dict[sub_step_id -> output]  (only successful/skipped-with-None)
            sub_results: list[StepResult] for each sub-step (for trace)
        """

        # max_concurrency=None → no cap; otherwise limit via Semaphore
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
                # SKIPPED → явный None (контракт: не absent key, не exception)
                outputs[sub_step.id] = None

        if failed and step.on_error != OnError.SKIP:
            raise VMError(f"Parallel step '{step.id}': sub-steps failed: {failed}")

        return outputs, sub_results

    async def _dispatch_leaf(
        self,
        step: Step,
        state: StateContext,
    ) -> tuple[Any, LLMUsage | None]:
        """Execute llm or tool sub-step. Condition/parallel not allowed here."""
        if step.type == StepType.LLM:
            return await self._execute_llm(step, state)
        if step.type == StepType.TOOL:
            output = await self._execute_tool(step, state)
            return output, None
        raise VMError(f"Sub-step '{step.id}': type '{step.type}' not allowed inside parallel")

    # ------------------------------------------------------------------
    # Fingerprint: deterministic no-op detection
    # ------------------------------------------------------------------

    @staticmethod
    def _state_fingerprint(state: StateContext) -> int:
        """
        Hash of current step_outputs snapshot.

        Converts values to str for hashability (covers dicts from parallel steps).
        Returns hash of frozenset of (key, str(value)) pairs from step_outputs.
        Empty state returns a consistent (platform-defined) value.
        """
        return hash(frozenset((k, str(v)) for k, v in state.step_outputs.items()))

    @staticmethod
    def _state_fingerprint_hex(state: StateContext) -> str:
        """
        SHA-256 hex digest of step_outputs — stable across processes and restarts.

        Used for state_snapshots serialisation (P2).
        _state_fingerprint (hash) is kept for in-process no-op detection (P1).
        """
        canonical = ",".join(f"{k}={v!r}" for k, v in sorted(state.step_outputs.items()))
        return hashlib.sha256(canonical.encode()).hexdigest()

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

        _MISSING = object()

        def replace(match: re.Match) -> str:
            expr = match.group(1)

            if "." in expr:
                step_id, field = expr.split(".", 1)
                # Use sentinel to distinguish missing key from None value
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
