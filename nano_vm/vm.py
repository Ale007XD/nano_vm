"""
nano_vm.vm
==========
ExecutionVM: deterministic execution of a Program.

v0.6.0: suspend/resume, BudgetInterrupt, CursorRepository, WebhookEvent
v0.7.0: ASTEngine в _execute_condition (eval() удалён), erase()
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

from .adapters.base import LLMAdapter
from .ast_engine import eval_condition
from .contracts import CapabilityRef
from .models import (
    GdprEraseEvent,
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
# WebhookEvent
# ---------------------------------------------------------------------------


class WebhookEvent:
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
# CursorRepository
# ---------------------------------------------------------------------------


class CursorRepository(Protocol):
    async def save(
        self, trace_id: str, step_id: str, state: StateContext, trace: Trace
    ) -> None: ...

    async def load(self, trace_id: str) -> tuple[str, StateContext, Trace] | None: ...

    async def delete(self, trace_id: str) -> None: ...


class InMemoryCursorRepository:
    def __init__(self) -> None:
        self._store: dict[str, tuple[str, StateContext, Trace]] = {}

    async def save(self, trace_id: str, step_id: str, state: StateContext, trace: Trace) -> None:
        self._store[trace_id] = (step_id, state, trace)

    async def load(self, trace_id: str) -> tuple[str, StateContext, Trace] | None:
        return self._store.get(trace_id)

    async def delete(self, trace_id: str) -> None:
        self._store.pop(trace_id, None)


# ---------------------------------------------------------------------------
# ExecutionVM
# ---------------------------------------------------------------------------


class ExecutionVM:
    def __init__(
        self,
        llm: LLMAdapter,
        tools: dict[str, Callable[..., Any]] | None = None,
        cursor_repository: CursorRepository | None = None,
    ) -> None:
        self._llm = llm
        self._tools: dict[str, Callable[..., Any]] = tools or {}
        self._cursor_repo: CursorRepository = cursor_repository or InMemoryCursorRepository()

    def register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        program: Program,
        context: dict[str, Any] | None = None,
    ) -> Trace:
        state = StateContext(data=context or {})
        trace = Trace(program_name=program.name)
        return await self._execute_loop(program, state, trace, start_step_id=None)

    async def resume_with_program(
        self,
        webhook_event: WebhookEvent,
        program: Program,
    ) -> Trace:
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
        state = state.with_data("__webhook__", webhook_event.payload)
        state = state.with_data("__webhook_source__", webhook_event.source)
        await self._cursor_repo.delete(webhook_event.trace_id)
        return await self._execute_loop(
            program=program,
            state=state,
            trace=trace,
            start_step_id=suspended_step_id,
            resume_after=True,
        )

    # ------------------------------------------------------------------
    # erase() — Sprint 3: GDPR tombstoning
    # ------------------------------------------------------------------

    def erase(self, event: GdprEraseEvent, state: StateContext) -> tuple[StateContext, int]:
        target_ids = set(event.target_ref_ids)
        counter = [0]

        def _erase_value(value: Any) -> Any:
            if isinstance(value, CapabilityRef):
                if value.ref_id in target_ids:
                    counter[0] += 1
                    return value.tombstone()
                return value
            if isinstance(value, dict):
                return {k: _erase_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_erase_value(item) for item in value]
            return value

        new_data = {k: _erase_value(v) for k, v in state.data.items()}
        new_outputs = {k: _erase_value(v) for k, v in state.step_outputs.items()}
        new_state = StateContext(data=new_data, step_outputs=new_outputs)
        return new_state, counter[0]

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
        step_index = {s.id: i for i, s in enumerate(program.steps)}
        steps = program.steps

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
                current_idx += 1

        steps_executed = len(trace.steps)
        last_fingerprint: int | None = None
        stalled_count = 0

        while current_idx < len(steps):
            # Budget guards
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

            # Stalled detection
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

            trace = trace.add_snapshot(steps_executed - 1, self._state_fingerprint_hex(state))

            for sub_result in sub_results:
                trace = trace.add_step(sub_result)
            trace = trace.add_step(result)

            if result.status == StepStatus.PENDING:
                trace = await self._suspend(step, state, trace)
                return trace

            if result.status == StepStatus.FAILED:
                return trace.finish(
                    TraceStatus.FAILED,
                    error=f"Step '{step.id}' failed: {result.error}",
                )

            # Explicit halt: non-condition terminal step ends this path.
            if step.is_terminal:
                return trace.finish(TraceStatus.SUCCESS, final_output=trace.last_output())

            # Condition step: jump to branch target.
            #
            # Branch semantics (v0.7.4):
            #   1. Execute the branch target step inline.
            #   2. If the target is a condition, recurse into its sub-branch.
            #   3. If the target has is_terminal=True (explicit halt marker),
            #      return SUCCESS immediately.
            #   4. Otherwise resume the main flow from target_idx + 1, which
            #      supports "inline" branches where the branch target is the
            #      next sequential step (e.g. amount_check -> create_payment).
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
                target_idx = step_index[next_id]
                target_step = steps[target_idx]

                # Execute the branch target inline.
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

                # Target is itself a condition — recurse into its sub-branch.
                if target_step.type == StepType.CONDITION:
                    return await self._execute_loop(
                        program=program,
                        state=state,
                        trace=trace,
                        start_step_id=target_result.output,
                    )

                # Branch target executed. Two cases:
                #
                # 1. target_step.next_step is set: the branch is "inline" —
                #    continue execution from the named step (allows condition
                #    branches to rejoin the main flow, e.g. amount_check →
                #    create_payment.next_step="poll_payment" → poll_payment).
                #
                # 2. Otherwise: the branch is terminal — return SUCCESS here.
                #    This is the default (v0.7.3-compatible) semantics and what
                #    all condition branches that jump to leaf steps must use.
                next_step_id: str | None = getattr(target_step, "next_step", None)
                if next_step_id:
                    if next_step_id not in step_index:
                        return trace.finish(
                            TraceStatus.FAILED,
                            error=(
                                f"Step '{target_step.id}': next_step "
                                f"'{next_step_id}' not found in program"
                            ),
                        )
                    current_idx = step_index[next_step_id]
                    continue

                # Default: terminal branch.
                return trace.finish(TraceStatus.SUCCESS, final_output=trace.last_output())

            current_idx += 1

        return trace.finish(TraceStatus.SUCCESS, final_output=trace.last_output())

    # ------------------------------------------------------------------
    # suspend / interrupt
    # ------------------------------------------------------------------

    async def _suspend(self, step: Step, state: StateContext, trace: Trace) -> Trace:
        trace = trace.suspend(step.id)
        await self._cursor_repo.save(
            trace_id=trace.trace_id,
            step_id=step.id,
            state=state,
            trace=trace,
        )
        return trace

    async def _emit_interrupt(self, interrupt_type: InterruptType, trace: Trace) -> None:
        pass  # hook point for vault-layer escalation

    # ------------------------------------------------------------------
    # Step execution
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
                        update={"status": StepStatus.SKIPPED, "error": error_msg}
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
            llm_out, llm_usage = await self._execute_llm(step, state)
            return llm_out, llm_usage, []
        if step.type == StepType.TOOL:
            tool_out = await self._execute_tool(step, state)
            return tool_out, None, []
        if step.type == StepType.CONDITION:
            cond_out: str | None = self._execute_condition(step, state)
            return cond_out, None, []
        if step.type == StepType.PARALLEL:
            par_out, sub_results = await self._execute_parallel(step, state)
            return par_out, None, sub_results
        raise VMError(f"Unknown step type: {step.type}")

    # ------------------------------------------------------------------
    # LLM step
    # ------------------------------------------------------------------

    async def _execute_llm(self, step: Step, state: StateContext) -> tuple[str, LLMUsage | None]:
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
    # Condition step — ASTEngine (no eval())
    # ------------------------------------------------------------------

    def _execute_condition(self, step: Step, state: StateContext) -> str | None:
        condition = step.condition or ""
        # Wrap each step output as {"output": <value>} so that dotted-path
        # $step_id.output resolves for scalar outputs, and
        # $step_id.output.field resolves for dict outputs.
        # state.data is merged last so output_key aliases remain accessible
        # as flat names (e.g. $validation for output_key="validation").
        wrapped_outputs: dict[str, Any] = {
            k: {"output": v} for k, v in state.step_outputs.items()
        }
        ctx: dict[str, Any] = {**wrapped_outputs, **state.data}
        try:
            result = bool(eval_condition(condition, ctx))
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

    async def _dispatch_leaf(self, step: Step, state: StateContext) -> tuple[Any, LLMUsage | None]:
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

    def _resolve(self, value: Any, state: StateContext) -> Any:  # noqa: C901
        if not isinstance(value, str):
            return value

        _MISSING = object()

        def _lookup(expr: str) -> Any:
            """Resolve dotted expression; return _MISSING if not found."""
            parts = expr.split(".")
            root = parts[0]
            if root in state.step_outputs:
                node: Any = state.step_outputs[root]
                for seg in parts[1:]:
                    if seg == "output":
                        # transparent — node IS the raw output already
                        continue
                    if isinstance(node, dict):
                        node = node.get(seg, _MISSING)
                        if node is _MISSING:
                            return _MISSING
                    else:
                        return _MISSING
                return node
            if len(parts) == 1:
                return state.data.get(root, _MISSING)
            return _MISSING

        # Fast path: entire value is a single $var — return typed value unchanged.
        # Preserves int/float/dict/list types for tool args (e.g. amount: int).
        single = re.fullmatch(r"\$(\w+(?:\.\w+)*)", value)
        if single:
            resolved = _lookup(single.group(1))
            return resolved if resolved is not _MISSING else value

        # Interpolation path: $var embedded in a larger string — stringify.
        def replace(match: re.Match[str]) -> str:
            resolved = _lookup(match.group(1))
            return str(resolved) if resolved is not _MISSING else str(match.group(0))

        return re.sub(r"\$(\w+(?:\.\w+)*)", replace, value)
