from __future__ import annotations

import asyncio
from typing import Any

from nano_vm.ast_engine import eval_condition
from nano_vm.models import (
    CapabilityRef,
    GdprEraseEvent,
    Program,
    StateContext,
    Step,
    StepType,
    Trace,
    TraceStatus,
)


class VMError(Exception):
    pass


class ResumeError(Exception):
    pass


class WebhookEvent:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class CursorRepository:
    pass


class InMemoryCursorRepository:
    pass


class ExecutionVM:
    def __init__(
        self,
        llm: Any = None,
        tools: dict[str, Any] | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools or {}

    async def run(self, program: Program, context: dict[str, Any] | None = None) -> Trace:
        state = StateContext(data=dict(context or {}))
        trace = Trace(program_name=program.name)
        steps_by_id = {s.id: s for s in program.steps}
        step_ids = [s.id for s in program.steps]
        current_id: str | None = step_ids[0] if step_ids else None
        final_output: Any = None

        jumped: bool = False  # True when we arrived via a condition branch
        while current_id and current_id in steps_by_id:
            step = steps_by_id[current_id]
            output, usage, next_id = await self._execute_step(step, state)
            # Store output_key in step_outputs
            key = step.output_key or step.id
            new_outputs = {**state.step_outputs, key: output}
            state = StateContext(data=dict(state.data), step_outputs=new_outputs)
            final_output = output

            if step.type == StepType.CONDITION:
                # Jump to branch; the branch step itself advances sequentially
                current_id = next_id
                jumped = True
            elif jumped:
                # We executed exactly one step after a condition jump — stop.
                current_id = None
            else:
                # Advance sequentially
                idx = step_ids.index(current_id)
                current_id = step_ids[idx + 1] if idx + 1 < len(step_ids) else None

        trace = trace.model_copy(
            update={"status": TraceStatus.SUCCESS, "final_output": final_output}
        )
        return trace

    async def _execute_step(
        self, step: Step, state: StateContext
    ) -> tuple[Any, Any, str | None]:
        if step.type == StepType.LLM:
            return await self._execute_llm(step, state)
        if step.type == StepType.TOOL:
            return await self._execute_tool(step, state)
        if step.type == StepType.CONDITION:
            return await self._execute_condition(step, state)
        raise VMError(f"Unknown step type: {step.type}")

    async def _execute_llm(
        self, step: Step, state: StateContext
    ) -> tuple[Any, Any, str | None]:
        ctx = {**state.data, "__step_outputs__": state.step_outputs}
        prompt = step.prompt
        for k, v in ctx.items():
            if isinstance(v, str):
                prompt = prompt.replace(f"${k}", v)
        messages = [{"role": "user", "content": prompt}]
        output = await self._llm.complete(messages)
        return output, None, None

    async def _execute_tool(
        self, step: Step, state: StateContext
    ) -> tuple[Any, Any, str | None]:
        tool = self._tools.get(step.tool)
        if tool is None:
            raise VMError(f"Tool not registered: {step.tool!r}")
        if asyncio.iscoroutinefunction(tool):
            output = await tool()
        else:
            output = tool()
        return output, None, None

    async def _execute_condition(
        self, step: Step, state: StateContext
    ) -> tuple[Any, Any, str | None]:
        # Flatten step_outputs into ctx so $key resolves both from data and step outputs
        ctx = {**state.step_outputs, **state.data, "__step_outputs__": state.step_outputs}
        try:
            result = eval_condition(step.condition, ctx)
        except Exception:
            result = False
        branch = step.then if result else step.otherwise
        return branch, None, branch

    # ------------------------------------------------------------------
    # erase() — Sprint 3
    # ------------------------------------------------------------------

    def erase(
        self, event: GdprEraseEvent, state: StateContext
    ) -> tuple[StateContext, int]:
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
        new_state = StateContext(
            data=new_data,
            step_outputs=dict(state.step_outputs),
        )
        return new_state, counter[0]
