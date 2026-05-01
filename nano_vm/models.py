"""
nano_vm.models
==============
Pure Pydantic models. No IO, no LLM calls.
Everything that enters and exits the VM.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StepType(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    CONDITION = "condition"
    PARALLEL = "parallel"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class OnError(str, Enum):
    FAIL = "fail"
    SKIP = "skip"
    RETRY = "retry"


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class Step(BaseModel):
    """One step in a program."""

    id: str
    type: StepType

    # llm step
    prompt: str | None = None
    system: str | None = None
    output_key: str | None = None

    # tool step
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)

    # condition step
    condition: str | None = None
    then: str | None = None
    otherwise: str | None = None

    # parallel step
    parallel_steps: list[Step] = Field(default_factory=list)

    # parallel step options
    max_concurrency: int | None = None  # None = no cap (all sub-steps at once)

    # error handling
    on_error: OnError = OnError.FAIL
    max_retries: int = 3  # attempts total (1 initial + 2 retries)

    @model_validator(mode="after")
    def _validate_by_type(self) -> Step:
        if self.type == StepType.LLM and not self.prompt:
            raise ValueError(f"Step '{self.id}': llm step requires prompt")
        if self.type == StepType.TOOL and not self.tool:
            raise ValueError(f"Step '{self.id}': tool step requires tool")
        if self.type == StepType.CONDITION and not self.condition:
            raise ValueError(f"Step '{self.id}': condition step requires condition")
        if self.type == StepType.CONDITION and not self.then and not self.otherwise:
            raise ValueError(
                f"Step '{self.id}': condition step requires at least one of: then, otherwise"
            )
        if self.type == StepType.PARALLEL and not self.parallel_steps:
            raise ValueError(
                f"Step '{self.id}': parallel step requires at least one parallel_steps entry"
            )
        if self.type == StepType.PARALLEL:
            for sub in self.parallel_steps:
                if sub.type in (StepType.CONDITION, StepType.PARALLEL):
                    raise ValueError(
                        f"Step '{self.id}': parallel sub-step '{sub.id}' "
                        f"cannot be of type '{sub.type}' (only llm/tool allowed)"
                    )
        return self


# ---------------------------------------------------------------------------
# Program
# ---------------------------------------------------------------------------


class Program(BaseModel):
    """Full program for execution in ExecutionVM."""

    name: str = "unnamed"
    description: str = ""
    version: str = "1.0"
    steps: list[Step] = Field(..., min_length=1)
    max_steps: int | None = None  # None = no cap; BUDGET_EXCEEDED when exceeded
    max_stalled_steps: int | None = None  # None = disabled; STALLED after N consecutive no-ops
    max_tokens: int | None = None  # None = no cap; BUDGET_EXCEEDED when total_tokens >= limit

    @classmethod
    def from_dict(cls, data: dict) -> Program:
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Program:
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml to load programs from YAML")
        return cls.model_validate(yaml.safe_load(yaml_str))

    def get_step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)


# ---------------------------------------------------------------------------
# StateContext
# ---------------------------------------------------------------------------


class StateContext(BaseModel, frozen=True):
    """
    Immutable snapshot of execution state.
    frozen=True: no mutation, only create new via model_copy.
    """

    data: dict[str, Any] = Field(default_factory=dict)
    step_outputs: dict[str, Any] = Field(default_factory=dict)

    def with_output(self, step_id: str, output: Any) -> StateContext:
        return self.model_copy(update={"step_outputs": {**self.step_outputs, step_id: output}})

    def with_data(self, key: str, value: Any) -> StateContext:
        return self.model_copy(update={"data": {**self.data, key: value}})

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


# ---------------------------------------------------------------------------
# LLMUsage + StepResult
# ---------------------------------------------------------------------------


class LLMUsage(BaseModel):
    """LLM token usage and cost for a single step."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None  # None if provider did not return cost

    def __str__(self) -> str:
        cost = f", cost=${self.cost_usd:.6f}" if self.cost_usd is not None else ""
        return f"tokens={self.total_tokens}{cost}"


class StepResult(BaseModel):
    step_id: str
    status: StepStatus
    output: Any = None
    error: str | None = None
    retries: int = 0
    usage: LLMUsage | None = None  # filled only for llm steps
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    duration_ms: float | None = None

    def finish(
        self,
        output: Any = None,
        error: str | None = None,
        usage: LLMUsage | None = None,
    ) -> StepResult:
        finished = datetime.now(timezone.utc)
        duration = (finished - self.started_at).total_seconds() * 1000
        status = StepStatus.FAILED if error else StepStatus.SUCCESS
        return self.model_copy(
            update={
                "status": status,
                "output": output,
                "error": error,
                "usage": usage,
                "finished_at": finished,
                "duration_ms": round(duration, 2),
            }
        )


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


class TraceStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    STALLED = "stalled"


class Trace(BaseModel):
    program_name: str
    status: TraceStatus = TraceStatus.RUNNING
    steps: list[StepResult] = Field(default_factory=list)
    final_output: Any = None
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    duration_ms: float | None = None
    state_snapshots: list[tuple[int, str]] = Field(default_factory=list)  # (step_index, sha256_hex)
    # Internal O(1) token counter — not part of public schema, not serialised
    _token_accumulator: int = PrivateAttr(default=0)

    def finish(
        self,
        status: TraceStatus,
        final_output: Any = None,
        error: str | None = None,
    ) -> Trace:
        finished = datetime.now(timezone.utc)
        duration = (finished - self.started_at).total_seconds() * 1000
        return self.model_copy(
            update={
                "status": status,
                "final_output": final_output,
                "error": error,
                "finished_at": finished,
                "duration_ms": round(duration, 2),
            }
        )

    def add_step(self, result: StepResult) -> Trace:
        new_trace = self.model_copy(update={"steps": [*self.steps, result]})
        new_trace._token_accumulator = (
            self._token_accumulator + (result.usage.total_tokens if result.usage else 0)
        )
        return new_trace

    def add_snapshot(self, step_index: int, fp_hex: str) -> Trace:
        """Record a state fingerprint snapshot for the given step_index."""
        entry = (step_index, fp_hex)
        return self.model_copy(update={"state_snapshots": [*self.state_snapshots, entry]})

    def last_output(self) -> Any:
        for result in reversed(self.steps):
            if result.status == StepStatus.SUCCESS:
                return result.output
        return None

    def total_tokens(self) -> int:
        return self._token_accumulator

    def total_cost_usd(self) -> float | None:
        costs = [s.usage.cost_usd for s in self.steps if s.usage and s.usage.cost_usd is not None]
        return round(sum(costs), 8) if costs else None
