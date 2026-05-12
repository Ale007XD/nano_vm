from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Re-export from contracts so tests can do:
#   from nano_vm.models import CapabilityRef, PolicySnapshot


class GdprEraseEvent(BaseModel):
    """System event triggering GDPR erasure of CapabilityRef values."""

    target_ref_ids: tuple[str, ...]
    reason: str = "gdpr_erasure"
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    issued_by: str | None = None

    model_config = {"frozen": True}

    @model_validator(mode="before")
    @classmethod
    def _validate_not_empty(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not values.get("target_ref_ids"):
            raise ValueError("target_ref_ids cannot be empty")
        return values


class InterruptType(str, Enum):
    BUDGET = "BUDGET"
    TIMEOUT = "TIMEOUT"


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OnError(str, Enum):
    FAIL = "fail"
    SKIP = "skip"
    RETRY = "retry"


class StepType(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    CONDITION = "condition"
    PARALLEL = "parallel"


class StepStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class TraceStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SUSPENDED = "SUSPENDED"
    BUDGET_EXCEEDED = "budget_exceeded"
    STALLED = "stalled"


class Step(BaseModel):
    id: str
    type: StepType
    prompt: str = ""
    output_key: str = ""
    tool: str = ""
    condition: str = ""
    then: str = ""
    otherwise: str = ""


class Program(BaseModel):
    name: str = ""
    steps: list[Step] = []

    @classmethod
    def from_dict(cls, data: dict) -> Program:
        steps = [Step(**s) for s in data.get("steps", [])]
        return cls(name=data.get("name", ""), steps=steps)


class StateContext(BaseModel):
    data: dict[str, Any] = Field(default_factory=dict)
    step_outputs: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    step_id: str = ""
    status: StepStatus = StepStatus.SUCCESS
    output: Any = None


class Trace(BaseModel):
    status: TraceStatus = TraceStatus.SUCCESS
    program_name: str = ""
    steps: list[Any] = Field(default_factory=list)
    state_snapshots: list[tuple[int, str]] = Field(default_factory=list)
    final_output: Any = None
    error: str | None = None
    _token_accumulator: int = 0

    def add_snapshot(self, step_index: int, fingerprint: str) -> Trace:
        new_snapshots = list(self.state_snapshots) + [(step_index, fingerprint)]
        return self.model_copy(update={"state_snapshots": new_snapshots})

    def canonical_snapshot_hash(self) -> str:
        snapshots = list(self.state_snapshots)
        if not snapshots:
            return hashlib.sha256(b"empty").hexdigest()
        leaves = [hashlib.sha256(f"{idx}:{fp}".encode()).digest() for idx, fp in snapshots]
        while len(leaves) > 1:
            if len(leaves) % 2 == 1:
                leaves.append(leaves[-1])
            leaves = [
                hashlib.sha256(leaves[i] + leaves[i + 1]).digest() for i in range(0, len(leaves), 2)
            ]
        return leaves[0].hex()


class VaultStepError(BaseModel):
    pass


class VaultStepMetadata(BaseModel):
    pass


class VaultStepResult(BaseModel):
    pass
