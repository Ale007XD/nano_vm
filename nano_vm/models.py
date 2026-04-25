"""
nano_vm.models
==============
Чистые Pydantic-модели. Никаких IO, никаких LLM-вызовов.
Всё, что входит в VM и выходит из неё.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StepType(str, Enum):
    LLM = "llm"          # вызов языковой модели
    TOOL = "tool"         # вызов Python-callable
    CONDITION = "condition"  # ветвление по значению из state


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class OnError(str, Enum):
    FAIL = "fail"      # остановить программу (default)
    SKIP = "skip"      # пропустить шаг, продолжить
    RETRY = "retry"    # повторить (см. Step.max_retries)


# ---------------------------------------------------------------------------
# Step — один шаг программы
# ---------------------------------------------------------------------------


class Step(BaseModel):
    """Описание одного шага программы."""

    id: str = Field(..., description="Уникальный идентификатор шага внутри программы")
    type: StepType

    # --- llm-шаг ---
    prompt: str | None = None
    system: str | None = None
    output_key: str | None = None   # куда писать ответ в state

    # --- tool-шаг ---
    tool: str | None = None          # имя инструмента из реестра VM
    args: dict[str, Any] = Field(default_factory=dict)

    # --- condition-шаг ---
    condition: str | None = None     # выражение: "$step_a.output == 'yes'"
    then: str | None = None          # id шага если True
    otherwise: str | None = None     # id шага если False

    # --- управление ошибками ---
    on_error: OnError = OnError.FAIL
    max_retries: int = 1

    @model_validator(mode="after")
    def _validate_by_type(self) -> "Step":
        if self.type == StepType.LLM and not self.prompt:
            raise ValueError(f"Step '{self.id}': llm-шаг требует prompt")
        if self.type == StepType.TOOL and not self.tool:
            raise ValueError(f"Step '{self.id}': tool-шаг требует tool")
        if self.type == StepType.CONDITION and not self.condition:
            raise ValueError(f"Step '{self.id}': condition-шаг требует condition")
        return self


# ---------------------------------------------------------------------------
# Program — список шагов + метаданные
# ---------------------------------------------------------------------------


class Program(BaseModel):
    """Полная программа для исполнения в ExecutionVM."""

    name: str = "unnamed"
    description: str = ""
    version: str = "1.0"
    steps: list[Step] = Field(..., min_length=1)

    @classmethod
    def from_dict(cls, data: dict) -> "Program":
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Program":
        try:
            import yaml  # опциональная зависимость
        except ImportError:
            raise ImportError("pip install pyyaml для загрузки программ из YAML")
        return cls.model_validate(yaml.safe_load(yaml_str))

    def get_step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)


# ---------------------------------------------------------------------------
# StateContext — состояние исполнения (immutable снимок)
# ---------------------------------------------------------------------------


class StateContext(BaseModel, frozen=True):
    """
    Снимок состояния на момент исполнения шага.
    frozen=True — нельзя мутировать, только создавать новый через model_copy.
    """

    data: dict[str, Any] = Field(default_factory=dict)
    step_outputs: dict[str, Any] = Field(default_factory=dict)  # step_id → output

    def with_output(self, step_id: str, output: Any) -> "StateContext":
        """Вернуть новый StateContext с добавленным output шага."""
        return self.model_copy(
            update={
                "step_outputs": {**self.step_outputs, step_id: output}
            }
        )

    def with_data(self, key: str, value: Any) -> "StateContext":
        """Вернуть новый StateContext с добавленным ключом в data."""
        return self.model_copy(
            update={"data": {**self.data, key: value}}
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


# ---------------------------------------------------------------------------
# StepResult — результат одного шага
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    step_id: str
    status: StepStatus
    output: Any = None
    error: str | None = None
    retries: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    duration_ms: float | None = None

    def finish(self, output: Any = None, error: str | None = None) -> "StepResult":
        finished = datetime.now(timezone.utc)
        duration = (finished - self.started_at).total_seconds() * 1000
        status = StepStatus.FAILED if error else StepStatus.SUCCESS
        return self.model_copy(update={
            "status": status,
            "output": output,
            "error": error,
            "finished_at": finished,
            "duration_ms": round(duration, 2),
        })


# ---------------------------------------------------------------------------
# Trace — полная история прогона
# ---------------------------------------------------------------------------


class TraceStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Trace(BaseModel):
    program_name: str
    status: TraceStatus = TraceStatus.RUNNING
    steps: list[StepResult] = Field(default_factory=list)
    final_output: Any = None
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    duration_ms: float | None = None

    def finish(
        self,
        status: TraceStatus,
        final_output: Any = None,
        error: str | None = None,
    ) -> "Trace":
        finished = datetime.now(timezone.utc)
        duration = (finished - self.started_at).total_seconds() * 1000
        return self.model_copy(update={
            "status": status,
            "final_output": final_output,
            "error": error,
            "finished_at": finished,
            "duration_ms": round(duration, 2),
        })

    def add_step(self, result: StepResult) -> "Trace":
        return self.model_copy(update={"steps": [*self.steps, result]})

    def last_output(self) -> Any:
        """Вернуть output последнего успешного шага."""
        for result in reversed(self.steps):
            if result.status == StepStatus.SUCCESS:
                return result.output
        return None
