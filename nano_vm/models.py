"""
nano_vm.models
==============
Pure Pydantic models. No IO, no LLM calls.
Everything that enters and exits the VM.

v0.6.0 additions (vault-layer primitives):
  - TraceStatus.SUSPENDED  — VM suspended awaiting webhook resume()
  - InterruptType           — typed interrupt signals (BUDGET, TIMEOUT)
  - VaultStepError          — structured error with retryable + compensation_required
  - VaultStepMetadata       — idempotency_key, trace_id, cached (OTel propagation)
  - VaultStepResult         — vault-layer StepResult contract (wraps tool output)
  - Trace.trace_id          — stable UUID, required for suspend/resume correlation
  - Trace.suspended_at      — timestamp set by VM._suspend()
  - Trace.suspended_step_id — step cursor for resume()

v0.7.0 additions (RFC: Deterministic Execution Architecture):
  - CapabilityRef   — replaces raw PII in CanonicalState; supports tombstoning (GDPR)
  - PolicySnapshot  — immutable rule snapshot per session; frozen=True
  - GdprEraseEvent  — typed GDPR erasure event; frozen=True
  - Trace.canonical_snapshot_hash() — Merkle root over state_snapshots
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    # types-PyYAML provides stubs; runtime import is lazy inside from_yaml()
    import yaml as _yaml_types  # noqa: F401

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
# v0.6.0: Interrupt + vault error/metadata types
# ---------------------------------------------------------------------------


class InterruptType(str, Enum):
    """
    Typed interrupt signals emitted by the VM.
    Budget = Interrupt, не control-flow условие (инвариант I7).
    """

    BUDGET = "BUDGET"
    TIMEOUT = "TIMEOUT"


class VaultStepError(BaseModel):
    """
    Structured error contract для vault-layer tool steps.
    retryable: SagaCoordinator использует для решения о retry vs escalate.
    compensation_required: SagaCoordinator использует для обхода в обратном порядке.
    """

    code: str  # machine-readable (e.g. "PAYMENT_DECLINED")
    message: str  # human-readable
    retryable: bool
    compensation_required: bool


class VaultStepMetadata(BaseModel):
    """
    Metadata, встраиваемые в каждый VaultStepResult.
    trace_id: OTel propagation с первого коммита (инвариант I из decisions_log).
    cached: True если результат из _tool_cache (persisted SQLite).
    """

    idempotency_key: str  # "{order_id}:{step_id}:{tool_name}"
    execution_time_ms: int
    tool_version: str
    cached: bool
    trace_id: str  # OTel trace propagation


class VaultStepResult(BaseModel):
    """
    Vault-layer StepResult contract (Section 4.4 spec).
    Используется MCPPolicyLayer и SagaCoordinator.
    Не заменяет существующий StepResult — является его vault-надстройкой.

    status: "SUCCESS" | "FAILED" | "PENDING"
      - PENDING: шаг инициирован (напр. payment), ждёт webhook -> VM.suspend()
      - SUCCESS: шаг завершён, data содержит результат
      - FAILED:  шаг провалился, error содержит VaultStepError
    """

    status: str  # строка для MCP-совместимости; валидируется ниже
    data: dict[str, Any] = Field(default_factory=dict)
    error: VaultStepError | None = None
    metadata: VaultStepMetadata

    @model_validator(mode="after")
    def _validate_status(self) -> VaultStepResult:
        allowed = {"SUCCESS", "FAILED", "PENDING"}
        if self.status not in allowed:
            raise ValueError(
                f"VaultStepResult.status must be one of {allowed}, got '{self.status}'"
            )
        return self

    @property
    def is_pending(self) -> bool:
        return self.status == "PENDING"

    @property
    def is_failed(self) -> bool:
        return self.status == "FAILED"

    @property
    def is_retryable(self) -> bool:
        return self.is_failed and self.error is not None and self.error.retryable

    @property
    def requires_compensation(self) -> bool:
        return self.is_failed and self.error is not None and self.error.compensation_required


# ---------------------------------------------------------------------------
# v0.7.0: CapabilityRef — replaces raw PII in CanonicalState
# ---------------------------------------------------------------------------


class CapabilityRef(BaseModel, frozen=True):
    """
    Замена сырых PII-данных в CanonicalState.

    RFC v0.7.0:
      ref_id       — URI ссылки в Vault (e.g. 'vault://secret/123')
      salt         — соль для salted hashing; генерируется один раз при создании ref
      is_tombstone — True после E_gdpr_erase; ProjectionLayer возвращает [REDACTED_TOMBSTONE]

    Инварианты:
      - frozen=True: иммутабельна после создания.
      - secure_hash() детерминирован для одного (ref_id, salt).
      - Tombstone сохраняет hash-chain: все tombstone-ссылки дают константный хеш.

    Использование:
      ref = CapabilityRef(ref_id="vault://users/42/email", salt="abc123")
      h   = ref.secure_hash()   # sha256("vault://users/42/emailabc123")

      # После GDPR-erasure:
      ref = ref.model_copy(update={"is_tombstone": True})
      ref.secure_hash()  # → "TOMBSTONE"
    """

    ref_id: str  # URI: "vault://secret/123"
    salt: str = Field(default_factory=lambda: uuid.uuid4().hex)
    is_tombstone: bool = False

    def secure_hash(self) -> str:
        """
        Детерминированный хеш ссылки.

        Returns:
          'TOMBSTONE'         — если is_tombstone == True
          sha256(ref_id+salt) — иначе (hex digest)
        """
        if self.is_tombstone:
            return "TOMBSTONE"
        raw = (self.ref_id + self.salt).encode()
        return hashlib.sha256(raw).hexdigest()

    def tombstone(self) -> CapabilityRef:
        """
        Возвращает новый CapabilityRef с is_tombstone=True.
        Используется при обработке E_gdpr_erase.
        """
        return self.model_copy(update={"is_tombstone": True})


# ---------------------------------------------------------------------------
# v0.7.0: PolicySnapshot — immutable rule snapshot per session
# ---------------------------------------------------------------------------


class PolicySnapshot(BaseModel, frozen=True):
    """
    Иммутабельный снимок политики для одной сессии.

    RFC v0.7.0:
      policy_id        — идентификатор политики
      version          — версия политики (semver)
      policy_hash      — sha256 конфига на момент создания снимка
      tool_capabilities — разрешённые capability per tool:
                          {'send_email': ['email.read_raw', 'email.send']}

    Инварианты:
      - frozen=True: не может быть изменён после создания (immutable per session).
      - policy_hash вычисляется детерминированно из (policy_id, version, tool_capabilities).
      - has_capability() — O(1) проверка без side effects.

    Использование (Gateway):
      snapshot = PolicySnapshot(
          policy_id="p-001",
          version="1.0.0",
          tool_capabilities={"send_email": ["email.read_raw"]},
      )
      ok = snapshot.has_capability("send_email", "email.read_raw")  # True
    """

    policy_id: str
    version: str
    policy_hash: str = Field(default="")
    tool_capabilities: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _compute_policy_hash(self) -> PolicySnapshot:
        """
        Вычисляет policy_hash если не задан явно.
        Детерминировано: sorted(tool_capabilities) гарантирует порядок.
        """
        if not self.policy_hash:
            caps_repr = ";".join(
                f"{k}:{','.join(sorted(v))}" for k, v in sorted(self.tool_capabilities.items())
            )
            raw = f"{self.policy_id}:{self.version}:{caps_repr}".encode()
            # Обходим frozen через object.__setattr__
            object.__setattr__(self, "policy_hash", hashlib.sha256(raw).hexdigest())
        return self

    def has_capability(self, tool_name: str, capability: str) -> bool:
        """Проверяет наличие capability для инструмента. Pure function, no IO."""
        return capability in self.tool_capabilities.get(tool_name, [])

    def allowed_tools(self) -> frozenset[str]:
        """Возвращает frozenset имён инструментов с хотя бы одной capability."""
        return frozenset(self.tool_capabilities.keys())


# ---------------------------------------------------------------------------
# v0.7.0: GdprEraseEvent — typed GDPR erasure event
# ---------------------------------------------------------------------------


class GdprEraseEvent(BaseModel, frozen=True):
    """
    Типизированное событие GDPR-стирания.

    RFC v0.7.0:
      event_id       — уникальный идентификатор события (uuid)
      target_ref_ids — tuple ref_id CapabilityRef'ов, подлежащих tombstoning
      issued_at      — UTC-время выпуска события
      issued_by      — идентификатор инициатора (user/service)

    Инварианты:
      - frozen=True: событие иммутабельно после создания.
      - target_ref_ids — tuple (совместимо с frozen).
      - VM.erase(event) обходит state.data и tombstones CapabilityRef с ref_id ∈ target_ref_ids.

    Использование:
      event = GdprEraseEvent(
          target_ref_ids=("vault://users/42/email", "vault://users/42/phone"),
          issued_by="gdpr-service",
      )
      new_state = vm.erase(state, event)
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_ref_ids: tuple[str, ...] = Field(default_factory=tuple)
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    issued_by: str = "unknown"
    reason: str = "gdpr_erasure"

    @model_validator(mode="after")
    def _validate_target_ref_ids(self) -> GdprEraseEvent:
        if not self.target_ref_ids:
            raise ValueError("GdprEraseEvent.target_ref_ids cannot be empty")
        return self


GdprEraseEvent.model_rebuild()


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
    max_steps: int | None = None
    max_stalled_steps: int | None = None
    max_tokens: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Program:
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
    cost_usd: float | None = None

    def __str__(self) -> str:
        cost = f", cost=${self.cost_usd:.6f}" if self.cost_usd is not None else ""
        return f"tokens={self.total_tokens}{cost}"


class StepResult(BaseModel):
    step_id: str
    status: StepStatus
    output: Any = None
    error: str | None = None
    retries: int = 0
    usage: LLMUsage | None = None
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
    SUSPENDED = "suspended"  # v0.6.0: VM suspended, awaiting resume() via webhook


class Trace(BaseModel):
    program_name: str
    status: TraceStatus = TraceStatus.RUNNING

    # v0.6.0: stable ID для suspend/resume correlation и OTel propagation
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    steps: list[StepResult] = Field(default_factory=list)
    final_output: Any = None
    error: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    duration_ms: float | None = None
    state_snapshots: list[tuple[int, str]] = Field(default_factory=list)

    # v0.6.0: suspend cursor
    suspended_step_id: str | None = None
    suspended_at: datetime | None = None

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

    def suspend(self, step_id: str) -> Trace:
        """
        v0.6.0: Переводит Trace в SUSPENDED, фиксирует cursor.
        Вызывается только из ExecutionVM._suspend().
        """
        return self.model_copy(
            update={
                "status": TraceStatus.SUSPENDED,
                "suspended_step_id": step_id,
                "suspended_at": datetime.now(timezone.utc),
            }
        )

    def add_step(self, result: StepResult) -> Trace:
        return self.model_copy(update={"steps": [*self.steps, result]})

    def add_snapshot(self, step_index: int, fp_hex: str) -> Trace:
        entry = (step_index, fp_hex)
        return self.model_copy(update={"state_snapshots": [*self.state_snapshots, entry]})

    def last_output(self) -> Any:
        for result in reversed(self.steps):
            if result.status == StepStatus.SUCCESS:
                return result.output
        return None

    def total_tokens(self) -> int:
        return sum(s.usage.total_tokens for s in self.steps if s.usage)

    def total_cost_usd(self) -> float | None:
        costs = [s.usage.cost_usd for s in self.steps if s.usage and s.usage.cost_usd is not None]
        return round(sum(costs), 8) if costs else None

    def canonical_snapshot_hash(self) -> str:
        """
        Merkle root над state_snapshots.

        Алгоритм:
          1. leaf_i = sha256(f"{idx}:{fp}") для каждого (idx, fp) в state_snapshots
          2. Нечётный последний лист дублируется (стандартный Bitcoin-Merkle)
          3. Рекурсивно попарно хешируем до одного корня
          4. Пустой список → sha256("empty")

        Pure function, no IO, детерминирован.
        """
        snapshots = self.state_snapshots
        if not snapshots:
            return hashlib.sha256(b"empty").hexdigest()

        # Шаг 1: листья как bytes (sha256.digest())
        current_b: list[bytes] = [
            hashlib.sha256(f"{idx}:{fp}".encode()).digest()
            for idx, fp in snapshots
        ]

        # Единственный лист: возвращаем hex напрямую
        if len(current_b) == 1:
            return current_b[0].hex()

        # Шаг 2-3: Merkle reduction — bytes конкатенация на каждом уровне
        while len(current_b) > 1:
            if len(current_b) % 2 == 1:
                current_b.append(current_b[-1])  # дублируем нечётный лист
            next_b: list[bytes] = []
            for i in range(0, len(current_b), 2):
                next_b.append(hashlib.sha256(current_b[i] + current_b[i + 1]).digest())
            current_b = next_b

        return current_b[0].hex()
