"""
nano-vm: deterministic VM for LLM program execution.

v0.6.0 additions:
  - TraceStatus.SUSPENDED + Trace.suspend() — для async webhook resume
  - WebhookEvent — typed input contract для VM.resume_with_program()
  - InMemoryCursorRepository / CursorRepository Protocol
  - InterruptType — typed interrupt signals (BUDGET, TIMEOUT)
  - VaultStepResult / VaultStepError / VaultStepMetadata — vault-layer contracts
  - ResumeError — специализированное исключение для resume failures

v0.7.0 additions (Sprint 1 — Deterministic Execution Architecture):
  - CapabilityRef — replaces raw PII in CanonicalState; secure_hash() + tombstone()
  - PolicySnapshot — immutable rule snapshot per session (frozen); from_config()
  - GovernanceEnvelope — typed wrapper for outgoing MCP data with policy audit trail
  - ProjectionTarget — enum of projection targets (LLM, TRACE, TOOL)
  - AbstractProjectionLayer — base class for all projection implementations
  - DeterministicSanitizer — concrete regex + field-rule sanitiser; no eval()

Quick start (unchanged):
    from nano_vm import ExecutionVM, Program
    from nano_vm.adapters import LiteLLMAdapter

    vm = ExecutionVM(llm=LiteLLMAdapter("groq/llama-3.3-70b-versatile"))
    trace = await vm.run(program, context={"user_input": "..."})

Suspend/resume (v0.6.0):
    trace = await vm.run(program, context={...})
    if trace.status == TraceStatus.SUSPENDED:
        # сохранить trace.trace_id, ждать webhook
        event = WebhookEvent(trace_id=trace.trace_id, payload={"status": "confirmed"})
        trace = await vm.resume_with_program(event, program)

CapabilityRef / projection (v0.7.0):
    from nano_vm import CapabilityRef, PolicySnapshot, DeterministicSanitizer
    from nano_vm import ProjectionTarget

    ref = CapabilityRef(ref_id="vault://secret/42", salt="s0m3s4lt")
    policy = PolicySnapshot.from_config(config, policy_id="pol-001", version="1.0.0")
    sanitizer = DeterministicSanitizer()
    projected = sanitizer.project(state, target=ProjectionTarget.LLM)
"""

from .analyzer import (
    ExecutionReceipt,
    RejectedTransition,
    TraceAnalyzer,
    TraceHealthReport,
)
from .contracts import (
    CapabilityRef,
    GovernanceEnvelope,
    PolicySnapshot,
)
from .models import (
    InterruptType,
    LLMUsage,
    OnError,
    Program,
    StateContext,
    Step,
    StepMetrics,
    StepResult,
    StepStatus,
    StepType,
    Trace,
    TraceStatus,
    VaultStepError,
    VaultStepMetadata,
    VaultStepResult,
)
from .planner import Planner, PlannerError
from .projection import (
    AbstractProjectionLayer,
    DeterministicSanitizer,
    ProjectionTarget,
)
from .validator import (
    IssueKind,
    IssueSeverity,
    ProgramValidator,
    ValidationIssue,
    ValidationReport,
)
from .vm import (
    CursorRepository,
    ExecutionVM,
    InMemoryCursorRepository,
    ResumeError,
    VMError,
    WebhookEvent,
)

__all__ = [
    # VM
    "ExecutionVM",
    "VMError",
    "ResumeError",
    "WebhookEvent",
    "CursorRepository",
    "InMemoryCursorRepository",
    # Planner
    "Planner",
    "PlannerError",
    # Program / Step
    "Program",
    "Step",
    "StepType",
    "StepStatus",
    "OnError",
    # State
    "StateContext",
    # Results
    "StepResult",
    "LLMUsage",
    # Trace
    "Trace",
    "TraceStatus",
    "StepMetrics",
    # v0.6.0 vault primitives
    "InterruptType",
    "VaultStepResult",
    "VaultStepError",
    "VaultStepMetadata",
    # v0.7.0 Sprint 1 — Deterministic Execution Architecture
    "CapabilityRef",
    "PolicySnapshot",
    "GovernanceEnvelope",
    "ProjectionTarget",
    "AbstractProjectionLayer",
    "DeterministicSanitizer",
    # v0.8.5 — ProgramValidator (pre-flight static analysis)
    "ProgramValidator",
    "ValidationReport",
    "ValidationIssue",
    "IssueKind",
    "IssueSeverity",
    # v0.8.5 — TraceAnalyzer / ExecutionReceipt (post-hoc analysis)
    "TraceAnalyzer",
    "TraceHealthReport",
    "ExecutionReceipt",
    "RejectedTransition",
]

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    # Single source of truth: the version hatch-vcs baked into the installed
    # distribution's metadata at build time. Previously hardcoded here as
    # "0.7.0" and never updated across 7 releases (actual: 0.8.8+).
    __version__ = _version("llm-nano-vm")
except _PackageNotFoundError:
    # Running from a source checkout with no installed distribution metadata
    # (e.g. straight `python -c "import nano_vm"` against a git clone).
    __version__ = "0.0.0+unknown"
