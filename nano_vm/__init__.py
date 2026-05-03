"""
nano-vm: deterministic VM for LLM program execution.

v0.6.0 additions:
  - TraceStatus.SUSPENDED + Trace.suspend() — для async webhook resume
  - WebhookEvent — typed input contract для VM.resume_with_program()
  - InMemoryCursorRepository / CursorRepository Protocol
  - InterruptType — typed interrupt signals (BUDGET, TIMEOUT)
  - VaultStepResult / VaultStepError / VaultStepMetadata — vault-layer contracts
  - ResumeError — специализированное исключение для resume failures

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
"""

from .models import (
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
    VaultStepError,
    VaultStepMetadata,
    VaultStepResult,
)
from .planner import Planner, PlannerError
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
    # v0.6.0 vault primitives
    "InterruptType",
    "VaultStepResult",
    "VaultStepError",
    "VaultStepMetadata",
]

__version__ = "0.6.0"
