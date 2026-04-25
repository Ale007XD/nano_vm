"""
nano-vm: deterministic VM for LLM program execution.

Quick start:
    from nano_vm import ExecutionVM, Program
    from nano_vm.adapters import LiteLLMAdapter

    vm = ExecutionVM(
        llm=LiteLLMAdapter("groq/llama-3.3-70b-versatile"),
        tools={"my_tool": my_async_fn},
    )
    program = Program.from_yaml(open("program.yaml").read())
    trace = await vm.run(program, context={"user_input": "..."})
    print(trace.final_output)
    print(f"Tokens used: {trace.total_tokens()}")
"""

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
from .vm import ExecutionVM, VMError
from .planner import Planner, PlannerError

__all__ = [
    "ExecutionVM", "VMError",
    "Planner", "PlannerError",
    "Program", "Step", "StepType", "StepStatus", "OnError",
    "StateContext", "StepResult", "LLMUsage", "Trace", "TraceStatus",
]

__version__ = "0.1.0"
