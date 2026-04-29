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

With Planner (P5):
    from nano_vm import Planner, ExecutionVM
    from nano_vm.adapters import LiteLLMAdapter

    adapter = LiteLLMAdapter("openai/gpt-4o-mini")
    planner = Planner(llm=adapter)

    program = await planner.generate(
        "Fetch the latest news, summarize each article, then classify by topic",
        available_tools=["fetch_news", "save_result"],
        context_keys=["topic"],
    )
    trace = await ExecutionVM(llm=adapter).run(program, context={"topic": "AI"})
    print(trace.final_output)

Testing with deterministic adapter:
    from nano_vm.adapters import MockLLMAdapter

    vm = ExecutionVM(llm=MockLLMAdapter("SAFE"))
    trace = await vm.run(program, context={"user_input": "test"})
    assert trace.status == TraceStatus.SUCCESS
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
from .planner import Planner, PlannerError
from .vm import ExecutionVM, VMError

__all__ = [
    "ExecutionVM",
    "VMError",
    "Planner",
    "PlannerError",
    "Program",
    "Step",
    "StepType",
    "StepStatus",
    "OnError",
    "StateContext",
    "StepResult",
    "LLMUsage",
    "Trace",
    "TraceStatus",
]

__version__ = "0.5.0"
