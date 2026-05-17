nano-vm

<p align="center">
  <a href="https://github.com/Ale007XD/nano_vm/actions">
    <img src="https://github.com/Ale007XD/nano_vm/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/llm-nano-vm/">
    <img src="https://img.shields.io/pypi/v/llm-nano-vm" alt="PyPI">
  </a>
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p><p align="center">
  <strong>Deterministic execution runtime for stateful workflows.</strong><br>
  Replayable. Observable. Enforcement-first.<br>
  LLM support is optional.
</p><p align="center">
  <em>Temporal-lite for deterministic AI and business process execution.</em>
</p>
---

What nano-vm Is

nano-vm is a deterministic execution runtime built around finite-state-machine semantics.

It orchestrates:

financial workflows

webhook-driven async processes

approval pipelines

event-driven automation

retry-safe orchestration

LLM pipelines

governance-bound execution graphs


The runtime — not the model, not the tool, not the prompt — controls state transitions.

Core invariant:

\delta(S,E) \rightarrow S'

Where:

 — current execution state

 — validated event

 — next deterministic state



---

Why nano-vm Exists

Most workflow engines optimize scheduling.

Most AI frameworks optimize prompting.

nano-vm optimizes execution correctness.

The system exists to guarantee:

deterministic transitions

replayable traces

exactly-once execution invariants

resumable async workflows

explicit governance boundaries

runtime-level enforcement


The FSM runtime is the source of truth.

LLMs are optional.


---

Mental Model

events / webhooks / tools / LLMs
                ↓
          ExecutionVM
                ↓
          deterministic FSM
                ↓
          replayable trace

Or formally:

nondeterminism ∈ signal generation
determinism    ∈ runtime execution


---

Core Execution Pipeline

Canonical runtime model:

E  = Signal(input)      → raw event
E' = Validator(E)       → validated event
A  = FSM(S, E')         → allowed transitions
a* = Policy(A, C)       → selected transition
S' = δ(S, a*)           → next state

Layer	Role	Deterministic

Signal	LLM / webhook / API / user input	❌
Validator	schema + policy validation	✅
FSM	transition authority	✅
Policy	transition selection	✅
Tool executor	side effects	enforced



---

Using nano-vm Without LLMs

LLMs are not required.

nano-vm can operate as a pure deterministic workflow engine.


---

Example — Financial Transaction Flow

from nano_vm import ExecutionVM, Program

async def reserve_funds(**kwargs):
    return {"reserved": True}

async def capture_payment(**kwargs):
    return {"captured": True}

async def send_receipt(**kwargs):
    return {"receipt_sent": True}

program = Program.from_dict({
    "name": "payment_flow",
    "steps": [
        {
            "id": "reserve",
            "type": "tool",
            "tool": "reserve_funds"
        },
        {
            "id": "capture",
            "type": "tool",
            "tool": "capture_payment"
        },
        {
            "id": "receipt",
            "type": "tool",
            "tool": "send_receipt"
        }
    ]
})

vm = ExecutionVM(
    tools={
        "reserve_funds": reserve_funds,
        "capture_payment": capture_payment,
        "send_receipt": send_receipt,
    }
)

trace = await vm.run(program)

print(trace.status)

No LLM was used.

The runtime still guarantees:

deterministic ordering

replayable execution

trace visibility

transition enforcement

exactly-once semantics



---

Suspend / Resume — Async Business Processes

nano-vm supports long-running workflows via deterministic suspension.

Example:

async def wait_bank_transfer(**kwargs):
    return "PENDING"

FSM transition:

RUNNING → SUSPENDED → RUNNING → SUCCESS

This enables:

payment settlement workflows

courier confirmation flows

approval systems

human-in-the-loop execution

webhook-driven orchestration



---

Example — Webhook-Driven Payment Settlement

from nano_vm.vm import ExecutionVM

async def initiate_payment(**kwargs):
    await register_webhook(kwargs["order_id"])
    return "PENDING"

program = Program.from_dict({
    "name": "payment_pipeline",
    "steps": [
        {
            "id": "payment",
            "type": "tool",
            "tool": "initiate_payment"
        },
        {
            "id": "finalize",
            "type": "tool",
            "tool": "finalize_order"
        }
    ]
})

vm = ExecutionVM(
    tools={
        "initiate_payment": initiate_payment,
        "finalize_order": finalize_order,
    }
)

trace = await vm.run(
    program,
    context={"order_id": "123"}
)

assert trace.status.name == "SUSPENDED"

trace = await vm.resume_with_program(
    program=program,
    trace_id=trace.trace_id,
    webhook_event={
        "type": "payment.confirmed",
        "order_id": "123"
    }
)


---

FSM Transition Model

ExecutionVM is a deterministic finite state machine.

Canonical transition function

\delta(S,E) \rightarrow S'


---

Runtime States

Current state	Event	Next state

RUNNING	tool success	RUNNING
RUNNING	tool returns "PENDING"	SUSPENDED
RUNNING	failure	FAILED
RUNNING	budget exceeded	BUDGET_EXCEEDED
RUNNING	no more steps	SUCCESS
SUSPENDED	resume event	RUNNING


Terminal states are absorbing.


---

Deterministic Guarantees

Guarantee	nano-vm

Step ordering enforced	✅
Replayable traces	✅
Deterministic transitions	✅
Exactly-once invariants	✅
Observable execution	✅
Suspend/resume	✅
Runtime-enforced policies	✅



---

Install

pip install llm-nano-vm

Optional LiteLLM support:

pip install llm-nano-vm[litellm]


---

Quick Start — LLM Pipeline

LLM support exists as an optional signal-generation layer.

from nano_vm import ExecutionVM, Program
from nano_vm.adapters import LiteLLMAdapter

program = Program.from_dict({
    "name": "refund_pipeline",
    "steps": [
        {
            "id": "classify",
            "type": "llm",
            "prompt": "Classify refund request: $user_input",
            "output_key": "decision"
        },
        {
            "id": "guardrail",
            "type": "condition",
            "condition": "'refund' in '$decision'",
            "then": "approve",
            "otherwise": "reject"
        },
        {
            "id": "approve",
            "type": "tool",
            "tool": "issue_refund"
        },
        {
            "id": "reject",
            "type": "tool",
            "tool": "send_rejection"
        }
    ]
})

vm = ExecutionVM(
    llm=LiteLLMAdapter("openai/gpt-4o-mini"),
    tools={
        "issue_refund": issue_refund,
        "send_rejection": send_rejection,
    }
)

trace = await vm.run(
    program,
    context={
        "user_input": "I was charged twice"
    }
)

The LLM cannot:

skip steps

reorder execution

terminate workflows

bypass conditions


The runtime controls execution.


---

Program DSL

Four core step types:

Type	Purpose

tool	execute Python function
llm	call model
condition	deterministic branching
parallel	concurrent execution



---

Example — Deterministic Branching

{
    "id": "guardrail",
    "type": "condition",
    "condition": "'yes' in '$decision'",
    "then": "approve",
    "otherwise": "reject"
}

Branch logic belongs to the runtime.

Not to the model.


---

Parallel Execution

{
    "id": "fetch",
    "type": "parallel",
    "parallel_steps": [
        {
            "id": "weather",
            "type": "tool",
            "tool": "get_weather"
        },
        {
            "id": "news",
            "type": "tool",
            "tool": "get_news"
        }
    ]
}

Execution remains deterministic even under concurrency.


---

Observability

Every execution produces a replayable trace.

trace.trace_id
trace.status
trace.steps
trace.total_tokens()
trace.total_cost_usd()

Per-step visibility:

for step in trace.steps:
    print(
        step.step_id,
        step.status,
        step.duration_ms
    )


---

Testing

Deterministic systems are testable systems.

from nano_vm.adapters import MockLLMAdapter

vm = ExecutionVM(
    llm=MockLLMAdapter("SAFE")
)

Same input → same execution graph.


---

State Determinism

nano-vm guarantees state determinism.

It does not guarantee semantic determinism.

Property	Guaranteed

State transitions	✅
Step ordering	✅
Trace structure	✅
Exact LLM wording	❌



---

Planner (Optional)

Planner converts intent into validated programs.

from nano_vm import Planner

planner = Planner(llm=adapter)

program = await planner.generate(
    "Fetch latest AI news and summarize",
    available_tools=["fetch", "summarize"]
)

Planner output remains probabilistic.

Execution remains deterministic.


---

Architecture Principles

1. FSM as Source of Truth

The runtime owns transitions.

Not the model.


---

2. Stateless Execution Core

Execution state lives in StateContext.

ExecutionVM.step() remains deterministic and replayable.


---

3. Enforcement over Prompting

Guardrails belong in runtime logic.

Not inside prompts.


---

4. Replayability

Every execution is reproducible at the state-transition layer.


---

Comparison

	LangChain	CrewAI	Temporal	nano-vm

LLM-native	✅	✅	❌	✅
Deterministic FSM	❌	❌	✅	✅
Replayable traces	partial	minimal	✅	✅
Suspend/resume	partial	partial	✅	✅
Runtime guardrails	❌	❌	partial	✅
Lightweight	❌	❌	❌	✅
Business workflows	partial	❌	✅	✅
AI workflows	✅	✅	partial	✅



---

Performance

The runtime overhead is near-zero.

The bottleneck is typically external IO:

LLM APIs

databases

webhooks

network-bound tools


Stress suite results:

Benchmark	Result

Replay violations	0
Double execution violations	0
Invalid transitions	0
Out-of-order corruption	0


Total stress operations:

1,000,000+ deterministic transitions verified


---

When to Use

Use nano-vm when:

workflow correctness matters

replayability matters

auditability matters

async orchestration matters

deterministic transitions matter

AI must operate under governance


Examples:

fintech

payment systems

compliance automation

async business workflows

deterministic AI pipelines

approval systems

webhook orchestration



---

When NOT to Use

Do not use nano-vm when:

workflows are fully unknown

unrestricted autonomous agents are required

open-ended reasoning is the primary goal

execution guarantees are unnecessary



---

Roadmap

Runtime

[x] Deterministic FSM runtime

[x] Suspend/resume

[x] Parallel execution

[x] Budget interrupts

[x] Replayable traces

[x] AST-safe conditions

[x] Exactly-once guarantees


AI Layer

[x] LiteLLM adapter

[x] Planner

[x] Mock adapters

[ ] Provider pools

[ ] Dynamic policy routing


Infrastructure

[x] MCP integration

[x] SQLite persistence

[ ] Distributed runtime

[ ] REST gateway

[ ] Execution dashboard



---

Philosophy

nano-vm is not an AI agent framework.

It is:

\text{Deterministic Execution Runtime for Stochastic Systems}

The runtime governs execution.

Probabilistic systems operate inside deterministic boundaries.


---

Contact

Author: @ale007xd

GitHub:

[nano-vm GitHub Repository](https://github.com/Ale007XD/nano_vm?utm_source=chatgpt.com)

PyPI:

[llm-nano-vm on PyPI](https://pypi.org/project/llm-nano-vm/?utm_source=chatgpt.com)


---

License

MIT License.
