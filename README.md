<p align="center">
  <a href="https://github.com/Ale007XD/nano_vm/actions">
    <img src="https://github.com/Ale007XD/nano_vm/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/llm-nano-vm/">
    <img src="https://img.shields.io/pypi/v/llm-nano-vm" alt="PyPI">
  </a>
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

<p align="center">
  <strong>Deterministic execution runtime for stateful workflows.</strong><br>
  Replayable. Observable. Enforcement-first.<br>
  LLM support is optional.
</p>

<p align="center">
  <em>Temporal-lite for deterministic AI and business process execution.</em>
</p>

---

## What nano-vm Is

nano-vm is a deterministic execution runtime built around finite-state-machine semantics.

It orchestrates financial workflows, webhook-driven async processes, approval pipelines,
event-driven automation, retry-safe orchestration, LLM pipelines, and
governance-bound execution graphs.

The runtime — not the model, not the tool, not the prompt — controls state transitions.

Core invariant:

```
δ(S, E) → S'
```

Where S is current execution state, E is a validated event, S' is the next deterministic state.

---

## Why nano-vm Exists

Most workflow engines optimize scheduling. Most AI frameworks optimize prompting.
nano-vm optimizes execution correctness.

The system guarantees: deterministic transitions, replayable traces, exactly-once execution
invariants, resumable async workflows, explicit governance boundaries, and runtime-level
enforcement.

The FSM runtime is the source of truth. LLMs are optional.

---

## Mental Model

```
events / webhooks / tools / LLMs
              ↓
        ExecutionVM
              ↓
        deterministic FSM
              ↓
        replayable trace
```

Formally:

```
nondeterminism ∈ signal generation
determinism    ∈ runtime execution
```

---

## Core Execution Pipeline

```
E  = Signal(input)      → raw event
E' = Validator(E)       → validated event
A  = FSM(S, E')         → allowed transitions
a* = Policy(A, C)       → selected transition
S' = δ(S, a*)           → next state
```

| Layer | Role | Deterministic |
| :--- | :--- | :---: |
| Signal | LLM / webhook / API / user input | ❌ |
| Validator | schema + policy validation | ✅ |
| FSM | transition authority | ✅ |
| Policy | transition selection | ✅ |
| Tool executor | side effects | enforced |

---

## Using nano-vm Without LLMs

LLMs are not required. nano-vm can operate as a pure deterministic workflow engine.

```python
from nano_vm import ExecutionVM, Program

program = Program.from_dict({
    "name": "payment_flow",
    "steps": [
        {"id": "reserve",  "type": "tool", "tool": "reserve_funds"},
        {"id": "capture",  "type": "tool", "tool": "capture_payment"},
        {"id": "receipt",  "type": "tool", "tool": "send_receipt"},
    ]
})

vm = ExecutionVM(tools={
    "reserve_funds":   reserve_funds,
    "capture_payment": capture_payment,
    "send_receipt":    send_receipt,
})

trace = await vm.run(program)
print(trace.status)  # SUCCESS
```

No LLM. The runtime still guarantees: deterministic ordering, replayable execution,
trace visibility, transition enforcement, exactly-once semantics.

---

## Suspend / Resume — Async Business Processes

Return the sentinel `"PENDING"` from any tool to suspend execution:

```python
async def initiate_payment(**kwargs) -> str:
    await register_webhook(kwargs["order_id"])
    return "PENDING"   # FSM → SUSPENDED, cursor persisted
```

FSM transition: `RUNNING → SUSPENDED → RUNNING → SUCCESS`

This enables: payment settlement, courier confirmation, approval systems,
human-in-the-loop, webhook orchestration.

```python
from nano_vm.vm import ExecutionVM

vm = ExecutionVM(
    tools={"initiate_payment": initiate_payment, "finalize_order": finalize_order},
)

trace = await vm.run(program, context={"order_id": "123"})
assert trace.status.name == "SUSPENDED"

trace = await vm.resume_with_program(
    program=program,
    trace_id=trace.trace_id,
    webhook_event={"type": "payment.confirmed", "order_id": "123"},
)
assert trace.status.name == "SUCCESS"
```

> **Note:** `"PENDING"` is a reserved FSM sentinel. Do not return it as a domain status.
> Use `"REQUIRES_ACTION"`, `"AWAITING_3DS"`, or any other string for domain-specific states.

---

## FSM Transition Model

`ExecutionVM` is a deterministic finite state machine.

| Current state | Event | Next state |
| :--- | :--- | :--- |
| `RUNNING` | tool success | `RUNNING` |
| `RUNNING` | tool returns `"PENDING"` | `SUSPENDED` |
| `RUNNING` | tool error (`on_error=fail`) | `FAILED` |
| `RUNNING` | tool error (`on_error=skip`) | `RUNNING` (output=`None`) |
| `RUNNING` | condition branch taken | `RUNNING` (jump to `then`/`otherwise`) |
| `RUNNING` | `max_steps` / `max_tokens` exceeded | `BUDGET_EXCEEDED` |
| `RUNNING` | `max_stalled_steps` exceeded | `STALLED` |
| `RUNNING` | no more steps | `SUCCESS` |
| `SUSPENDED` | `resume_with_program()` called | `RUNNING` (from cursor) |
| terminal | — | absorbing (no further transitions) |

Terminal states: `SUCCESS`, `FAILED`, `BUDGET_EXCEEDED`, `STALLED`.

---

## Install

```bash
pip install llm-nano-vm
pip install llm-nano-vm[litellm]   # for LLM provider support
```

---

## Quick Start — LLM Pipeline (guardrail that never skips)

```python
from nano_vm import ExecutionVM, Program
from nano_vm.adapters import LiteLLMAdapter

program = Program.from_dict({
    "name": "customer_refund",
    "steps": [
        {
            "id": "analyze",
            "type": "llm",
            "prompt": "Is this a valid refund request? Reply 'yes' or 'no'.\nRequest: $user_input",
            "output_key": "decision",
        },
        {
            "id": "guardrail",
            "type": "condition",
            "condition": "'yes' in '$decision'",
            "then": "process_refund",
            "otherwise": "reject",
        },
        {"id": "process_refund", "type": "tool", "tool": "issue_refund",    "is_terminal": True},
        {"id": "reject",         "type": "tool", "tool": "send_rejection",  "is_terminal": True},
    ],
})

vm = ExecutionVM(
    llm=LiteLLMAdapter("openai/gpt-4o-mini"),
    tools={"issue_refund": ..., "send_rejection": ...},
)

trace = await vm.run(program, context={"user_input": "I was charged twice"})
print(trace.status)           # SUCCESS
print(trace.total_cost_usd()) # e.g. 0.000034
```

The `guardrail` step cannot be skipped, reordered, or overridden by the model.

---

## Program DSL

Four step types:

| Type | Purpose |
| :--- | :--- |
| `llm` | call the model; result stored in `output_key` |
| `tool` | call a Python function; return `"PENDING"` to suspend |
| `condition` | branch on an expression; `then` / `otherwise` |
| `parallel` | run independent sub-steps concurrently via `asyncio.gather` |

**Step fields (v0.7.4+):**

| Field | Default | Description |
| :--- | :--- | :--- |
| `on_error` | `fail` | `fail` · `skip` · `retry` |
| `max_retries` | `3` | total attempts; backoff: 1s, 2s, 4s… cap 30s |
| `max_concurrency` | `None` | parallel blocks only |
| `is_terminal` | `False` | return `SUCCESS` after this step (leaf nodes) |
| `next_step` | `None` | jump to named step instead of returning `SUCCESS` |

**Program budget options (v0.4.0+):**

| Option | Default | Description |
| :--- | :--- | :--- |
| `max_steps` | `None` | `BUDGET_EXCEEDED` if exceeded |
| `max_stalled_steps` | `None` | `STALLED` after N consecutive no-op fingerprints |
| `max_tokens` | `None` | `BUDGET_EXCEEDED` when total tokens exceed limit |

### Variable interpolation

| Syntax | Resolves to |
| :--- | :--- |
| `$key` | value from initial context (typed — int/dict/list preserved) |
| `$step_id.output` | output of a previous step |
| `$step_id.output.field` | field within a step's dict output |

### Condition expressions — Security

> **⚠ ASTEngine replaces `eval()`.** Conditions are parsed into a validated JSON AST
> and evaluated by a pure, sandboxed interpreter. No Python builtins are accessible.

**Supported operators:** `==`, `!=`, `>`, `<`, `in`, `not in`, `and`, `or`, `not`,
`contains`, dotted-path `$var.field`.

**Not supported:** method calls (`.lower()`, `.strip()`, etc.), arithmetic, parentheses
grouping. Using an unsupported form raises `ASTEvalError` at parse time with an explicit
message (v0.7.5+).

**Rules for safe use:**

- Condition logic must be authored by you, not generated from untrusted input at runtime.
- LLM output may appear as a *value being tested* (`'yes' in '$decision'`), never as the
  condition expression itself.
- If you need case-insensitive matching, control the LLM output format via the prompt
  (`Reply ONLY with: yes or no`) rather than calling `.lower()` in the condition.

```python
# ❌ WRONG — method call raises ASTEvalError (v0.7.5+)
{"condition": "'yes' in '$decision'.lower()"}

# ✅ CORRECT — pure value comparison
{"condition": "'yes' in '$decision'"}

# ❌ WRONG — user input becomes the expression
{"condition": "$user_input", "then": "pay"}

# ✅ CORRECT — you author the expression; LLM output is only the tested value
{"condition": "'yes' in '$decision'", "then": "process_refund"}
```

---

## Branch Semantics (v0.7.4)

Condition branch targets are terminal by default. Use `next_step` for inline continuation:

```python
# Branch target is terminal — FSM returns SUCCESS after notify_success
{"id": "notify_success", "type": "tool", "tool": "send_email", "is_terminal": True}

# Branch target continues to poll_payment
{"id": "create_payment", "type": "tool", "tool": "create_payment_intent",
 "next_step": "poll_payment"}
```

Terminal leaf steps (`notify_*`, `reject_*`, `alert_*`) must be placed **before** any
inline chain steps in the flat steps array and marked `is_terminal: true`.

---

## Parallel Execution

```python
{
    "id": "fetch",
    "type": "parallel",
    "max_concurrency": 5,
    "on_error": "skip",
    "parallel_steps": [
        {"id": "weather", "type": "tool", "tool": "get_weather", "args": {"city": "$city"}},
        {"id": "news",    "type": "tool", "tool": "get_news",    "args": {"topic": "$topic"}},
    ],
}
```

Wall-clock time = slowest sub-step. Partial result: failed sub-step with `on_error: skip`
produces `None`, not an exception.

---

## Observability

```python
trace.trace_id              # UUID4 — stable for OTel propagation
trace.status                # TraceStatus.SUCCESS | FAILED | SUSPENDED | BUDGET_EXCEEDED | STALLED
trace.final_output
trace.total_tokens()        # O(1) incremental accumulator
trace.total_cost_usd()      # requires LiteLLMAdapter
trace.state_snapshots       # list[(step_index, sha256_hex)]

for step in trace.steps:
    print(step.step_id, step.status, step.duration_ms, step.usage)
```

---

## Budget Interrupts

```python
from nano_vm.vm import ExecutionVM, InterruptType

class InstrumentedVM(ExecutionVM):
    async def _emit_interrupt(self, interrupt_type: InterruptType) -> None:
        await notify_operator(f"interrupt: {interrupt_type.value}")
```

Budget exhaustion (`BudgetInterrupt`) fires before the next step executes.
The LLM cannot observe or influence it.

---

## Testing — Deterministic by Design

```python
from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.adapters import MockLLMAdapter

vm = ExecutionVM(llm=MockLLMAdapter("yes"))   # always returns "yes"

# Per-call sequence
vm = ExecutionVM(llm=MockLLMAdapter(["SAFE", "yes"]))

# Per-prompt substring mapping
vm = ExecutionVM(llm=MockLLMAdapter({
    "Classify": "SAFE",
    "eligible": "yes",
    "__default__": "ok",
}))

trace = await vm.run(program, context={"user_input": "refund"})
assert trace.status == TraceStatus.SUCCESS
```

Same input → same step sequence. No API key required.

### State Determinism vs Semantic Determinism

nano-vm guarantees **State Determinism** — step execution order, no skipping, reproducible
trace structure — regardless of LLM output. It does not guarantee **Semantic Determinism**
(LLM text may differ across runs even at `temperature=0.0`). Use `MockLLMAdapter` for both.

---

## Planner (Optional)

```python
from nano_vm import Planner

planner = Planner(llm=adapter, max_retries=2, temperature=0.0)
program = await planner.generate(
    "Fetch latest AI news, summarize, classify by topic",
    available_tools=["fetch_rss", "summarize", "classify"],
    context_keys=["user_id"],
)
trace = await vm.run(program)
```

Exactly 1 LLM call → validated `Program`. Planner output is probabilistic; execution
remains deterministic. Review Planner-generated programs before deploying to production.

---

## MCP Integration

nano-vm pairs with [nano-vm-mcp](https://github.com/Ale007XD/nano-vm-mcp) — an MCP server
that exposes `run_program`, `get_trace`, `list_programs`, `get_program`, `delete_program`
over stdio or SSE transport with bearer auth and SQLite WAL persistence.

### Architecture

```
MCP Client
  → nano-vm-mcp (Gateway)
      → GovernedRunProgramHandler   ← PolicySnapshot, CapabilityRef
          → llm-nano-vm (Kernel)    ← deterministic FSM, ASTEngine, ProjectionLayer
      → GovernanceEnvelope store    ← SQLite WAL, append-only audit log
```

### GovernanceEnvelope

Each successful execution step produces a `GovernanceEnvelope` (frozen Pydantic model)
stored in the `governance_envelopes` table:

| Field | Description |
| :--- | :--- |
| `execution_id` | Session / trace identifier |
| `step_id` | Step index within the execution |
| `policy_hash` | SHA-256 of the active `PolicySnapshot` |
| `canonical_snapshot_hash` | Merkle/delta hash of `CanonicalState` |
| `payload` | Projected (sanitized) step output |

### CapabilityRef and GDPR Tombstoning

Sensitive values are stored as `CapabilityRef` tokens (`vault://secret/<id>`).
On a GDPR erasure event, the ref is tombstoned (`is_tombstone=True`). All subsequent
projections return `[REDACTED_TOMBSTONE]`, preserving the hash chain.

---

## Custom Adapter

```python
class MyAdapter:
    async def complete(self, messages: list[dict], **kwargs) -> str:
        ...  # call any LLM API
```

Built-in via `[litellm]` extra:

```python
LiteLLMAdapter("groq/llama-3.3-70b-versatile")
LiteLLMAdapter("openrouter/llama-3.3-70b-instruct:free")
LiteLLMAdapter("ollama/llama3")
LiteLLMAdapter("openai/gpt-4o-mini")
```

---

## Performance

The VM introduces near-zero overhead. The bottleneck is the LLM API or external I/O.

### v0.7.5 stress suite (179/179 tests · 0 violations)

| Suite | Result |
| :--- | :--- |
| MoMo PoC v4 | 9/9 PASS |
| Stripe PoC v1 | 9/9 PASS |
| FSM invariant suite (v0.6.0) | 13/13 · 1,020,000 ops · 0 violations |
| Integration suite (v0.7.3) | 10/10 · 1,096,500 ops · 0 violations |
| 10k stress (v0.7.0) | 14,327 graphs/sec · 0.70 s/run |

### Integration benchmark detail (v0.7.3)

Environment: QEMU/KVM · Intel Xeon E5-2697A v4 · 2 cores · Python 3.12 · Mock adapter.

| ID | Scenario | Mean TPS | p95 avg |
| :--- | :--- | ---: | ---: |
| BM-INT-01 | Refund pipeline | 2,300/s | 0.66 ms |
| BM-INT-02 | Double-execution guard | 2,400/s | 0.67 ms |
| BM-INT-03 | Budget enforcement | 1,100/s | 331 ms |
| BM-INT-04 | Parallel throughput | 436/s | 542 ms |
| BM-INT-05 | MCP store round-trip | 3,000/s | 0.42 ms |
| BM-INT-06 | GovernanceEnvelope | 1,300/s | 171 ms |
| BM-INT-07 | Crash consistency | 7/s | 233 ms |
| BM-INT-08 | Replay equivalence | 1,300/s | 1.30 ms |
| BM-INT-09 | Adversarial retries | 2,400/s | 0.64 ms |
| BM-INT-10 | Long-horizon | 30/s | 3,606 ms |

Reproduce:

```bash
python benchmarks/stress_test.py
python benchmarks/benchmark_v030.py
python benchmarks/benchmark_v040.py
python benchmarks/run_all.py
python benchmarks/benchmark_double.py
python benchmarks/benchmark_nano_vm.py
python benchmarks/benchmark_stress_060
python benchmarks/benchmark_integration.py
```

---

## Comparison

| | LangChain | CrewAI | Temporal | **nano-vm** |
| :--- | :---: | :---: | :---: | :---: |
| LLM-native | ✅ | ✅ | ❌ | ✅ |
| Deterministic FSM | ❌ | ❌ | ✅ | ✅ |
| Replayable traces | partial | minimal | ✅ | ✅ |
| Suspend/resume | partial | partial | ✅ | ✅ |
| Runtime guardrails | ❌ | ❌ | partial | ✅ |
| Lightweight | ❌ | ❌ | ❌ | ✅ |
| Business workflows | partial | ❌ | ✅ | ✅ |
| AI workflows | ✅ | ✅ | partial | ✅ |

**vs Marvin / DSPy:** those optimize *what* the LLM produces (structured outputs, prompt
tuning). nano-vm controls *when* and *whether* steps run — orthogonal concerns, composable.

---

## When to Use

**Use nano-vm when:**

- workflow structure is known in advance
- correctness and auditability matter (fintech, compliance, enterprise)
- you need a reproducible trace for debugging or logging
- guardrails must be enforced at the system level, not in the prompt
- async orchestration with suspend/resume is required

**Do NOT use when:**

- workflow must be discovered fully at runtime
- the task is open-ended creative reasoning
- fully autonomous multi-agent coordination is required

---

## Roadmap

**Done:**

- [x] Deterministic FSM runtime (v0.1)
- [x] `parallel` steps — `asyncio.gather` (v0.2.0)
- [x] `retry` policy + `max_concurrency` (v0.3.0)
- [x] Budget guards: `max_steps`, `max_stalled_steps`, `max_tokens` (v0.4.0)
- [x] `state_snapshots` — sha256 fingerprint per step (v0.4.0)
- [x] `Planner` — intent → Program in 1 LLM call (v0.5.0)
- [x] FSM invariant stress suite — 13/13 · 1,020,000 ops (v0.6.0)
- [x] `suspend / resume` — `"PENDING"` sentinel + `CursorRepository` (v0.7.0)
- [x] `BudgetInterrupt` + `_emit_interrupt()` hook (v0.7.0)
- [x] `VaultStepResult` + `VaultStepMetadata` — MCP-compatible DTOs (v0.7.0)
- [x] `Trace.trace_id` — UUID4, OTel-ready (v0.7.0)
- [x] `erase()` — GDPR tombstoning with hash-chain preservation (v0.7.0)
- [x] `ASTEngine` — `eval()` removed; sandboxed condition evaluator (v0.7.0)
- [x] Integration benchmark suite — 10/10 · 1,096,500 ops (v0.7.3)
- [x] `Step.is_terminal`, `Step.next_step` — branch semantics (v0.7.4)
- [x] `$step_id.output` / `$step_id.output.field` resolution fix (v0.7.4)
- [x] `_resolve` typed return + multi-segment dotted path (v0.7.4)
- [x] ASTEngine METHOD_CALL guard — `ASTEvalError` at parse time (v0.7.5)
- [x] `py.typed` marker — PEP 561 (v0.7.4)
- [x] MCP server — `nano-vm-mcp` with GovernanceEnvelope, CapabilityRef, SSE + stdio

**Upcoming — DSL hardening (v0.8.x):**

- [ ] `Step.allowed_outputs` — LLM output validation against enum at step level
- [ ] `Step.timeout_seconds` + `asyncio.wait_for` in `_execute_llm`
- [ ] `ProgramValidator` — static analysis: unreachable steps, missing targets, cycle detection

**Upcoming — execution graph (v0.8.x):**

- [ ] `depends_on` + `TopologicalSorter` — declarative dependency graph over `parallel`

**Upcoming — observability (v0.8.x):**

- [ ] OpenTelemetry span per FSM step
- [ ] Incremental counters in `Trace`: `llm_calls`, `tool_calls`, `retries_total`

**Upcoming — gateway (v0.9.x):**

- [ ] `nano-vm-mcp`: StateContext SQLite persistence — close inter-session duplicate risk
- [ ] `nano-vm-mcp`: `idempotency_store` — inter-session exactly-once guarantee
- [ ] `nano-vm-mcp`: `GovernedToolExecutor` + circuit breaker
- [ ] Blueprint registry — `resume()` without explicit program argument
- [ ] `replan_on_interrupt` — Planner-driven continuation on budget interrupts

---

## Contact & Support

**Author:** [@ale007xd](https://t.me/ale007xd) on Telegram · [@ale007xd](https://x.com/ale007xd) on X

[![USDT (TON)](https://img.shields.io/badge/USDT%20(TON)-2ea2cc?style=flat-square)](https://tonviewer.com/UQCakyytrEGBikOi3eYMpveGHXDB1-fd6lcuQC9VvKqMrI-9)

**USDT (TON):** `UQCakyytrEGBikOi3eYMpveGHXDB1-fd6lcuQC9VvKqMrI-9`

---

## License

[MIT License](LICENCE).
