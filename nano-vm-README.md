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
  <strong>Deterministic parallel execution for LLM pipelines.</strong><br>
  Use when your workflow structure is known and correctness is non-negotiable.<br>
  Guardrails enforced by the VM, not by the prompt.
</p>

<p align="center">
  <em>LangChain = flexible but unpredictable &nbsp;·&nbsp; llm-nano-vm = predictable but still flexible</em>
</p>

---

## The Problem with LLM Agents

| | Prompting | LLM Agents | **llm-nano-vm** |
| :--- | :---: | :---: | :---: |
| Execution guarantee | ❌ none | ❌ at model's discretion | ✅ enforced by VM |
| Step skipping possible | ✅ yes | ✅ yes | ❌ never |
| Reproducible trace | ❌ | ❌ | ✅ |
| Debuggable | ❌ | hard | full trace |
| Cost/latency visibility | ❌ | partial | per-step |

> "LangChain cannot guarantee execution order. llm-nano-vm can."

---

## Mental Model

```
nondeterminism ∈ Planner (1 LLM call, optional)
determinism    ∈ ExecutionVM (FSM)
```

- **Planner** — LLM converts user intent → Program DSL
- **Program** — declarative workflow you define and version
- **ExecutionVM** — finite state machine; runs the program step by step
- **Trace** — full execution log: status, cost, tokens, duration per step

The LLM is a stateless worker. Control stays in your code.

---

## Execution Pipeline

Canonical model — every execution follows this pipeline without exception:

```
E  = LLM(input)       →  raw event (signal decoding, probabilistic)
E' = Validator(E)     →  validated + enriched context (deterministic)
A(S) = FSM(S, E')     →  allowed actions for current state (deterministic)
a*   = Policy(A, C)   →  selected action (deterministic pure function)
S'   = δ(S, a*)       →  next state (deterministic)
```

| Layer | Component | Trust | Role |
| :--- | :--- | :--- | :--- |
| Signal decoder | LLM / Planner | **untrusted** | converts input → event; may hallucinate |
| Validator | BlueprintCompiler | deterministic | schema + safety checks; enriches context |
| Control logic | ExecutionVM (FSM) | **source of truth** | defines allowed actions A(S) and transitions δ |
| Selector | Policy | deterministic pure fn | selects a* from A(S); no IO, no side effects |
| Effectors | Tools / MCP | enforced | executes a*; no control logic |

**Key invariant:** LLM output can influence *what content* is produced inside a step.
It cannot influence *which step runs next*, *whether a step is skipped*, or *when execution terminates*.
That is enforced by the VM, not by the prompt.

> **Current implementation note:** In the current release, `A(S)` typically contains
> a single action per step — Policy acts as a deterministic enforcement gate (allow/deny).
> The full `argmax`-based selection becomes relevant when multiple tool candidates exist
> per state (fallback tools, A/B execution paths). See Roadmap.

---

## FSM Transition Table

`ExecutionVM` is a finite state machine. The full δ-function:

| Current state | Step type | Outcome | Next state |
| :--- | :--- | :--- | :--- |
| `RUNNING` | `llm` | success | `RUNNING` (advance to next step) |
| `RUNNING` | `llm` | all retries exhausted | `FAILED` |
| `RUNNING` | `tool` | success | `RUNNING` |
| `RUNNING` | `tool` | returns sentinel `"PENDING"` | `SUSPENDED` |
| `RUNNING` | `tool` | error, `on_error=fail` | `FAILED` |
| `RUNNING` | `tool` | error, `on_error=skip` | `RUNNING` (output=`None`) |
| `RUNNING` | `condition` | branch taken | `RUNNING` (jump to `then`/`otherwise`) |
| `RUNNING` | `condition` | no branch matches | `FAILED` |
| `RUNNING` | `parallel` | all sub-steps done | `RUNNING` |
| `RUNNING` | any | `max_steps` exceeded | `BUDGET_EXCEEDED` |
| `RUNNING` | any | `max_tokens` exceeded | `BUDGET_EXCEEDED` |
| `RUNNING` | any | `max_stalled_steps` exceeded | `STALLED` |
| `RUNNING` | — | no more steps | `SUCCESS` |
| `SUSPENDED` | — | `resume_with_program()` called | `RUNNING` (from cursor) |
| `FAILED` / `SUCCESS` / `BUDGET_EXCEEDED` / `STALLED` | — | — | terminal (no further transitions) |

Terminal states are absorbing — once reached, no further step is executed. The append-only trace invariant holds: a step that reached `SUCCESS` cannot execute again within the same session.

---

## Install

```bash
pip install llm-nano-vm
pip install llm-nano-vm[litellm]   # for built-in provider support
```

---

## Quick Start — Guardrail That Never Skips

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
            "id": "guardrail",           # ALWAYS runs — VM enforces it
            "type": "condition",
            "condition": "'yes' in '$decision'.lower()",
            "then": "process_refund",
            "otherwise": "reject",
        },
        {
            "id": "process_refund",
            "type": "tool",
            "tool": "issue_refund",
        },
        {
            "id": "reject",
            "type": "tool",
            "tool": "send_rejection",
        },
    ],
})

vm = ExecutionVM(
    llm=LiteLLMAdapter("openai/gpt-4o-mini"),
    tools={"issue_refund": ..., "send_rejection": ...},
)
trace = await vm.run(program, context={"user_input": "I was charged twice"})

print(trace.trace_id)         # "3f2a1b4c-..." (UUID4, stable for OTel propagation)
print(trace.status)           # SUCCESS
print(trace.final_output)     # tool result
print(trace.total_cost_usd()) # e.g. 0.000034
```

The `guardrail` step cannot be skipped, reordered, or overridden by the model.
That is the guarantee.

---

## Suspend / Resume — Webhook-Driven Execution

`ExecutionVM` can suspend mid-graph when a tool returns the sentinel `"PENDING"` and
resume from a persisted cursor when an external event arrives (payment webhook, courier
confirmation, etc.).

```python
from nano_vm.vm import ExecutionVM, InMemoryCursorRepository

# Tool suspends execution by returning "PENDING"
async def initiate_payment(**kwargs) -> str:
    await register_webhook_handler(order_id=kwargs["order_id"])
    return "PENDING"   # VM sees this, suspends, persists cursor

vm = ExecutionVM(
    llm=adapter,
    tools={"initiate_payment": initiate_payment, ...},
    cursor_repo=InMemoryCursorRepository(),   # swap for SqliteCursorRepository in production
)

trace = await vm.run(program, context={"order_id": "123"})
assert trace.status == TraceStatus.SUSPENDED

# Resume when webhook fires
trace = await vm.resume_with_program(
    program=program,
    trace_id=trace.trace_id,
    webhook_event={"type": "payment.confirmed", "order_id": "123"},
)
assert trace.status == TraceStatus.SUCCESS
```

**`resume()` vs `resume_with_program()`:** `resume()` requires a Blueprint registry
(planned for P8). Until then, pass the program explicitly via `resume_with_program()`.

`InMemoryCursorRepository` ships for tests and dry-run. For production, implement the
`CursorRepository` Protocol backed by `SqliteCursorRepository(infrastructure.db)`.

---

## Budget Interrupts

Budget exhaustion is emitted as a **system interrupt** (`BudgetInterrupt`), not a
condition branch. The VM raises `InterruptType.BUDGET` before touching the next step —
the LLM cannot observe or influence it.

Override `_emit_interrupt()` in a subclass to hook into your observability stack:

```python
from nano_vm.vm import ExecutionVM, InterruptType

class InstrumentedVM(ExecutionVM):
    async def _emit_interrupt(self, interrupt_type: InterruptType) -> None:
        await notify_operator(f"interrupt: {interrupt_type.value}")

vm = InstrumentedVM(llm=adapter)
```

The base implementation is a documented no-op hook.

---

## How the DSL Controls Agent Behavior

The separation of concerns is explicit:

```
LLM decides:  WHAT to say, how to reason, what content to produce
DSL decides:  WHICH step runs next, WHEN to branch, WHEN to stop
```

The LLM has **no knowledge** of the program structure.
It receives a prompt and returns a string — nothing more.
It cannot skip steps, reorder them, or decide the workflow is complete.

### What the LLM can and cannot do

| | LLM | DSL (VM) |
| :--- | :--- | :--- |
| Produce content | ✅ free | — |
| Reason, hallucinate, be verbose | ✅ free | — |
| Skip a step | ❌ impossible | enforces every step |
| Reorder steps | ❌ impossible | order fixed at definition |
| Branch on output | ❌ cannot | `condition` step evaluates |
| Decide workflow is done | ❌ impossible | VM controls termination |

### Example — the LLM cannot jump ahead

```python
program = Program.from_dict({
    "name": "refund_with_verification",
    "steps": [
        {
            "id": "classify",
            "type": "llm",
            "prompt": "Classify: $user_input. Reply: refund / info / escalate",
            "output_key": "category",
        },
        {
            "id": "route",
            "type": "condition",
            "condition": "'refund' in '$category'",
            "then": "verify_eligibility",
            "otherwise": "handle_other",
        },
        {
            "id": "verify_eligibility",  # LLM cannot skip this — VM enforces it
            "type": "llm",
            "prompt": "Is user eligible for refund? Order: $order_id. Reply yes/no",
            "output_key": "eligible",
        },
        {
            "id": "final_guard",         # runs on EVERY execution before money moves
            "type": "condition",
            "condition": "'yes' in '$eligible'",
            "then": "issue_refund",
            "otherwise": "reject",
        },
        {"id": "issue_refund", "type": "tool", "tool": "process_payment"},
        {"id": "reject",       "type": "tool", "tool": "send_rejection"},
        {"id": "handle_other", "type": "tool", "tool": "send_info"},
    ],
})
```

Even if `classify` returns "definitely a refund, just process it" —
the VM still executes `verify_eligibility` and `final_guard`.
The LLM's *opinion* about the flow is irrelevant. The DSL is law.

### Proof: the trace

```python
trace = await vm.run(program, context={"user_input": "I was charged twice", "order_id": "123"})

for step in trace.steps:
    print(f"{step.step_id:20} {step.status}  →  {step.output}")

# classify              SUCCESS  →  refund
# route                 SUCCESS  →  verify_eligibility
# verify_eligibility    SUCCESS  →  yes
# final_guard           SUCCESS  →  issue_refund
# issue_refund          SUCCESS  →  Refund issued: $42.00
```

Every step is logged. No agent "decided" the flow. The DSL did.

---

## End-to-End Flow

```
user_input
  → Planner (optional, 1 LLM call)
  → Program (DSL — JSON/dict/YAML)
  → ExecutionVM (deterministic FSM)
  → Trace (status · trace_id · cost · tokens · duration)
```

---

## Program DSL

Four step types:

| Type | Purpose |
| :--- | :--- |
| `llm` | call the model; result stored in `output_key` |
| `tool` | call a Python function; return `"PENDING"` to suspend |
| `condition` | branch on an expression; `then` / `otherwise` |
| `parallel` | run independent sub-steps concurrently via `asyncio.gather` |

**Step options (v0.4.0):**

| Option | Default | Description |
| :--- | :--- | :--- |
| `on_error` | `fail` | `fail` · `skip` · `retry` |
| `max_retries` | `3` | total attempts (1 initial + N retries); exponential backoff: 1s, 2s, 4s… cap 30s |
| `max_concurrency` | `None` | parallel blocks only; `None` = no cap (all sub-steps at once) |

**Program budget options (v0.4.0):**

| Option | Default | Description |
| :--- | :--- | :--- |
| `max_steps` | `None` | max total steps executed; `BUDGET_EXCEEDED` if exceeded before next step |
| `max_stalled_steps` | `None` | max consecutive no-op steps (same state fingerprint); `STALLED` if exceeded |
| `max_tokens` | `None` | max total tokens across all LLM steps; `BUDGET_EXCEEDED` if exceeded before next step |

### Variable interpolation

| Syntax | Resolves to |
| :--- | :--- |
| `$key` | value from initial context |
| `$step_id.output` | output of a previous step |

> **⚠ Security note — condition expressions:**  
> `condition` strings are evaluated by the **ASTEngine** — a deterministic, sandboxed
> evaluator. No Python builtins are accessible. **Do not interpolate raw user input
> into condition expressions.** Condition logic should be authored by you (the developer).
> LLM output used as a branching signal should only appear in context variables that your
> condition *tests* (e.g. `'yes' in '$decision'`), never as the condition expression itself.

### Example — multi-step pipeline

```json
{
  "name": "doc_pipeline",
  "steps": [
    { "id": "extract",   "type": "tool", "tool": "extract_text",   "output_key": "raw_text" },
    { "id": "summarize", "type": "llm",  "prompt": "Summarize: $raw_text", "output_key": "summary" },
    { "id": "check",     "type": "condition",
      "condition": "len('$summary') > 100",
      "then": "store", "otherwise": "flag" },
    { "id": "store",     "type": "tool", "tool": "save_to_db" },
    { "id": "flag",      "type": "tool", "tool": "flag_for_review" }
  ]
}
```

### Example — parallel steps (v0.2.0+)

```python
program = Program.from_dict({
    "name": "enrich",
    "steps": [
        {
            "id": "fetch",
            "type": "parallel",
            "output_key": "fetched",
            "max_concurrency": 5,
            "on_error": "skip",
            "parallel_steps": [
                {"id": "weather", "type": "tool", "tool": "get_weather", "args": {"city": "$city"}},
                {"id": "news",    "type": "tool", "tool": "get_news",    "args": {"topic": "$topic"}},
            ],
        },
        {
            "id": "summarize",
            "type": "llm",
            "prompt": "Weather: $weather.output\nNews: $news.output\nSummarize. If a field is None, skip it.",
        },
    ],
})
```

`fetch` runs both tools concurrently via `asyncio.gather`. Wall-clock time = slowest single sub-step.
Sequential execution resumes at `summarize` only after all sub-steps complete (or are skipped).

**Partial result contract:** if a sub-step fails with `on_error: skip`, its output is set to `None`.
Downstream steps receive `None` — not an absent key, not an exception.

---

## MCP-Compatible Contracts (v0.7.0)

`VaultStepResult` and `VaultStepMetadata` are DTOs for vault / MCP integration.
`status` is a plain string (`"SUCCESS" | "FAILED" | "PENDING"`), not an enum —
required for round-trip JSON serialization through the MCP layer.

```python
from nano_vm.models import VaultStepResult, VaultStepMetadata
from uuid import uuid4

result = VaultStepResult(
    status="SUCCESS",
    data={"payment_id": "pay_123"},
    metadata=VaultStepMetadata(
        idempotency_key="order_1:pay_step:initiate_payment",
        execution_time_ms=142,
        tool_version="1.0.0",
        cached=False,
        trace_id=str(uuid4()),
    ),
)
```

`@model_validator` enforces `status ∈ {"SUCCESS", "FAILED", "PENDING"}` at construction time.

---

## Testing — Deterministic by Design

`MockLLMAdapter` ships with the package for writing tests without a real LLM:

```python
from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.adapters import MockLLMAdapter

# Always returns the same string
vm = ExecutionVM(llm=MockLLMAdapter("SAFE"))

# Per-call sequence
vm = ExecutionVM(llm=MockLLMAdapter(["SAFE", "yes"]))

# Per-prompt mapping (substring match on last user message)
vm = ExecutionVM(llm=MockLLMAdapter({
    "Classify": "SAFE",
    "eligible": "yes",
    "__default__": "ok",
}))

trace = await vm.run(program, context={"user_input": "refund"})
assert trace.status == TraceStatus.SUCCESS
assert [s.step_id for s in trace.steps] == ["classify", "route", "verify_eligibility", ...]
```

Same input → same step sequence. Always. Testable in CI without any API key.

### State Determinism vs Semantic Determinism

llm-nano-vm guarantees **State Determinism**: given a Program, the VM executes
steps in the order the DSL defines, never skips a required step, and produces
a complete, reproducible trace — regardless of what the LLM returns.

It does **not** guarantee **Semantic Determinism**: the text content produced by
an LLM step may differ across runs even at `temperature=0.0`. Use `MockLLMAdapter`
when you need both.

| | State Determinism | Semantic Determinism |
| :--- | :---: | :---: |
| Step execution order | ✅ VM enforces | — |
| Step cannot be skipped | ✅ VM enforces | — |
| Invariants hold (no double-execution) | ✅ VM enforces | — |
| LLM output identical across runs | — | ❌ not guaranteed |
| Reproducible trace structure | ✅ always | — |
| Reproducible trace content | — | ❌ depends on LLM |

---

## Observability

```python
trace.trace_id              # UUID4 — stable identifier for OTel propagation
trace.status                # TraceStatus.SUCCESS | FAILED | SUSPENDED | BUDGET_EXCEEDED | STALLED
trace.final_output          # last step output
trace.total_tokens()        # O(1) — incremental accumulator
trace.total_cost_usd()      # sum across all steps (requires LiteLLMAdapter)
trace.state_snapshots       # list[(step_index, sha256_hex)] — one entry per executed step
trace.error                 # set on FAILED / BUDGET_EXCEEDED / STALLED

for step in trace.steps:
    print(step.step_id, step.status, step.duration_ms, step.usage)
```

Parallel blocks expose sub-step hierarchy:

```python
# fetch              SUCCESS   142ms  usage=None
#   ├─ weather       SUCCESS    98ms  usage=None
#   └─ news          SKIPPED   429ms  usage=None
# summarize          SUCCESS  1204ms  usage=TokenUsage(prompt=312, completion=87)
```

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

- exactly **1** LLM call
- outputs a validated `Program` object
- non-deterministic input → deterministic execution
- signature stable since v0.5.0

---

## Custom Adapter

Any object implementing the async Protocol works:

```python
class MyAdapter:
    async def complete(self, messages: list[dict], **kwargs) -> str:
        ...  # call any LLM API
```

Built-in adapters via `[litellm]` extra:

```python
LiteLLMAdapter("groq/llama-3.3-70b-versatile")
LiteLLMAdapter("openrouter/llama-3.3-70b-instruct:free")
LiteLLMAdapter("ollama/llama3")
LiteLLMAdapter("openai/gpt-4o-mini")
```

---

## Performance

The VM itself introduces near-zero overhead. Your bottleneck is the LLM API.

Benchmarked on Linux, 2-core VPS, Python 3.12.3. Mock adapter (no I/O).

| Metric | Scenario | Throughput | Latency | Overhead |
| :--- | :--- | :--- | :--- | :--- |
| BM1: retry baseline | 0 retries | 3509 RPS | 0.285 ms | — |
| BM1: retry path | 2 retries | 4308 RPS | 0.232 ms | v0.3.0 parity ✅ |
| BM5: max_steps | no budget (baseline) | 558 RPS | 1.793 ms | — |
| BM5: max_steps | max_steps=1000 active | 616 RPS | 1.623 ms | **±9.5% (noise)** |
| BM7: max_tokens | no budget (baseline) | 458 RPS | 2.184 ms | — |
| BM7: max_tokens | budget active | 420 RPS | 2.379 ms | **+8.9%** |
| Parallel steps (20) | OpenRouter (network) | 11.38 steps/sec | 1.7574 s | — |
| BM11: Planner | determinism check | — | — | ✅ unique fingerprints=1 |
| BM8: multi-model | OpenRouter free tier | pending | pending | rate limit — off-peak run |
| **BM_double: raw agent** | 1000 runs, fail_prob=0.3 | — | — | **~20% double-executions** |
| **BM_double: FSM runtime** | 3000 runs total | — | — | **0 double-executions** |

> **BM7 fixed in v0.5.0:** `total_tokens()` O(1) via incremental `_token_accumulator` in `Trace.add_step`.  
> **BM_double:** structural guarantee — FSM trace invariant `I_k(T) ∈ {0,1}`, not a retry policy.

### v0.6.0 — FSM invariant stress suite

Validates δ(S, E) → S' under chaos, injection, replay, and concurrent load.
Array size: **10,000** per test · **5 runs** · seed=42 · Python 3.12 · real `llm-nano-vm` installed.

```
System: Linux · x86_64 (2 cores) · Python 3.12 · venv
Suite:  13 tests (BM-01–BM-12 + BM-VM)
Result: 13/13 PASSED · Score 100% · ⬢ DETERMINISTIC EXECUTION RUNTIME VERIFIED
```

| Tag | Test | Mean ms | Throughput /s | Key metric |
| :--- | :--- | ---: | ---: | :--- |
| BM-01 | Idempotency Under Replay Stress | 279 | 35,794 | 450k replays · **0 violations** · cache hit 100% |
| BM-02 | Duplicate Execution Attack | 222 | 45,114 | 50k double-triggers · **0 double executions** |
| BM-03 | Crash Mid-Step Recovery | 170 | 58,741 | 50k crash/resume cycles · **0 wrong resumes** |
| BM-04 | Non-Deterministic LLM Injection | 68 | 148,018 | 13 noise variants · **0 FSM-influenced transitions** |
| BM-05 | Tool Failure Cascade A→B→C | 135 | 73,847 | fail_prob=40% · **0 cascade violations** |
| BM-06 | Long-Running Tool + Timeout Drift | 73 | 137,531 | 66.8% timeout rate · **0 partial transitions** |
| BM-07 | Out-of-Order Event Delivery | 123 | 81,234 | shuffled sequences · **0 invalid accepted** |
| BM-08 | State Explosion / Memory Pressure | 486 | 20,567 | 70k transitions · **StateContext bounded \|S\|=12** |
| BM-09 | Partial StepResult Corruption | 66 | 151,479 | 8 corruption types · **50k/50k normalized** |
| BM-10 | Transition Validity Invariant | 123 | 81,068 | 90.5% blocked · **0 direct mutations** |
| BM-11 | Reentrancy Stress | 175 | 57,187 | 2–8 concurrent calls · **0 double mutations** |
| BM-12 | Chaos Mode — Full System Stress | 2352 | 4,252 | 83k escalations · **0 invalid final states** |
| BM-VM | nano-vm Double Execution Safety | 53 | 190,428 | 300 real `vm.run` · **0 double executions** |

**Total operations across suite: 1,020,000**

### v0.7.0 — Stress test (10k FSM graphs × 5 runs, Mock adapter)

```
┌──────┬──────────────┬────────────┬────────┬────────┐
│ Run  │ Time (sec)   │ Speed      │ OK     │ Failed │
├──────┼──────────────┼────────────┼────────┼────────┤
│  1   │   0.70       │ 14 286 /s  │  8973  │  1027  │
│  2   │   0.70       │ 14 286 /s  │  8973  │  1027  │
│  3   │   0.69       │ 14 493 /s  │  8973  │  1027  │
│  4   │   0.70       │ 14 286 /s  │  8973  │  1027  │
│  5   │   0.70       │ 14 286 /s  │  8973  │  1027  │
├──────┼──────────────┼────────────┼────────┼────────┤
│ AVG  │   0.70       │ 14 327 /s  │   —    │   —    │
└──────┴──────────────┴────────────┴────────┴────────┘
```

89.73% success rate matches `P(value ≤ 0.9) = 0.9` — 1027 errors from `unregistered_tool_to_force_error`
triggered at `value > 0.9`. Identical results across all 5 runs (dataset fixed before loop).
`VMError: Tool not found` caught per-coroutine; event loop continues across 200 concurrent tasks.

Reproduce locally:

```bash
pip install llm-nano-vm[litellm]
python benchmarks/stress_test.py
python benchmarks/benchmark_v030.py
python benchmarks/benchmark_v040.py
python benchmarks/run_all.py              # BM1–BM11 (BM8 requires OPENROUTER_API_KEY)
python benchmarks/benchmark_double.py
python benchmarks/benchmark_nano_vm.py   # v0.6.0 FSM invariant suite
python benchmarks/benchmark_stress_060   # v0.7.0 10k stress
```

---

## When to Use

**Use llm-nano-vm when:**

- the workflow structure is known in advance
- correctness and auditability matter (fintech, compliance, enterprise)
- you need a reproducible trace for debugging or logging
- you want guardrails enforced at the system level, not in the prompt

**Do NOT use when:**

- the workflow is unknown and must be discovered at runtime
- the task is open-ended creative reasoning
- you need fully autonomous multi-agent coordination

---

## Comparison

| | LangChain | AutoGPT / CrewAI | Prefect / Airflow | **llm-nano-vm** |
| :--- | :--- | :--- | :--- | :--- |
| Layer | orchestration | reasoning / autonomy | workflow scheduler | execution guarantees |
| Execution order | flexible | model-driven | enforced | enforced |
| Guardrails | prompt-level | prompt-level | task-level | VM-level |
| Parallel execution | manual | model-driven | native | scoped, deterministic |
| Trace | partial | minimal | job logs | full, per-step + sub-step |
| LLM-native | yes | yes | no | yes |
| Overhead | heavy | heavy | heavy | near-zero (stdlib only) |
| Best for | flexible pipelines | autonomous tasks | data/ETL pipelines | compliance-grade LLM workflows |

**vs Marvin / DSPy:** those optimize *what* the LLM produces (structured outputs, prompt tuning). llm-nano-vm controls *when* and *whether* steps run — orthogonal concerns, composable.

---

## Roadmap

- [x] FSM execution engine (v0.1)
- [x] `llm / tool / condition` step types
- [x] LiteLLM adapter + cost tracking
- [x] Published to PyPI as `llm-nano-vm`
- [x] `parallel` steps — `asyncio.gather` for independent sub-steps (v0.2.0)
- [x] `MockLLMAdapter` — deterministic testing without API keys (v0.2.0)
- [x] `max_concurrency` — cap concurrent sub-steps per parallel block (v0.3.0)
- [x] `retry` policy per sub-step — exponential backoff, max_attempts (v0.3.0)
- [x] `max_steps` budget — BUDGET_EXCEEDED after N steps (v0.4.0)
- [x] `max_stalled_steps` — STALLED on N consecutive no-op state fingerprints (v0.4.0)
- [x] `max_tokens` budget — BUDGET_EXCEEDED when token count exceeds limit (v0.4.0)
- [x] `state_snapshots` — sha256 fingerprint per step in Trace (v0.4.0)
- [x] `Planner` — LLM intent → validated Program in 1 call; determinism confirmed (v0.5.0)
- [x] Benchmark suite BM1–BM11 (`benchmarks/run_all.py`) (v0.5.0)
- [x] Double-execution safety benchmark — 0/3000 FSM vs ~20% stateless (v0.5.0)
- [x] `total_tokens()` O(1) — incremental `_token_accumulator` in `Trace.add_step` (v0.5.0)
- [x] MCP server — `run_program`, `get_trace`, `list_programs`, `get_program`, `delete_program` · stdio + SSE · bearer auth · SQLite WAL ([nano-vm-mcp](https://github.com/Ale007XD/nano-vm-mcp))
- [x] FSM invariant stress suite BM-01–BM-12 + BM-VM — 13/13 PASS · 1,020,000 ops · 0 violations (v0.6.0)
- [x] `suspend / resume` — `"PENDING"` sentinel + `CursorRepository` + `resume_with_program()` (v0.7.0)
- [x] `BudgetInterrupt` — isolated system interrupt, `_emit_interrupt()` hook (v0.7.0)
- [x] `VaultStepResult` + `VaultStepMetadata` — MCP-compatible DTOs (v0.7.0)
- [x] `Trace.trace_id` — UUID4, OTel-ready (v0.7.0)
- [x] `erase()` — nested `CapabilityRef` tombstoning; GDPR erasure with hash-chain preservation (v0.7.0)
- [x] `ASTEngine` — `eval()` removed from condition steps; deterministic sandboxed evaluator (v0.7.0)
- [ ] BM8 real-latency numbers — pending off-peak OpenRouter run
- [ ] Blueprint registry (P8) — enables `resume()` without explicit program argument
- [ ] REST API — pay-per-run, API keys (nano-vm-server)

---

## 💼 llm-nano-vm Pro

- 🆓 **Core** (this repo) — MIT, fully open-source
- 💼 **Pro layer** — planned commercial extensions

Planned Pro features:

- 📊 Visual execution graph (Trace UI)
- 🌐 Distributed multi-node execution
- 🔄 Provider pools & smart routing
- 🔐 Access control & multi-user support
- 📈 Cost analytics dashboard

---

## Contact & Support

**Author:** [@ale007xd](https://t.me/ale007xd) on Telegram · [@ale007xd](https://x.com/ale007xd) on X

### ☕ Support the project

[![Buy Me a Coffee](https://img.shields.io/badge/☕-Buy%20Me%20a%20Coffee-yellow?style=flat-square)](https://www.buymeacoffee.com/ale007xd)
[![USDT (TON)](https://img.shields.io/badge/USDT%20(TON)-2ea2cc?style=flat-square&logo=ton)](https://tonviewer.com/UQCakyytrEGBikOi3eYMpveGHXDB1-fd6lcuQC9VvKqMrI-9)

**Direct wallet — USDT (TON):**
```
UQCakyytrEGBikOi3eYMpveGHXDB1-fd6lcuQC9VvKqMrI-9
```

---

## License

This project is licensed under the [MIT License](LICENCE).
