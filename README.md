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
  <strong>Execution governance runtime with deterministic receipts and replayable traces.</strong><br>
  LLMs are signal generators. Execution authority belongs to the runtime.
</p>

---

## What nano-vm produces

Most AI frameworks answer: *how do you coordinate agents?*  
Almost nobody answers: *what proof do you have that execution happened correctly?*

nano-vm answers the second question by producing verifiable artifacts at every layer:

| Artifact | What it is | How it's produced |
| :--- | :--- | :--- |
| **Trace** | Step-by-step execution record with SHA-256 Merkle chain | Runtime — one entry per step |
| **ExecutionReceipt** | Minimal decision state derived from Trace | `TraceAnalyzer.receipt()` — deterministic projection |
| **GovernanceEnvelope** | Per-step policy hash + canonical state snapshot | Runtime — stored in SQLite WAL |

The core contract:

```
Receipt = f(Trace)
```

The receipt is a deterministic, recomputable projection of the trace. It never contains information that isn't already in the trace. It never requires the LLM to generate it.

```
Program
  ↓
Validator        ← static analysis before execution
  ↓
ExecutionVM      ← FSM transition authority
  ↓
Trace            ← authoritative execution record
  ↓
TraceAnalyzer    ← post-hoc interpretation
  ↓
Receipt          ← minimal state for continuation decisions
```

---

## Install

```bash
pip install llm-nano-vm
pip install llm-nano-vm[litellm]   # for LLM provider support
```

---

## Quick Start — Tool Pipeline

LLMs are not required. nano-vm runs as a pure deterministic workflow engine.

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

The runtime still guarantees: deterministic ordering, replayable execution,
trace visibility, transition enforcement, idempotent re-execution across restarts.

---

## Quick Start — LLM Pipeline

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
            "allowed_outputs": ["yes", "no"],   # runtime enum gate — not a prompt hint
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

## Suspend / Resume

Return `"PENDING"` from any tool to suspend execution:

```python
async def initiate_payment(**kwargs) -> str:
    await register_webhook(kwargs["order_id"])
    return "PENDING"   # FSM → SUSPENDED, cursor persisted
```

FSM transition: `RUNNING → SUSPENDED → RUNNING → SUCCESS`

This enables: payment settlement, courier confirmation, approval systems,
human-in-the-loop, webhook orchestration. The process can restart. The cursor survives.

```python
trace = await vm.run(program, context={"order_id": "123"})
assert trace.status.name == "SUSPENDED"

trace = await vm.resume_with_program(
    program=program,
    trace_id=trace.trace_id,
    webhook_event={"type": "payment.confirmed", "order_id": "123"},
)
assert trace.status.name == "SUCCESS"
```

> **Note:** `"PENDING"` is a reserved FSM sentinel. Use `"REQUIRES_ACTION"` or `"AWAITING_3DS"` for domain-specific states.

---

## LLM Output Enforcement — `allowed_outputs`

Validates the model's raw output against an explicit enum *before* it enters the FSM context. This is a runtime gate, not a prompt hint.

```python
{
    "id": "classify",
    "type": "llm",
    "prompt": "Classify the request. Reply ONLY with: refund / query / other",
    "output_key": "category",
    "allowed_outputs": ["refund", "query", "other"],
    "on_error": "skip",   # output → "refund" (first element) on mismatch
}
```

| `on_error` | On mismatch |
| :--- | :--- |
| `fail` (default) | `VMError` → `trace.FAILED` |
| `skip` | output replaced with `allowed_outputs[0]` |
| `retry` | retry up to `max_retries`; `VMError` if exhausted |

Per-step LLM timeout:

```python
{
    "id": "classify",
    "type": "llm",
    "timeout_seconds": 10.0,
    "on_timeout": "fail",   # or "fallback" → allowed_outputs[0] or ''
}
```

---

## ProgramValidator — Pre-flight Static Analysis

Validates a program before execution. Catches structural issues that would cause runtime failures.

```python
from nano_vm import ProgramValidator

validator = ProgramValidator(program)
report = validator.validate()

print(report.is_valid())   # False if any ERROR-severity issue found
for issue in report.issues:
    print(issue.severity, issue.code, issue.message)
```

Checks performed:

| Check | Severity | Description |
| :--- | :--- | :--- |
| `missing_targets` | ERROR | branch targets that reference non-existent steps |
| `unreachable_steps` | ERROR | steps unreachable from any execution path |
| `cycle_detection` | ERROR | cycles that would cause infinite loops |
| `no_failure_terminal` | WARNING | no reachable terminal step with a failure outcome |

`is_valid()` returns `True` only when no `ERROR`-severity issues exist. `WARNING` is informational — simple linear programs without failure terminals are valid.

---

## TraceAnalyzer — Post-hoc Execution Analysis

Analyzes a completed trace. Pure post-processing — no changes to the runtime state.

```python
from nano_vm import TraceAnalyzer

analyzer = TraceAnalyzer(trace)
report = analyzer.report()     # TraceHealthReport — lazy, cached

print(report.rollback_density)          # 0.0 – 1.0
print(report.tool_churn_rate)           # 0.0 – 1.0
print(report.path_variance)             # 0.0 – 1.0
print(report.transition_entropy)        # bits
print(report.invariant_violation_rate)  # 0.0 – 1.0
```

Alert thresholds (informational — do not interrupt FSM):

| Metric | Alert threshold | Signal |
| :--- | ---: | :--- |
| `rollback_density` | > 0.3 | excessive compensating transitions |
| `tool_churn_rate` | > 0.4 | unstable tool selection |
| `path_variance` | > 0.5 | non-deterministic branching |
| `transition_entropy` | > 2.5 bits | ~5+ unique transition pairs |
| `invariant_violation_rate` | > 0.2 | repeated constraint failures |

Alerts are warnings, not errors. The FSM is never interrupted by the analyzer.

---

## ExecutionReceipt — Receipt = f(Trace)

The receipt is a deterministic projection of the trace. It contains the minimal state needed for a continuation decision — whether to resume, replay, or escalate.

```python
receipt = analyzer.receipt()   # lazy, cached; recomputable at any time

print(receipt.trace_id)
print(receipt.final_status)         # TraceStatus
print(receipt.resumable)            # bool
print(receipt.replayable)           # bool
print(receipt.failed_steps)         # int
print(receipt.retried_steps)        # int
print(receipt.rejected_transitions) # tuple[RejectedTransition, ...]
print(receipt.health)               # TraceHealthReport
```

`RejectedTransition` captures each failed step with its reason and timestamp:

```python
for rt in receipt.rejected_transitions:
    print(rt.step_id, rt.reason, rt.timestamp)
```

**Contract invariants:**

- `Receipt ⊆ Trace` — the receipt never contains information outside the trace
- No LLM generation — the receipt is computed, never summarized
- No operator-specific fields — the receipt is infrastructure, not application logic
- Recomputable — given the same trace, `receipt()` always returns the same result

---

## Evaluator Blindness by Design

Benchmark research across 10 frontier models found that every model changes behavior when it detects it's being monitored. The best model scored 84% on "acts the same whether watched or not."

The `ProjectionLayer` gives the LLM only a `target=LLM` projection of state. Governance metadata — `GovernanceEnvelope`, `canonical_hash`, `policy_hash`, retry counters — never reaches the prompt. The model cannot observe its own audit trail.

**Evaluator blindness is structural, not configured.**

---

## FSM Transition Model

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

Failure states are first-class outcomes. `FAILED`, `NEED_HELP`, `INSUFFICIENT_DATA`, `POLICY_BLOCKED` are legitimate terminal states equivalent to `SUCCESS` — not exceptions to be swallowed.

---

## Program DSL

Four step types:

| Type | Purpose |
| :--- | :--- |
| `llm` | call the model; result stored in `output_key` |
| `tool` | call a Python function; return `"PENDING"` to suspend |
| `condition` | branch on an expression; `then` / `otherwise` |
| `parallel` | run independent sub-steps concurrently via `asyncio.gather` |

**Step fields:**

| Field | Default | Description |
| :--- | :--- | :--- |
| `on_error` | `fail` | `fail` · `skip` · `retry` |
| `max_retries` | `3` | total attempts; backoff: 1s, 2s, 4s… cap 30s |
| `max_concurrency` | `None` | parallel blocks only |
| `is_terminal` | `False` | return `SUCCESS` after this step (leaf nodes) |
| `next_step` | `None` | jump to named step instead of returning `SUCCESS` |
| `allowed_outputs` | `None` | LLM-only — accepted output enum; `ValidationError` if empty |
| `timeout_seconds` | `None` | LLM-only — `asyncio.wait_for` timeout in seconds |
| `on_timeout` | `'fail'` | `'fail'` · `'fallback'` (→ `allowed_outputs[0]` or `''`) |

**Program budget options:**

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

### Condition expressions

> **⚠ ASTEngine replaces `eval()`.** Conditions are parsed into a validated JSON AST
> and evaluated by a pure, sandboxed interpreter. No Python builtins are accessible.

**Supported:** `==`, `!=`, `>`, `<`, `in`, `not in`, `and`, `or`, `not`, `contains`, dotted-path `$var.field`.

**Not supported:** method calls (`.lower()`, `.strip()`), arithmetic, parentheses grouping. Using an unsupported form raises `ASTEvalError` at parse time.

```python
# ❌ WRONG — method call raises ASTEvalError
{"condition": "'yes' in '$decision'.lower()"}

# ✅ CORRECT
{"condition": "'yes' in '$decision'"}
```

---

## MCP Integration

nano-vm pairs with [nano-vm-mcp](https://github.com/Ale007XD/nano-vm-mcp) — an MCP gateway
that exposes `run_program`, `get_trace`, `list_programs`, `get_program`, `delete_program`
over stdio or SSE transport with bearer auth, SQLite WAL persistence, and GovernanceEnvelope audit trail.

```
Claude Code / MCP Client
        ↓
  nano-vm-mcp              ← decides how execution is allowed to proceed
        ↓
  ExecutionVM              ← transition authority
        ↓
  GovernanceEnvelope       ← proves it happened
```

### GovernanceEnvelope

Each successful execution step produces a `GovernanceEnvelope` stored in SQLite WAL:

| Field | Description |
| :--- | :--- |
| `execution_id` | Session / trace identifier |
| `step_id` | Step index within the execution |
| `policy_hash` | SHA-256 of the active `PolicySnapshot` |
| `canonical_snapshot_hash` | Merkle/delta hash of `CanonicalState` |
| `payload` | Projected (sanitized) step output |

### CapabilityRef and GDPR Tombstoning

Sensitive values are stored as `CapabilityRef` tokens (`vault://secret/<id>`).
On a GDPR erasure event, the ref is tombstoned. All subsequent projections return
`[REDACTED_TOMBSTONE]`, preserving the hash chain. Forensic auditability survives erasure.

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

**State Determinism vs Semantic Determinism:** nano-vm guarantees step execution order, no skipping, reproducible trace structure — regardless of LLM output. It does not guarantee that LLM text is identical across runs. Use `MockLLMAdapter` for both.

---

## Performance

| Suite | Result |
| :--- | :--- |
| FSM invariant suite | 13/13 · 1,020,000 ops · 0 violations |
| Integration suite | 10/10 · 1,096,500 ops · 0 violations |
| 10k stress | 14,327 graphs/sec · 0.70 s/run |
| MoMo PoC v4 | 9/9 PASS |
| Stripe PoC v1 | 9/9 PASS |

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

Environment: QEMU/KVM · Intel Xeon E5-2697A v4 · 2 cores · Python 3.12 · Mock adapter.

---

## Comparison

| | LangChain | CrewAI | Temporal | **nano-vm** |
| :--- | :---: | :---: | :---: | :---: |
| LLM-native | ✅ | ✅ | ❌ | ✅ |
| Deterministic execution | ❌ | ❌ | ✅ | ✅ |
| Replayable traces | partial | minimal | ✅ | ✅ |
| Deterministic receipt | ❌ | ❌ | ❌ | ✅ |
| Suspend/resume | partial | partial | ✅ | ✅ |
| Runtime guardrails | ❌ | ❌ | partial | ✅ |
| LLM output enforcement | ❌ | ❌ | ❌ | ✅ |
| Pre-flight validator | ❌ | ❌ | ❌ | ✅ |
| Evaluator blindness | ❌ | ❌ | ❌ | ✅ |
| Lightweight / embedded | ❌ | ❌ | ❌ | ✅ |

**vs Marvin / DSPy:** those optimize *what* the LLM produces. nano-vm controls *when* and *whether* steps run — orthogonal concerns, composable.

**vs Temporal / Cadence:** Temporal guarantees *durable* execution — your workflow survives crashes, retries,
and distributed failures. nano-vm guarantees *governed* execution — every transition is policy-constrained,
every output is bounded, every run produces a verifiable receipt. Different guarantees, different buyer.
Use Temporal when you need a workflow to finish. Use nano-vm when you need to prove *how* it finished.

---

## When to Use

**Use nano-vm when:**

- workflow structure is known in advance
- correctness and auditability matter (fintech, compliance, enterprise)
- you need a reproducible trace for debugging or audit
- guardrails must be enforced at the system level, not in the prompt
- async orchestration with suspend/resume is required
- you need a deterministic receipt to prove what happened

**Do not use when:**

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
- [x] `Trace.trace_id` — UUID4, OTel-ready (v0.7.0)
- [x] `erase()` — GDPR tombstoning with hash-chain preservation (v0.7.0)
- [x] `ASTEngine` — `eval()` removed; sandboxed condition evaluator (v0.7.0)
- [x] Integration benchmark suite — 10/10 · 1,096,500 ops (v0.7.3)
- [x] `Step.is_terminal`, `Step.next_step` — branch semantics (v0.7.4)
- [x] ASTEngine METHOD_CALL guard — `ASTEvalError` at parse time (v0.7.5)
- [x] `py.typed` marker — PEP 561 (v0.7.4)
- [x] MCP server — `nano-vm-mcp` with GovernanceEnvelope, CapabilityRef, SSE + stdio
- [x] `Step.allowed_outputs` — LLM output validation against enum (v0.8.0)
- [x] `Step.timeout_seconds` + `on_timeout` — per-step LLM timeout (v0.8.0)
- [x] `inspect.iscoroutinefunction` — Python 3.14 deprecation fix (v0.8.2)
- [x] `TraceAnalyzer` — rollback density, tool churn rate, path variance, transition entropy, invariant violation rate (v0.8.3)
- [x] `LLMAdapter.complete()` — `str | tuple[str, dict | None]` Protocol (v0.8.4)
- [x] `ProgramValidator` — missing targets, unreachable steps, cycle detection, no_failure_terminal (v0.8.5)
- [x] `ExecutionReceipt` + `RejectedTransition` — `Receipt = f(Trace)` contract (v0.8.5)

**Upcoming:**

- [ ] OpenTelemetry span per FSM step
- [ ] `nano-vm-mcp`: `GovernedToolExecutor` circuit breaker
- [ ] `vm.step()` incremental execution endpoint
- [ ] `depends_on` + `TopologicalSorter` — declarative dependency graph

---

## Contact & Support

**Author:** [@ale007xd](https://t.me/ale007xd) on Telegram · [@ale007xd](https://x.com/ale007xd) on X

[![USDT (TON)](https://img.shields.io/badge/USDT%20(TON)-2ea2cc?style=flat-square)](https://tonviewer.com/UQCakyytrEGBikOi3eYMpveGHXDB1-fd6lcuQC9VvKqMrI-9)

**USDT (TON):** `UQCakyytrEGBikOi3eYMpveGHXDB1-fd6lcuQC9VvKqMrI-9`

---

## License

[MIT License](LICENCE).
