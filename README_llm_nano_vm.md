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

## FSM Transition Table

`ExecutionVM` is a finite state machine. The full δ-function:

| Current state | Step type | Outcome | Next state |
| :--- | :--- | :--- | :--- |
| `RUNNING` | `llm` | success | `RUNNING` (advance to next step) |
| `RUNNING` | `llm` | all retries exhausted | `FAILED` |
| `RUNNING` | `tool` | success | `RUNNING` |
| `RUNNING` | `tool` | returns `"PENDING"` sentinel | `SUSPENDED` |
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
| `FAILED` / `SUCCESS` / `BUDGET_EXCEEDED` / `STALLED` | — | — | terminal |

Terminal states are absorbing — once reached, no further step is executed.
`SUSPENDED` is resumable — cursor is persisted; execution continues from the suspended step.

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

print(trace.status)           # SUCCESS
print(trace.final_output)     # tool result
print(trace.total_cost_usd()) # e.g. 0.000034
```

The `guardrail` step cannot be skipped, reordered, or overridden by the model.

---

## suspend / resume via Webhook (v0.6.0)

For async workflows — payment confirmations, courier events, external approvals:

```python
from nano_vm.vm import ExecutionVM, InMemoryCursorRepository

# Tool signals async wait via "PENDING" sentinel
async def initiate_payment(order_id: str) -> str:
    await register_webhook_handler(order_id)
    return "PENDING"   # VM suspends here, persists cursor

vm = ExecutionVM(
    llm=adapter,
    cursor_repo=InMemoryCursorRepository(),  # use SqliteCursorRepository in production
    tools={"initiate_payment": initiate_payment, ...},
)

trace = await vm.run(program, context={"order_id": "123"})
assert trace.status == TraceStatus.SUSPENDED

# When webhook fires:
trace = await vm.resume_with_program(
    program=program,
    trace_id=trace.trace_id,
    webhook_event={"type": "payment.confirmed", "order_id": "123"},
)
assert trace.status == TraceStatus.SUCCESS
```

`InMemoryCursorRepository` — tests and dry-run only.
Production: implement `CursorRepository` Protocol backed by `infrastructure.db` (SQLite WAL).

---

## BudgetInterrupt (v0.6.0)

Budget exhaustion is a **system interrupt**, not a control-flow condition.
The LLM cannot observe or influence it.

```python
from nano_vm.vm import ExecutionVM, InterruptType

class InstrumentedVM(ExecutionVM):
    async def _emit_interrupt(self, interrupt_type: InterruptType) -> None:
        await notify_operator(f"interrupt: {interrupt_type.value}")

vm = InstrumentedVM(llm=adapter)
```

Override `_emit_interrupt()` via subclass (standard inheritance, no magic).
Base implementation is a no-op hook — documented, not silent.

---

## How the DSL Controls Agent Behavior

```
LLM decides:  WHAT to say, how to reason, what content to produce
DSL decides:  WHICH step runs next, WHEN to branch, WHEN to stop
```

The LLM has **no knowledge** of the program structure.
It receives a prompt and returns a string — nothing more.

| | LLM | DSL (VM) |
| :--- | :--- | :--- |
| Produce content | ✅ free | — |
| Skip a step | ❌ impossible | enforces every step |
| Reorder steps | ❌ impossible | order fixed at definition |
| Branch on output | ❌ cannot | `condition` step evaluates |
| Decide workflow is done | ❌ impossible | VM controls termination |

---

## Program DSL

Four step types:

| Type | Purpose |
| :--- | :--- |
| `llm` | call the model; result stored in `output_key` |
| `tool` | call a Python function; return `"PENDING"` to suspend |
| `condition` | branch on an expression; `then` / `otherwise` |
| `parallel` | run independent sub-steps concurrently via `asyncio.gather` |

**Step options:**

| Option | Default | Description |
| :--- | :--- | :--- |
| `on_error` | `fail` | `fail` · `skip` · `retry` |
| `max_retries` | `3` | total attempts; exponential backoff: 1s, 2s, 4s… cap 30s |
| `max_concurrency` | `None` | parallel blocks only; `None` = no cap |

**Program budget options:**

| Option | Default | Description |
| :--- | :--- | :--- |
| `max_steps` | `None` | `BUDGET_EXCEEDED` if exceeded |
| `max_stalled_steps` | `None` | `STALLED` after N consecutive no-op steps |
| `max_tokens` | `None` | `BUDGET_EXCEEDED` when total tokens ≥ limit; O(1) per step |

### Variable interpolation

| Syntax | Resolves to |
| :--- | :--- |
| `$key` | value from initial context |
| `$step_id.output` | output of a previous step |

> **⚠ Security note — condition expressions:**  
> `condition` strings are evaluated via `eval()` with `__builtins__` cleared.
> This is a partial sandbox. **Do not interpolate raw user input into condition
> expressions.** LLM output used as a branching signal should only appear in
> context variables that your condition *tests* (`'yes' in '$decision'`),
> never as the condition expression itself.  
> **Numeric context variables are injected directly** — no string coercion
> needed for comparisons like `$value > 0.9`.

---

## Testing — Deterministic by Design

```python
from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.adapters import MockLLMAdapter

vm = ExecutionVM(llm=MockLLMAdapter({"Classify": "SAFE", "__default__": "ok"}))

trace = await vm.run(program, context={"user_input": "refund"})
assert trace.status == TraceStatus.SUCCESS
assert [s.step_id for s in trace.steps] == ["classify", "route", "verify_eligibility", ...]
```

Same input → same step sequence. Always. No API key required.

---

## Observability

```python
trace.status                # SUCCESS | FAILED | BUDGET_EXCEEDED | STALLED | SUSPENDED
trace.trace_id              # UUID4 — stable for OTel propagation (v0.6.0)
trace.final_output
trace.total_tokens()        # O(1) — incremental accumulator
trace.total_cost_usd()
trace.state_snapshots       # list[(step_index, sha256_hex)]
trace.error

for step in trace.steps:
    print(step.step_id, step.status, step.duration_ms, step.usage)
```

---

## Performance

VM overhead is near-zero. Bottleneck in production: LLM API latency and DB I/O.

### v0.6.0 — Stress test: 10 000 FSM graphs × 5 runs

```
System: Linux · x86_64 (2 cores) · Python 3.12
Test:   10 000 items × 5 deterministic runs, concurrency=200, Mock adapter

  Run 1:  0.70 s  14 286 it/s   8973 OK / 1027 ERR
  Run 2:  0.70 s  14 286 it/s   8973 OK / 1027 ERR
  Run 3:  0.69 s  14 493 it/s   8973 OK / 1027 ERR
  Run 4:  0.70 s  14 286 it/s   8973 OK / 1027 ERR
  Run 5:  0.70 s  14 286 it/s   8973 OK / 1027 ERR
  ─────────────────────────────────────────────────
  AVG:    0.70 s  14 327 it/s

  Determinism:       ✅ identical results across all 5 runs
  Failure isolation: ✅ VMError caught per-coroutine, event loop unaffected
  Error rate:        10.27% matches P(value > 0.9) = 0.1 exactly
```

### v0.5.0 — Double-execution safety

```
  Raw stateless agent:   ~20% double-executions / 1000 runs
  FSM runtime (vm.run):  0 double-executions / 3000 runs
```

### v0.4.0 — Budget mechanism overhead

```
  BM5  max_steps=1000   ±9.5%  (within noise — single int check)
  BM7  max_tokens       fixed in v0.5.0: O(1) via _token_accumulator
```

### v0.3.0 — 20 parallel steps via OpenRouter

```
  Total: 1.7574 s · 20 steps · 11.38 steps/sec · VM overhead ~1.80 ms/step
```

---

## Planner (Optional)

```python
from nano_vm import Planner

planner = Planner(llm=adapter, max_retries=2, temperature=0.0)
program = await planner.generate(
    "Fetch latest AI news, summarize, classify by topic",
    available_tools=["fetch_rss", "summarize", "classify"],
)
trace = await vm.run(program)
```

Exactly 1 LLM call. Outputs a validated `Program`. Determinism confirmed (BM11).

---

## Comparison

| | LangChain | AutoGPT / CrewAI | Prefect / Airflow | **llm-nano-vm** |
| :--- | :--- | :--- | :--- | :--- |
| Execution order | flexible | model-driven | enforced | enforced |
| Guardrails | prompt-level | prompt-level | task-level | VM-level |
| Async suspend/resume | ❌ | ❌ | native | ✅ v0.6.0 |
| Parallel execution | manual | model-driven | native | scoped, deterministic |
| Trace | partial | minimal | job logs | full, per-step + sub-step |
| Overhead | heavy | heavy | heavy | near-zero |
| Best for | flexible pipelines | autonomous tasks | data/ETL | compliance-grade LLM workflows |

---

## When to Use

**Use llm-nano-vm when:**
- workflow structure is known in advance
- correctness and auditability matter (fintech, compliance, enterprise)
- you need async suspend/resume for webhook-driven flows
- you want guardrails enforced at the system level, not in the prompt

**Do NOT use when:**
- workflow is unknown and must be discovered at runtime
- task is open-ended creative reasoning
- you need fully autonomous multi-agent coordination

---

## Roadmap

- [x] FSM execution engine (v0.1)
- [x] `llm / tool / condition` step types
- [x] LiteLLM adapter + cost tracking
- [x] `parallel` steps — `asyncio.gather` (v0.2.0)
- [x] `MockLLMAdapter` — deterministic testing (v0.2.0)
- [x] `max_concurrency` + `retry` policy per sub-step (v0.3.0)
- [x] `max_steps` / `max_stalled_steps` / `max_tokens` budget (v0.4.0)
- [x] `state_snapshots` — sha256 per step (v0.4.0)
- [x] `Planner` — intent → Program in 1 call (v0.5.0)
- [x] `total_tokens()` O(1) via `_token_accumulator` (v0.5.0)
- [x] Double-execution safety: 0/3000 FSM vs ~20% stateless (v0.5.0)
- [x] `suspend` / `resume_with_program()` via `"PENDING"` sentinel (v0.6.0)
- [x] `BudgetInterrupt` — isolated signal, `_emit_interrupt()` hook (v0.6.0)
- [x] `VaultStepResult` + `VaultStepMetadata` — MCP-compatible contracts (v0.6.0)
- [x] `Trace.trace_id` UUID4 — OTel propagation (v0.6.0)
- [x] MCP server — `run_program`, `get_trace`, SQLite WAL, SSE + Bearer auth ([nano-vm-mcp](https://github.com/Ale007XD/nano-vm-mcp))
- [ ] `SqliteCursorRepository` — production `CursorRepository` implementation
- [ ] `resume()` — Blueprint registry lookup (P8 of nano-vm-vault)
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

[![Buy Me a Coffee](https://img.shields.io/badge/☕-Buy%20Me%20a%20Coffee-yellow?style=flat-square)](https://www.buymeacoffee.com/ale007xd)
[![USDT (TON)](https://img.shields.io/badge/USDT%20(TON)-2ea2cc?style=flat-square&logo=ton)](https://tonviewer.com/UQCakyytrEGBikOi3eYMpveGHXDB1-fd6lcuQC9VvKqMrI-9)

```
UQCakyytrEGBikOi3eYMpveGHXDB1-fd6lcuQC9VvKqMrI-9
```

---

## License

[MIT](LICENCE)
