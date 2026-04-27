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
  <strong>When you need to <em>prove</em> what happened — and why.</strong><br>
  llm-nano-vm is a deterministic execution engine for LLM workflows.<br>
  Guardrails enforced by the VM, not by the prompt.
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

## Install

```bash
pip install llm-nano-vm
pip install llm-nano-vm[litellm]   # for built-in provider support
```

---

## Quick Start — Guardrail That Never Skips

```python
from nano_vm import ExecutionVM, Program
from nano_vm.adapters.litellm_adapter import LiteLLMAdapter

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

adapter = LiteLLMAdapter("openai/gpt-4o-mini")
vm = ExecutionVM(adapter=adapter, tools={"issue_refund": ..., "send_rejection": ...})
trace = await vm.run(program, context={"user_input": "I was charged twice"})

print(trace.status)           # SUCCESS
print(trace.final_output)     # tool result
print(trace.total_cost_usd()) # e.g. 0.000034
```

The `guardrail` step cannot be skipped, reordered, or overridden by the model.
That is the guarantee.

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
  → Program (DSL — JSON/dict)
  → ExecutionVM (deterministic FSM)
  → Trace (status · cost · tokens · duration)
```

---

## Program DSL

Three step types:

| Type | Purpose |
| :--- | :--- |
| `llm` | call the model; result stored in `output_key` |
| `tool` | call a Python function |
| `condition` | branch on an expression; `then` / `otherwise` |

### Variable interpolation

| Syntax | Resolves to |
| :--- | :--- |
| `$key` | value from initial context |
| `$step_id.output` | output of a previous step |

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

---

## Observability

```python
trace.status            # TraceStatus.SUCCESS | FAILED
trace.final_output      # last step output
trace.total_tokens()    # sum across all steps
trace.total_cost_usd()  # sum across all steps (requires LiteLLMAdapter)

for step in trace.steps:
    print(step.step_id, step.status, step.duration, step.usage)
```

---

## Planner (Optional)

```python
from nano_vm import Planner

planner = Planner(adapter=adapter)
program = await planner.generate("Fetch latest AI news, summarize, classify by topic")
trace = await vm.run(program)
```

- exactly **1** LLM call
- outputs a validated Program object
- non-deterministic input → deterministic execution

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

| Metric | Value |
| :--- | :--- |
| VM throughput (no LLM) | ~535 programs/sec |
| VM latency per step | ~1.80 ms |
| Test coverage | 33/33 tests passing |

> **Note:** throughput is measured with a mock adapter (no network).
> Real end-to-end latency is dominated by LLM API response time (typically 0.5–5 s/step).

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

| | LangChain | AutoGPT / CrewAI | **llm-nano-vm** |
| :--- | :--- | :--- | :--- |
| Layer | orchestration | reasoning / autonomy | execution guarantees |
| Execution order | flexible | model-driven | enforced |
| Guardrails | prompt-level | prompt-level | VM-level |
| Trace | partial | minimal | full, per-step |
| Best for | flexible pipelines | autonomous tasks | compliance-grade workflows |

---

## Roadmap

- [x] FSM execution engine (v0.1)
- [x] `llm / tool / condition` step types
- [x] LiteLLM adapter + cost tracking
- [x] Published to PyPI as `llm-nano-vm`
- [ ] `parallel` steps — `asyncio.gather` for independent tool calls (v0.4)
- [ ] MCP server — `run_program`, `get_trace`, `list_programs` (nano-vm-mcp)
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
