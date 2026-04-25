# nano-vm

**Deterministic VM for LLM program execution.**

Most LLM agents are unpredictable: the model decides what to do next on every step. nano-vm flips this — you define the program, the VM executes it deterministically. The LLM is a worker, not a decision-maker.

```
user_input → Planner (1 LLM call, non-deterministic)
           → Program (your DSL, deterministic)
           → ExecutionVM (deterministic)
           → Trace
```

## Install

```bash
pip install nano-vm          # core only (pydantic)
pip install nano-vm[litellm] # + LiteLLM adapter (all providers)
```

## Quick start

### Option A: Write the program yourself

```python
import asyncio
from nano_vm import ExecutionVM, Program
from nano_vm.adapters import LiteLLMAdapter

# A workflow where determinism is CRITICAL.
# An LLM agent might hallucinate and skip the guardrail.
# nano-vm guarantees the check ALWAYS runs before any action.
program = Program.from_dict({
    "name": "customer_refund",
    "steps": [
        {
            "id": "analyze",
            "type": "llm",
            "prompt": "Is this a valid refund request per our policy?\n"
                      "Request: $user_input\nPolicy: $refund_policy\n"
                      "Reply with 'yes' or 'no' followed by reason.",
            "output_key": "decision",
        },
        {
            "id": "guardrail",
            "type": "condition",
            # .lower() ensures case-insensitive match regardless of LLM output
            "condition": "'yes' in '$decision'.lower()",
            "then": "process_refund",
            "otherwise": "reject",
        },
        {
            # Only reached if guardrail passes — VM guarantees this
            "id": "process_refund",
            "type": "tool",
            "tool": "issue_refund",
            "args": {"reason": "$decision"},
        },
        {
            # Guaranteed path for invalid requests
            "id": "reject",
            "type": "llm",
            "prompt": "Politely explain the denial: $user_input",
        },
    ],
})

vm = ExecutionVM(
    llm=LiteLLMAdapter(
        model="groq/llama-3.3-70b-versatile",
        fallbacks=["openrouter/llama-3.3-70b-instruct:free"],
        temperature=0.0,  # deterministic output
    ),
    tools={"issue_refund": my_refund_fn},
)

async def main():
    trace = await vm.run(program, context={
        "user_input": "I want a refund for order #1234",
        "refund_policy": "Refunds allowed within 30 days of purchase.",
    })

    print(f"Status:  {trace.status}")
    print(f"Result:  {trace.final_output}")
    print(f"Tokens:  {trace.total_tokens()}")
    if trace.total_cost_usd() is not None:
        print(f"Cost:    ${trace.total_cost_usd():.6f}")

    # Full observability — every step logged
    for step in trace.steps:
        cost = f"  ${step.usage.cost_usd:.6f}" if step.usage and step.usage.cost_usd else ""
        tokens = f"  tokens={step.usage.total_tokens}" if step.usage else ""
        print(f"  [{step.step_id}] {step.status}  {step.duration_ms:.0f}ms{tokens}{cost}")

asyncio.run(main())
```

### Option B: Let the Planner generate the program

```python
from nano_vm import ExecutionVM, Planner
from nano_vm.adapters import LiteLLMAdapter

adapter = LiteLLMAdapter("groq/llama-3.3-70b-versatile", temperature=0.0)
planner = Planner(llm=adapter, tools=["search"])
vm = ExecutionVM(llm=adapter, tools={"search": my_search_fn})

async def main():
    # Planner makes ONE LLM call to create the program
    program = await planner.generate("Find latest AI news and summarize")
    # VM executes deterministically
    trace = await vm.run(program)
    print(trace.final_output)
```

## Program DSL

Programs are plain dicts or YAML — no framework lock-in.

```python
# llm — call the language model
{
    "id": "step_1",
    "type": "llm",
    "prompt": "Answer this: $user_input",  # $var resolved from context
    "output_key": "answer",               # save output to state
}

# tool — call a Python function (sync or async)
{
    "id": "step_2",
    "type": "tool",
    "tool": "search",
    "args": {"query": "$answer"},         # $step_id.output syntax
}

# condition — branch on a value
# Note: $variables resolve to strings; use .lower() for case-insensitive match
{
    "id": "step_3",
    "type": "condition",
    "condition": "'approved' in '$step_2.output'.lower()",
    "then": "step_4",
    "otherwise": "step_5",
}
```

### Variable resolution

| Syntax | Resolves to |
|--------|-------------|
| `$key` | `context["key"]` passed to `vm.run()` |
| `$step_id.output` | output of a previous step |

### Error handling per step

```python
{
    "id": "risky_step",
    "type": "tool",
    "tool": "external_api",
    "args": {},
    "on_error": "skip",   # fail (default) | skip | retry
    "max_retries": 3,
}
```

## Observability via Trace

Every run returns a full, structured execution trace:

```python
trace = await vm.run(program, context={...})

# Summary
print(trace.status)           # success | failed
print(trace.final_output)     # output of the last successful step
print(trace.duration_ms)      # total wall time
print(trace.total_tokens())   # sum of tokens across all llm steps
print(trace.total_cost_usd()) # sum of costs (None if provider doesn't report)

# Per-step breakdown
for step in trace.steps:
    print(step.step_id, step.status, step.duration_ms)
    if step.usage:
        # Only present for llm steps
        print(f"  tokens: {step.usage.total_tokens}")
        print(f"  cost:   ${step.usage.cost_usd:.6f}")
```

## Bring your own adapter

No litellm? Implement one method:

```python
class MyAdapter:
    async def complete(self, messages: list[dict], **kwargs) -> str:
        # your HTTP client here
        return "response text"

vm = ExecutionVM(llm=MyAdapter())
```

## Providers via LiteLLM

```python
LiteLLMAdapter("groq/llama-3.3-70b-versatile")
LiteLLMAdapter("anthropic/claude-sonnet-4-20250514")
LiteLLMAdapter("openrouter/llama-3.3-70b-instruct:free")
LiteLLMAdapter("ollama/llama3")  # local

# With automatic fallback chain
LiteLLMAdapter(
    model="groq/llama-3.3-70b-versatile",
    fallbacks=["openrouter/llama-3.3-70b-instruct:free", "anthropic/claude-sonnet-4-20250514"],
)
```

## Why not just use an LLM agent?

| | LLM Agent | nano-vm |
|---|---|---|
| Who decides next step | LLM (every call) | You (program definition) |
| Reproducibility | varies | same input → same path |
| Guardrails | best-effort | structurally enforced |
| Debuggability | hard | full Trace per step |
| Cost visibility | none | tokens + cost per step |
| LLM calls per run | many | 1 per llm-step |

Use nano-vm when you know the workflow and want guaranteed execution.  
Use an open agent when the workflow itself is unknown.

## License

MIT
