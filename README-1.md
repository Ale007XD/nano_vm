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

program = Program.from_dict({
    "name": "summarize_and_translate",
    "steps": [
        {
            "id": "summarize",
            "type": "llm",
            "prompt": "Summarize in 2 sentences: $text",
            "output_key": "summary",
        },
        {
            "id": "translate",
            "type": "llm",
            "prompt": "Translate to English: $summary",
        },
    ],
})

vm = ExecutionVM(
    llm=LiteLLMAdapter(
        model="groq/llama-3.3-70b-versatile",
        fallbacks=["openrouter/llama-3.3-70b-instruct:free"],
        temperature=0.0,  # deterministic output
    )
)

async def main():
    trace = await vm.run(program, context={"text": "Your long text here..."})
    print(trace.final_output)
    print(f"Done in {trace.duration_ms:.0f}ms")

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
    program = await planner.generate("Find latest AI news and summarize")
    trace = await vm.run(program, context={})
    print(trace.final_output)
```

## Program DSL

Programs are plain dicts or YAML. Three step types:

```python
# llm — call the language model
{
    "id": "step_1",
    "type": "llm",
    "prompt": "Answer this: $user_input",  # $var resolves from context
    "output_key": "answer",               # save output to state
}

# tool — call a Python function
{
    "id": "step_2",
    "type": "tool",
    "tool": "search",
    "args": {"query": "$answer"},         # $step_id.output syntax
}

# condition — branch on a value
{
    "id": "step_3",
    "type": "condition",
    "condition": "'yes' in '$step_2.output'",
    "then": "step_4",
    "otherwise": "step_5",
}
```

### Variable resolution

| Syntax | Resolves to |
|--------|-------------|
| `$key` | `context["key"]` |
| `$step_id.output` | output of a previous step |

### Error handling per step

```python
{
    "id": "risky_step",
    "type": "tool",
    "tool": "external_api",
    "args": {},
    "on_error": "skip",   # fail | skip | retry
    "max_retries": 3,
}
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
# Groq
LiteLLMAdapter("groq/llama-3.3-70b-versatile")

# Anthropic
LiteLLMAdapter("anthropic/claude-sonnet-4-20250514")

# OpenRouter
LiteLLMAdapter("openrouter/llama-3.3-70b-instruct:free")

# Ollama (local)
LiteLLMAdapter("ollama/llama3")

# With automatic fallback chain
LiteLLMAdapter(
    model="groq/llama-3.3-70b-versatile",
    fallbacks=["openrouter/llama-3.3-70b-instruct:free", "anthropic/claude-sonnet-4-20250514"],
)
```

## Trace

Every run returns a full execution trace:

```python
trace.status        # success | failed | running
trace.final_output  # output of the last successful step
trace.duration_ms   # total execution time

for step in trace.steps:
    print(step.step_id, step.status, step.output, step.duration_ms)
```

## Why not just use an LLM agent?

| | LLM Agent | nano-vm |
|---|---|---|
| Who decides next step | LLM (every time) | You (program definition) |
| Reproducibility | ❌ varies | ✅ same input → same path |
| Debuggability | hard | full Trace |
| Cost | many LLM calls | 1 call per llm-step |
| Flexibility | high | medium |

Use nano-vm when you know the workflow and want guaranteed execution. Use an open agent when the workflow itself is unknown.

## License

MIT
