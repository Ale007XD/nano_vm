<p align="center">
  <a href="https://github.com/Ale007XD/nano-vm/actions/workflows/ci.yml">
    <img src="https://github.com/Ale007XD/nano-vm/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/nano-vm/">
    <img src="https://img.shields.io/pypi/v/nano-vm?color=blue" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/nano-vm/">
    <img src="https://img.shields.io/pypi/pyversions/nano-vm" alt="Python versions">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/github/license/Ale007XD/nano-vm" alt="License">
  </a>
</p><p align="center">
  🧠 <strong>Deterministic VM for LLM program execution.</strong><br>
  Turn unpredictable LLM behavior into structured, reproducible workflows.
</p>---

🤝 Contact & Support

Author: "@ale007xd" (https://t.me/ale007xd) | "@ale007xd" (https://x.com/ale007xd)

Support the project:

""Buy Me a Coffee" (https://img.shields.io/badge/☕-Buy%20Me%20a%20Coffee-yellow?style=flat-square)" (https://www.buymeacoffee.com/ale007xd)
""TON USDT" (https://img.shields.io/badge/USDT%20(TON)-UQCakyytrEGBikOi3eYMpveGHXDB1--fd6lcuQC9VvKqMrI--9-2ea2cc?style=flat-square&logo=ton)" (https://tonviewer.com/UQCakyytrEGBikOi3eYMpveGHXDB1-fd6lcuQC9VvKqMrI-9)

---

nano-vm

🧠 Mental Model

- LLM → stateless worker
- Program → declarative workflow (DSL)
- ExecutionVM → deterministic state machine
- Trace → full execution log

«nano-vm is essentially a finite state machine for LLM workflows.»

---

⚠️ The Problem

LLM agents are unpredictable:

- they decide what to do next
- they may skip checks
- behavior changes across runs

---

✅ The Solution

user_input → Planner (1 LLM call)
           → Program (DSL)
           → ExecutionVM (deterministic)
           → Trace

- Planner = flexible but non-deterministic
- VM = strict and deterministic

---

🚀 Install

pip install nano-vm
pip install nano-vm[litellm]

---

⚡ Quick Example (Deterministic Guardrail)

program = Program.from_dict({
    "name": "customer_refund",
    "steps": [
        {
            "id": "analyze",
            "type": "llm",
            "prompt": "...",
            "output_key": "decision",
        },
        {
            "id": "guardrail",
            "type": "condition",
            "condition": "'yes' in '$decision'.lower()",
            "then": "process_refund",
            "otherwise": "reject",
        },
    ],
})

👉 VM guarantees:

- guardrail ALWAYS runs
- no skipped steps
- deterministic execution path

---

🤖 Planner (Optional)

program = await planner.generate("Find latest AI news")

- 1 LLM call
- outputs DSL program
- NOT deterministic

---

📜 Program DSL

{
  "id": "step",
  "type": "llm" | "tool" | "condition"
}

Variables

Syntax| Meaning
"$key"| input context
"$step.output"| previous step result

---

🔍 Observability (Trace)

trace = await vm.run(program)

trace.status
trace.total_tokens()
trace.total_cost_usd()

Every step includes:

- duration
- tokens
- cost
- status

👉 Full debugging without guesswork.

---

⚖️ nano-vm vs Agents

| LLM Agent| nano-vm
Control| LLM decides| You define
Determinism| ❌| ✅
Debugging| hard| full trace
Guardrails| weak| enforced

---

❌ When NOT to use nano-vm

Do NOT use if:

- workflow is unknown
- task is creative/open-ended
- you want autonomous reasoning

Use it when:

- flow is known
- correctness matters
- reproducibility is required

---

🔌 Custom Adapter

class MyAdapter:
    async def complete(self, messages, **kwargs):
        return "response"

---

📡 Providers (LiteLLM)

LiteLLMAdapter("groq/llama-3.3-70b-versatile")
LiteLLMAdapter("ollama/llama3")

---

📄 License

MIT
