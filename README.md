<p align="center">
  <a href="https://github.com/Ale007XD/nano_vm/actions">
    <img src="https://github.com/Ale007XD/nano_vm/workflows/CI/badge.svg" alt="CI">
  </a>
  <img src="https://img.shields.io/badge/pypi-coming--soon-lightgrey" alt="PyPI">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p><p align="center">
  🧠 <strong>Deterministic VM for LLM program execution.</strong><br>
  Turn unpredictable LLM behavior into structured, reproducible workflows.
</p>---

🤝 Contact & Support

Author: "@ale007xd" (https://t.me/ale007xd) | "@ale007xd" (https://x.com/ale007xd)

☕ Support the project

""Buy Me a Coffee" (https://img.shields.io/badge/☕-Buy%20Me%20a%20Coffee-yellow?style=flat-square)" (https://www.buymeacoffee.com/ale007xd)

""USDT (TON)" (https://img.shields.io/badge/USDT%20(TON)-UQCakyytrEGBikOi3eYMpveGHXDB1--fd6lcuQC9VvKqMrI--9-2ea2cc?style=flat-square&logo=ton)" (https://tonviewer.com/UQCakyytrEGBikOi3eYMpveGHXDB1-fd6lcuQC9VvKqMrI-9)

Direct wallet:

USDT (TON):
UQCakyytrEGBikOi3eYMpveGHXDB1-fd6lcuQC9VvKqMrI-9

---

nano-vm

🧠 Mental Model

- LLM → stateless worker
- Program → declarative workflow (DSL)
- ExecutionVM → deterministic state machine
- Trace → full execution log

«nano-vm is a finite state machine (FSM) for LLM workflows.»

---

⚠️ The Problem

LLM agents are unpredictable:

- decide next steps dynamically
- may skip critical checks
- behavior varies between runs

---

✅ The Solution

user_input → Planner (1 LLM call, optional)
           → Program (DSL)
           → ExecutionVM (deterministic)
           → Trace

- Planner = flexible, non-deterministic
- VM = strict, deterministic

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
            "prompt": "Is this a valid refund request?\nRequest: $user_input",
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

👉 Guarantees:

- guardrail ALWAYS runs
- no skipped steps
- deterministic execution path

---

🤖 Planner (Optional)

program = await planner.generate("Find latest AI news and summarize")

- exactly 1 LLM call
- outputs DSL program
- not deterministic

---

📜 Program DSL

{
  "id": "step_1",
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
trace.final_output
trace.total_tokens()
trace.total_cost_usd()

Each step includes:

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
- task is creative / open-ended
- you need autonomous reasoning

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
LiteLLMAdapter("openrouter/llama-3.3-70b-instruct:free")
LiteLLMAdapter("ollama/llama3")

---

📄 License

MIT
