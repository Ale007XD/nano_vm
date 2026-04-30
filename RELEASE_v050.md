# llm-nano-vm v0.5.0

**Planner GA · Benchmark suite BM1–BM11 · Three benchmark infrastructure fixes**

---

## What's new

### Planner — production-ready

`Planner` converts a natural-language intent into a validated `Program` in exactly one LLM call.

```python
from nano_vm import Planner

planner = Planner(llm=adapter, max_retries=2, temperature=0.0)
program = await planner.generate(
    "Fetch latest AI news, summarize, classify by topic",
    available_tools=["fetch_rss", "summarize", "classify"],
)
trace = await vm.run(program)
```

Signature is stable. All parameters keyword-optional.

---

### Benchmark suite v0.5.0

`benchmarks/benchmark_v050.py` — BM1–BM11:

| # | Scenario | Adapter |
| :-- | :--- | :--- |
| BM1 | Retry baseline (0 retries / 2 retries) | Mock |
| BM2 | Condition branching | Mock |
| BM3 | Parallel 10 sub-steps | Mock |
| BM4 | Parallel with concurrency cap | Mock |
| BM5 | `max_steps` budget overhead | Mock |
| BM6 | `max_stalled_steps` budget | Mock |
| BM7 | `max_tokens` budget overhead | Mock |
| BM8 | Real multi-model calls (OpenRouter free tier) | LiteLLM |
| BM9 | State fingerprint uniqueness | Mock |
| BM10 | Cost tracking across steps | Mock |
| BM11 | Planner determinism | Mock |

**BM11 result:** `✓ YES deterministic, unique fingerprints=1` — confirmed on VPS.  
**BM8 status:** blocked by rate limit (429) during peak hours — real latency numbers coming in patch release.

---

## Bug fixes

### `@dataclass` crash under `importlib.util` load

`run_all.py` now registers `sys.modules[name] = mod` **before** `spec.loader.exec_module(mod)`.

Root cause: `@dataclass` calls `sys.modules.get(cls.__module__).__dict__` at decoration time.
Without prior registration → `None` → `AttributeError`. Affected `benchmark_v050` and `benchmark_stress`.

### `sys.exit(1)` in import guard killed the runner process

`benchmark_v050.py` import guard changed from `sys.exit(1)` to `raise ImportError(...)`.
`run_all.py`'s `try/except Exception` catches `ImportError` but not `SystemExit`.

### Stale OpenRouter free-tier models

`mistral-7b-instruct:free` and `deepseek-chat-v3-0324:free` both return 404 — removed from OpenRouter.
Replaced with models verified live via `GET /api/v1/models`:
- `meta-llama/llama-3.3-70b-instruct:free`
- `google/gemma-3-27b-it:free`

---

## Known issues

**BM8 rate limit (429)** — `llama-3.3-70b` and `gemma-3-27b` free-tier pool saturates during daytime (UTC+8).
Will run off-peak and publish numbers. Workaround candidates: `qwen/qwen3-coder:free`, `nvidia/nemotron-nano-9b-v2:free`.

---

## Install

```bash
pip install llm-nano-vm==0.5.0
pip install llm-nano-vm[litellm]==0.5.0   # for LiteLLMAdapter
```

---

## Files changed (benchmark infrastructure)

| File | Change |
| :--- | :--- |
| `benchmarks/run_all.py` | `sys.modules` fix ×3, per-call `✓/✗ + error` logging |
| `benchmarks/benchmark_v050.py` | import guard `sys.exit → raise ImportError`, MODELS updated |
| `nano_vm/planner.py` | no changes — verified correct |

---

**Full changelog:** [CHANGELOG.md](CHANGELOG.md)
