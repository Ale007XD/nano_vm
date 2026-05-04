# Changelog

All notable changes to `llm-nano-vm` will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.6.0] — 2025-05-04

### Added

- **FSM invariant stress suite (`benchmarks/benchmark_nano_vm.py`).**  
  13 tests (BM-01–BM-12 + BM-VM) validating δ(S, E) → S' under chaos, injection, replay,
  and concurrent load. Array size: 10,000 per test · 5 runs · seed=42.  
  Result: **13/13 PASSED · 1,020,000 total operations · 0 invariant violations.**

- **`## Execution Pipeline` section in README.**  
  Canonical formal model:
  ```
  E  = LLM(input)     →  raw event (probabilistic, untrusted)
  E' = Validator(E)   →  validated context (deterministic)
  A(S) = FSM(S, E')   →  allowed actions (deterministic)
  a*   = Policy(A, C) →  selected action (deterministic pure fn)
  S'   = δ(S, a*)     →  next state (deterministic)
  ```
  Includes layer responsibility table and `Implementation Note` on current
  single-action `A(S)` vs future multi-candidate Policy.

### Benchmark results (BM-01–BM-12 + BM-VM, Linux · x86_64 · Python 3.12 · venv)

| Tag | Test | Mean ms | Throughput /s | Result |
| :--- | :--- | ---: | ---: | :--- |
| BM-01 | Idempotency Under Replay Stress | 279 | 35,794 | 450k replays · **0 violations** |
| BM-02 | Duplicate Execution Attack | 222 | 45,114 | 50k triggers · **0 double exec** |
| BM-03 | Crash Mid-Step Recovery | 170 | 58,741 | **0 wrong resumes** |
| BM-04 | Non-Deterministic LLM Injection | 68 | 148,018 | 13 noise variants · **0 FSM influence** |
| BM-05 | Tool Failure Cascade A→B→C | 135 | 73,847 | **0 cascade violations** |
| BM-06 | Long-Running Tool + Timeout Drift | 73 | 137,531 | 66.8% timeout · **0 partial transitions** |
| BM-07 | Out-of-Order Event Delivery | 123 | 81,234 | **0 invalid sequences accepted** |
| BM-08 | State Explosion / Memory Pressure | 486 | 20,567 | **StateContext bounded \|S\|=12** |
| BM-09 | Partial StepResult Corruption | 66 | 151,479 | 8 types · **50k/50k normalized** |
| BM-10 | Transition Validity Invariant | 123 | 81,068 | 90.5% blocked · **0 mutations** |
| BM-11 | Reentrancy Stress | 175 | 57,187 | **0 double mutations** |
| BM-12 | Chaos Mode — Full System Stress | 2352 | 4,252 | 83k escalations · **0 invalid final states** |
| BM-VM | nano-vm Double Execution Safety | 53 | 190,428 | 300 real `vm.run` · **0 double exec** |

> **BM-04:** I1 confirmed empirically — LLM noise ("оплата", "¿pagar?", "✓", "affirmative"…)
> produces zero influence on δ(S, E).  
> **BM-12:** mathematical consequence of δ having no path to an invalid state, not a probabilistic result.  
> **BM-VM:** `I_k(T) ∈ {0,1}` trace invariant holds across 300 real `ExecutionVM` runs.

### Known issues

- **BM8 still blocked by rate limit (429) during peak hours** (inherited from v0.5.0).  
  Workaround candidates: `qwen/qwen3-coder:free`, `nvidia/nemotron-nano-9b-v2:free`.

---

## [0.5.0] — 2025-04-30

### Added

- **`Planner` — full implementation.**  
  `Planner(llm, max_retries=2, temperature=0.0).generate(intent, available_tools?, context_keys?)` converts
  a natural-language intent into a validated `Program` in exactly one LLM call.  
  Signature is stable; all parameters are keyword-optional.

- **Benchmark suite v0.5.0 (`benchmarks/benchmark_v050.py`).**  
  BM1–BM11 covering retry baseline, budget guards, token tracking, state fingerprints, parallel concurrency,
  Planner determinism (BM11), and real OpenRouter multi-model calls (BM8).

- **`benchmarks/run_all.py` — unified benchmark runner.**  
  Loads `benchmark_v040`, `benchmark_v050`, and `benchmark_stress` via `importlib.util`; prints
  per-call `✓ / ✗ + error` inline; exits with non-zero status only on hard failures.

### Fixed

- **`@dataclass` crash under `importlib.util` load** (`run_all.py`).  
  `sys.modules[name] = mod` is now registered *before* `spec.loader.exec_module(mod)`.  
  Root cause: `@dataclass` calls `sys.modules.get(cls.__module__).__dict__` at decoration time;
  without prior registration the lookup returns `None` → `AttributeError`.  
  Affected modules: `benchmark_v050`, `benchmark_stress` (contain `@dataclass`);
  `benchmark_v040` was unaffected (functions only).

- **`sys.exit(1)` in import guard killed the `run_all.py` process** (`benchmark_v050.py`).  
  Changed to `raise ImportError(...)`. The enclosing `try/except Exception` in `run_all.py`
  catches `ImportError` but not `SystemExit`.

- **Stale OpenRouter free-tier models** (`benchmark_v050.py`).  
  `mistralai/mistral-7b-instruct:free` (404) and `deepseek/deepseek-chat-v3-0324:free` (404)
  replaced with live models verified via `GET /api/v1/models`:
  - `meta-llama/llama-3.3-70b-instruct:free`
  - `google/gemma-3-27b-it:free`

- **BM8 error output was silent.**  
  `CallResult.error` was populated but never printed. Added per-call `✓ / ✗ + r.error` logging
  inside `run_suite_real`.

### Known issues

- **BM8 blocked by rate limit (429) during peak hours.**  
  Both `llama-3.3-70b-instruct:free` and `gemma-3-27b-it:free` share a free-tier pool on OpenRouter
  and return `429` during daytime (approx. 07:00–23:00 UTC+8).  
  BM8 real-latency numbers will be published in a follow-up patch after off-peak run.  
  Workaround candidates: `qwen/qwen3-coder:free`, `nvidia/nemotron-nano-9b-v2:free`.

---

## [0.4.0] — 2025-03-XX

### Added

- `max_steps` budget — `BUDGET_EXCEEDED` after N total steps executed.
- `max_stalled_steps` budget — `STALLED` after N consecutive no-op state fingerprints.
- `max_tokens` budget — `BUDGET_EXCEEDED` when cumulative token count exceeds limit.
- `state_snapshots` in `Trace` — `list[(step_index, sha256_hex)]`, one entry per executed step.
- `on_error`, `max_retries` per-step options.
- `max_concurrency` for `parallel` blocks.

### Performance (BM5–BM7, Mock adapter, 2-core VPS)

| Benchmark | Baseline | With budget | Delta |
| :--- | :--- | :--- | :--- |
| BM5 `max_steps` | 558 RPS / 1.793 ms | 616 RPS / 1.623 ms | ±9.5% (noise) |
| BM7 `max_tokens` | 458 RPS / 2.184 ms | 420 RPS / 2.379 ms | +8.9% (O(N²) per step) |

---

## [0.3.0] — 2025-02-XX

### Added

- `max_concurrency` — cap concurrent sub-steps per `parallel` block.
- `retry` policy per step — exponential backoff: 1 s, 2 s, 4 s … cap 30 s.

### Performance (BM1, Mock adapter, 2-core VPS)

| Scenario | Throughput | Latency |
| :--- | :--- | :--- |
| 0 retries | 3 509 RPS | 0.285 ms |
| 2 retries | 4 308 RPS | 0.232 ms |

No regression vs v0.2.0 — `max_concurrency` / `retry` adds zero overhead when not triggered.

---

## [0.2.0] — 2025-01-XX

### Added

- `parallel` step type — `asyncio.gather` for independent sub-steps.
- `MockLLMAdapter` — deterministic testing without API keys (sequence or prompt-map mode).

---

## [0.1.0] — 2025-01-XX

### Added

- FSM execution engine (`ExecutionVM`).
- `llm`, `tool`, `condition` step types.
- `LiteLLMAdapter` + cost tracking.
- `Trace` — full per-step log: status, cost, tokens, duration.
- Published to PyPI as `llm-nano-vm`.
