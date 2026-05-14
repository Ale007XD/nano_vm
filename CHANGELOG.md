# Changelog

All notable changes to `llm-nano-vm` will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.7.0] — 2026-05-11

### Added

- **`suspend / resume` — webhook-driven execution.**  
  A tool returning the sentinel `"PENDING"` transitions the VM to `TraceStatus.SUSPENDED`
  and persists a cursor. Execution resumes from that cursor when an external event arrives
  (payment webhook, courier confirmation, etc.).

  ```python
  trace = await vm.run(program, context={"order_id": "123"})
  assert trace.status == TraceStatus.SUSPENDED

  trace = await vm.resume_with_program(
      program=program,
      trace_id=trace.trace_id,
      webhook_event={"type": "payment.confirmed", "order_id": "123"},
  )
  assert trace.status == TraceStatus.SUCCESS
  ```

  `InMemoryCursorRepository` ships for tests and dry-run.
  Inject `cursor_repo=SqliteCursorRepository(...)` for production.
  `resume_with_program()` is the stable API until Blueprint registry (P8) is implemented.

- **`BudgetInterrupt` — isolated system interrupt.**  
  Budget exhaustion emits `InterruptType.BUDGET` before the next step executes.
  The LLM cannot observe or influence this signal. Override `_emit_interrupt()` in a
  subclass to route to your observability stack (no magic, standard Python inheritance).

- **`VaultStepResult` + `VaultStepMetadata` — MCP-compatible DTOs.**  
  `status` is a plain string (`"SUCCESS" | "FAILED" | "PENDING"`), not an enum —
  required for round-trip JSON serialization through the MCP layer.
  `@model_validator` enforces the allowed set at construction time.

- **`Trace.trace_id` — UUID4, OTel-ready.**  
  Auto-generated (`default_factory=uuid4`). Propagated into `VaultStepMetadata.trace_id`
  and `AuditEvent.trace_id`. OTel exporter is a separate concern (P13) — the field is
  stable from this release.

- **`erase()` — nested `CapabilityRef` tombstoning.**  
  `erase(state, target_ids) → (StateContext, int)` — pure function, returns new state
  plus count of tombstoned refs. Traverses arbitrary depth of nested `dict` / `list`.
  `is_tombstone=True` causes `secure_hash()` to return `"TOMBSTONE"` and all projections
  to return `[REDACTED_TOMBSTONE]`, preserving the hash chain without exposing erased data.

- **`ASTEngine` — `eval()` removed from production execution path.**  
  Condition expressions are parsed into a validated JSON AST and evaluated by a pure,
  sandboxed evaluator. No Python builtins accessible.  
  Supported operators: `==`, `!=`, `>`, `<`, `in`, `not in`, `and`, `or`, `contains`.

- **`ProjectionLayer` API.**  
  `AbstractProjectionLayer` with targets `LLM`, `TRACE`, `TOOL`.  
  `DeterministicSanitizer` base implementation (regex + field rules) injected into
  FSM lifecycle hooks.

- **`TraceStatus.SUSPENDED`** added to the FSM transition table.  
  `SUSPENDED` is not a terminal state — `resume_with_program()` transitions back to `RUNNING`.

- **Stress test suite v0.7.0 (`benchmarks/benchmark_stress_060`).**  
  10,000 FSM graphs × 5 runs · Mock adapter (CPU-bound).  
  Average: **14,327 graphs/sec · 0.70 s/run**.  
  89.73% success rate matches `P(value ≤ 0.9) = 0.9` by design.
  Identical results across all 5 runs — dataset fixed before loop (determinism confirmed).

### Changed

- `ExecutionVM.__init__` — `cursor_repo` kwarg added (default: `InMemoryCursorRepository()`).
  Fully backward-compatible; existing code requires no changes.

- `vm.py` — condition step evaluation delegates to `ASTEngine` instead of `eval()`.

- README — new sections: **Suspend / Resume**, **Budget Interrupts**, **MCP-Compatible Contracts**.
  FSM transition table updated with `SUSPENDED` state and `resume_with_program()` row.
  Security note updated: `eval()` replaced by ASTEngine.
  Roadmap updated with all v0.7.0 items.

### Breaking Changes

None. All v0.6.0 public APIs are preserved. New parameters are keyword-optional with safe defaults.

| Symbol | Change |
| :--- | :--- |
| `ExecutionVM.__init__` | `cursor_repo` kwarg added (default: `InMemoryCursorRepository()`) |
| `Trace` | `trace_id` field added (UUID4, auto-generated) |
| `TraceStatus` | `SUSPENDED` value added |

---

## [0.6.0] — 2026-05-03

### Added

- **FSM invariant stress suite (`benchmarks/benchmark_nano_vm.py`).**  
  13 tests (BM-01–BM-12 + BM-VM) validating δ(S, E) → S' under chaos, injection, replay,
  and concurrent load. Array size: 10,000 per test · 5 runs · seed=42.  
  Result: **13/13 PASSED · 1,020,000 total operations · 0 invariant violations.**

- **`## Execution Pipeline` section in README.**  
  Canonical formal model with layer responsibility table and Implementation Note on
  current single-action `A(S)` vs future multi-candidate Policy.

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

### Known issues

- **BM8 blocked by rate limit (429) during peak hours** (inherited from v0.5.0).  
  Workaround candidates: `qwen/qwen3-coder:free`, `nvidia/nemotron-nano-9b-v2:free`.

---

## [0.5.0] — 2025-04-30

### Added

- **`Planner` — full implementation.**  
  `Planner(llm, max_retries=2, temperature=0.0).generate(intent, available_tools?, context_keys?)`
  converts a natural-language intent into a validated `Program` in exactly one LLM call.
  Signature stable; all parameters keyword-optional.

- **Benchmark suite v0.5.0 (`benchmarks/benchmark_v050.py`).**  
  BM1–BM11 covering retry, budget guards, token tracking, fingerprints, parallel concurrency,
  Planner determinism (BM11), and real OpenRouter multi-model calls (BM8).

- **`benchmarks/run_all.py` — unified benchmark runner.**  
  Loads modules via `importlib.util`; prints per-call `✓ / ✗ + error` inline.

### Fixed

- `@dataclass` crash under `importlib.util` load — `sys.modules[name] = mod` registered
  before `spec.loader.exec_module(mod)`.
- `sys.exit(1)` in import guard replaced with `raise ImportError(...)`.
- Stale OpenRouter free-tier models replaced with live verified models.
- `CallResult.error` now printed inline in `run_suite_real`.

### Known issues

- **BM8 blocked by rate limit (429) during peak hours.**

---

## [0.4.0] — 2025-03-XX

### Added

- `max_steps` budget — `BUDGET_EXCEEDED` after N total steps.
- `max_stalled_steps` budget — `STALLED` after N consecutive no-op fingerprints.
- `max_tokens` budget — `BUDGET_EXCEEDED` when cumulative token count exceeds limit.
- `state_snapshots` in `Trace` — `list[(step_index, sha256_hex)]`.
- `on_error`, `max_retries` per-step options.
- `max_concurrency` for `parallel` blocks.

### Fixed

- **`total_tokens()` O(N²) per step** — fixed in v0.5.0 via incremental `_token_accumulator`
  in `Trace.add_step` (O(1)).

---

## [0.3.0] — 2025-02-XX

### Added

- `max_concurrency` — cap concurrent sub-steps per `parallel` block.
- `retry` policy per step — exponential backoff: 1 s, 2 s, 4 s … cap 30 s.

---

## [0.2.0] — 2025-01-XX

### Added

- `parallel` step type — `asyncio.gather` for independent sub-steps.
- `MockLLMAdapter` — deterministic testing without API keys.

---

## [0.1.0] — 2025-01-XX

### Added

- FSM execution engine (`ExecutionVM`).
- `llm`, `tool`, `condition` step types.
- `LiteLLMAdapter` + cost tracking.
- `Trace` — full per-step log: status, cost, tokens, duration.
- Published to PyPI as `llm-nano-vm`.
