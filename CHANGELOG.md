# Changelog

All notable changes to `llm-nano-vm` will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.7.4] — 2026-05-16

### Added

- **`Step.is_terminal: bool = False` — explicit halt marker.**  
  When `True`, the FSM returns `SUCCESS` immediately after executing that step.
  Required for leaf steps in condition branches that share the same flat step array
  with the main flow (e.g. `notify_success`, `reject_payment`, `alert_ops_step`).
  Without this marker, the FSM would continue linearly to the next step in the array.

- **`Step.next_step: str | None = None` — inline branch continuation.**  
  When set on a branch target, the FSM jumps to the named step instead of terminating.
  Enables condition branches that rejoin the main sequential flow
  (e.g. `amount_check → create_payment(next_step="poll_payment") → poll_payment → …`).
  Supports multi-hop chains: each step in the chain can specify its own `next_step`.

- **`test_v074_condition_semantics.py` — 22 tests for new branch semantics.**  
  Coverage: CB (condition branch), CTX (_execute_condition ctx), RES (_resolve typed return).

### Fixed

- **`_execute_condition`: `$step_id.output` and `$step_id.output.field` now resolve correctly.**  
  Previously `_execute_condition` built ctx as `{**state.step_outputs, **state.data}` where
  `step_outputs["validate_amount"] = "OK"` (raw scalar). ASTEngine's dotted-path resolver
  tried `ctx["validate_amount"]["output"]` on a string → `None` → condition always `False`.  
  Fix: step_outputs are wrapped as `{"output": v}` before merging into ctx, so
  `$validate_amount.output` → `ctx["validate_amount"]["output"]` = `"OK"` ✓ and
  `$poll_payment.output.payment_status` → correct field traversal ✓.  
  `output_key` flat aliases (`$validation`) continue to work via `state.data` merge.

- **`_resolve`: typed return for single-variable expressions.**  
  Previously `_resolve("$amount", state)` always returned `str("50000")` even when
  `state.data["amount"] = 50000` (int). Tool functions with typed signatures
  (e.g. `def validate_amount(amount: int)`) received a string and raised `TypeError`.  
  Fix: when the entire value is a single `$var` token, the original typed value is returned
  unchanged. String interpolation (`"order-$id-suffix"`) still stringifies as before.

- **`_resolve`: multi-segment dotted path `$a.b.c.d` in tool args.**  
  Previously regex `\$(\w+(?:\.\w+)?)` captured only one dot segment.
  `$generate_ids.output.order_id` was resolved as `$generate_ids.output` — losing `.order_id`.  
  Fix: regex extended to `\$(\w+(?:\.\w+)*)`, lookup uses full segment traversal with
  transparent `output` skip for scalar step outputs.

- **Condition branch semantics: terminal by default, `next_step` for inline continuation.**  
  Previously condition branch executed the target step and returned `SUCCESS` (v0.7.3),
  which prevented multi-step flows where the branch target was part of the main pipeline
  (e.g. `amount_check → create_payment → poll_payment`).  
  The BUG-FSM-CONDITION-CHAIN fix (v0.7.3) introduced condition→condition chaining but
  did not address non-condition inline targets.  
  v0.7.4 semantics:
  1. Execute branch target inline.
  2. If target is a `condition` step — recurse into its sub-branch (unchanged from v0.7.3).
  3. If `target.next_step` is set — continue from the named step (new inline continuation).
  4. Otherwise — return `SUCCESS` (terminal, v0.7.3-compatible default).

### Breaking Changes

None. All v0.7.3 programs without `is_terminal` or `next_step` fields behave identically
(branch target is terminal by default, condition→condition chains work as before).
New fields are keyword-optional with safe defaults (`False` / `None`).

| Symbol | Change |
| :--- | :--- |
| `Step` | `is_terminal: bool = False` field added |
| `Step` | `next_step: str | None = None` field added |

---

## [0.7.3] — 2026-05-14

### Added

- **Integration benchmark suite (`benchmarks/benchmark_integration.py`) — 10/10 PASS.**  
  End-to-end validation across the full stack: FSM kernel + MCP gateway + CapabilityRef
  contracts + GovernanceEnvelope + GDPR tombstoning + suspend/resume.  
  3 cycles × 5 runs × 10,000 items per scenario · **1,096,500 total operations · 0 violations.**

  Test environment: QEMU/KVM · Intel Xeon E5-2697A v4 @ 2.60 GHz · 2 cores / 2 threads ·
  2 GB ECC RAM · Python 3.12 · Mock adapter (no I/O).

  | ID | Scenario | Total items | Mean TPS | p95 avg | Verdict |
  | :--- | :--- | ---: | ---: | ---: | :--- |
  | BM-INT-01 | Refund pipeline | 150,000 | 2,300/s | 0.66 ms | ✓ PASS |
  | BM-INT-02 | Double-execution guard | 150,000 | 2,400/s | 0.67 ms | ✓ PASS |
  | BM-INT-03 | Budget enforcement | 150,000 | 1,100/s | 331 ms | ✓ PASS |
  | BM-INT-04 | Parallel throughput | 15,000 | 436/s | 542 ms | ✓ PASS |
  | BM-INT-05 | MCP store round-trip | 151,500 | 3,000/s | 0.42 ms | ✓ PASS |
  | BM-INT-06 | GovernanceEnvelope | 150,000 | 1,300/s | 171 ms | ✓ PASS |
  | BM-INT-07 | Crash consistency | 30,000 | 7/s | 233 ms | ✓ PASS |
  | BM-INT-08 | Replay equivalence | 75,000 | 1,300/s | 1.30 ms | ✓ PASS |
  | BM-INT-09 | Adversarial retries | 75,000 | 2,400/s | 0.64 ms | ✓ PASS |
  | BM-INT-10 | Long-horizon | 150,000 | 30/s | 3,606 ms | ✓ PASS |

  Extended metrics:
  - **BM-INT-07** crash_rate = 100% (expected 50–90%) — deterministic on 2-core QEMU guest;
    hardware-sensitive metric, not a regression.
  - **BM-INT-08** trace_hash_match = 100.00% — Merkle hash chain fully reproducible across replay.
  - **BM-INT-09** adversarial mix: 3,000 duplicate events · 1,000 out-of-order · 1,000 delayed.
  - **BM-INT-10** peak RSS = 216 MB · peak alloc = 4.29 MB — bounded on 2 GB VPS.

### Changed

- README — new section `### v0.7.3 — Integration benchmark suite` added to Performance.
  Reproduce block updated with `benchmark_integration.py` entry.
  Roadmap updated with BM-INT suite line.

### Breaking Changes

None.

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
