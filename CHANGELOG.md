# Changelog

All notable changes to `llm-nano-vm` will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.7.5] — 2026-05-18

### Added

- **`ASTEngine` — METHOD_CALL guard in `_tokenise`.**  
  Previously `$x.lower()` was parsed as `VarNode(name='x.lower')` — the parentheses fell
  outside the regex, resulting in `None` comparison → silent `False` at runtime.  
  Fix: a `METHOD_CALL` token pattern is inserted before `VAR` in `_TOKEN_PATTERNS`.
  When `_tokenise` encounters `$word.method(`, it raises `ASTEvalError` immediately at
  parse time with an explicit message.  
  Affected patterns: `.lower()`, `.strip()`, `.upper()`, `.split()`, any method call on a
  variable reference.

- **`test_v075.py` — 14 tests (OC-01–07 + MC-01–07).**
  - OC-01–07: otherwise-chain verification — 3-level chains, symmetric `then`/`otherwise`,
    `next_step` continuation, regression against v0.7.3-compatible programs. All 7 PASS
    without any vm.py patch — `BUG-FSM-OTHERWISE-CHAIN` was not reproducible in v0.7.4.
  - MC-01–07: METHOD_CALL VMError — explicit error on `.lower()`/`.strip()`/`.upper()`,
    valid operator regression (`in`, `not in`, `contains`, `==`, dotted-path). All 7 PASS.

### Fixed

- **`BUG-ASTENGINE-NO-METHOD-CALLS` — `ASTEvalError` at parse time instead of silent `False`.**  
  Any condition using `$var.lower()`, `$var.strip()`, `$var.upper()` silently evaluated
  to `False` in v0.7.4 and earlier. Post-fix, such expressions raise immediately with an
  actionable error message.

- **`BUG-FSM-OTHERWISE-CHAIN` — diagnosed as NOT reproducible in v0.7.4 vm.py.**  
  OC-01–07 (including 3-level `otherwise` chains) all PASS without patch. The bug existed
  in a pre-v0.7.4 snapshot; the v0.7.4 fix of `_execute_loop` already covered both
  `then` and `otherwise` recursion symmetrically. Closed.

- **Stripe PoC — `PENDING` sentinel collision resolved.**  
  `create_payment_intent` previously returned `'PENDING'` for 3DS status, triggering
  FSM `SUSPENDED` instead of continuing execution.  
  Fix: renamed to `'REQUIRES_ACTION'` (matches actual Stripe `requires_action` status).
  Condition DSL and test mocks updated.

- **Stripe PoC — `refund_guardrail` condition `.lower()` removed.**  
  After `BUG-ASTENGINE-NO-METHOD-CALLS` fix, `'"yes" in $refund_decision.lower()'` raised
  `ASTEvalError` at parse time.  
  Fix: condition → `'"yes" in $refund_decision'`; prompt → `'Reply ONLY with the word: yes or no'`.

### Known Limitations

- `'PENDING'` is a reserved FSM suspend sentinel. Tools must not return `'PENDING'` as a
  domain status. Use `'REQUIRES_ACTION'`, `'AWAITING_3DS'`, or any other string.
- ASTEngine does not support method calls, arithmetic, or parentheses grouping.  
  Supported: `==`, `!=`, `>`, `<`, `in`, `not in`, `and`, `or`, `not`, `contains`,
  dotted-path `$var.field`.

### CI

179/179 tests — 121 regression + 22 FA + 22 v0.7.4 + 14 v0.7.5.  
MoMo PoC v4: 9/9. Stripe PoC v1: 9/9.

### Breaking Changes

None. All v0.7.4 programs are fully compatible.

---

## [0.7.4] — 2026-05-16

### Added

- **`Step.is_terminal: bool = False`** — when `True`, FSM returns `SUCCESS` immediately
  after executing that step. Required for leaf steps in condition branches that share the
  flat step array with the main flow (e.g. `notify_success`, `reject_payment`).

- **`Step.next_step: str | None = None`** — when set on a branch target, the FSM jumps
  to the named step instead of terminating. Enables condition branches that rejoin the
  main sequential flow. Supports multi-hop chains.

- **`test_v074_condition_semantics.py`** — 22 tests covering CB, CTX, RES.

### Fixed

- **`_execute_condition`: `$step_id.output` and `$step_id.output.field` now resolve correctly.**  
  Root cause: step_outputs were merged as raw scalars into ctx; `$validate_amount.output`
  resolved to `None` → condition always `False`.  
  Fix: step_outputs wrapped as `{"output": v}` before merge.

- **`_resolve`: typed return for single-variable expressions.**  
  `$amount` now returns the original `int`/`dict`/`list` value; string interpolation
  (`"order-$id-suffix"`) still stringifies.

- **`_resolve`: multi-segment dotted path `$a.b.c.d` in tool args.**  
  Regex extended to `\$(\w+(?:\.\w+)*)`.

- **Condition branch semantics v0.7.4:**  
  1. Execute branch target.  
  2. If target is `condition` → recurse.  
  3. If `target.next_step` set → jump to named step.  
  4. Default → `SUCCESS` (terminal, v0.7.3-compatible).

### Breaking Changes

None. New fields are keyword-optional with safe defaults.

| Symbol | Change |
| :--- | :--- |
| `Step` | `is_terminal: bool = False` added |
| `Step` | `next_step: str | None = None` added |

---

## [0.7.3] — 2026-05-14

### Added

- **Integration benchmark suite** — 10/10 PASS, 1,096,500 ops, 0 violations.

  | ID | Scenario | Mean TPS | p95 avg | Verdict |
  | :--- | :--- | ---: | ---: | :--- |
  | BM-INT-01 | Refund pipeline | 2,300/s | 0.66 ms | ✓ PASS |
  | BM-INT-02 | Double-execution guard | 2,400/s | 0.67 ms | ✓ PASS |
  | BM-INT-03 | Budget enforcement | 1,100/s | 331 ms | ✓ PASS |
  | BM-INT-04 | Parallel throughput | 436/s | 542 ms | ✓ PASS |
  | BM-INT-05 | MCP store round-trip | 3,000/s | 0.42 ms | ✓ PASS |
  | BM-INT-06 | GovernanceEnvelope | 1,300/s | 171 ms | ✓ PASS |
  | BM-INT-07 | Crash consistency | 7/s | 233 ms | ✓ PASS |
  | BM-INT-08 | Replay equivalence | 1,300/s | 1.30 ms | ✓ PASS |
  | BM-INT-09 | Adversarial retries | 2,400/s | 0.64 ms | ✓ PASS |
  | BM-INT-10 | Long-horizon | 30/s | 3,606 ms | ✓ PASS |

### Breaking Changes

None.

---

## [0.7.0] — 2026-05-11

### Added

- **`suspend / resume`** — `"PENDING"` sentinel → `SUSPENDED`; `resume_with_program()` restores
  from cursor. `InMemoryCursorRepository` for tests; `SqliteCursorRepository` for production.
- **`BudgetInterrupt`** — `InterruptType.BUDGET` before next step; `_emit_interrupt()` hook.
- **`VaultStepResult` + `VaultStepMetadata`** — MCP-compatible DTOs with string `status`.
- **`Trace.trace_id`** — UUID4, OTel-ready.
- **`erase()`** — nested `CapabilityRef` tombstoning; hash-chain preserved.
- **`ASTEngine`** — `eval()` removed; operators: `==`, `!=`, `>`, `<`, `in`, `not in`,
  `and`, `or`, `not`, `contains`.
- **`ProjectionLayer` API** — `AbstractProjectionLayer`, `DeterministicSanitizer`.
- **`TraceStatus.SUSPENDED`** — non-terminal.

### Breaking Changes

None.

| Symbol | Change |
| :--- | :--- |
| `ExecutionVM.__init__` | `cursor_repo` kwarg added |
| `Trace` | `trace_id` added |
| `TraceStatus` | `SUSPENDED` added |

---

## [0.6.0] — 2026-05-03

### Added

- FSM invariant stress suite — 13/13 PASS · 1,020,000 ops · 0 violations.
- `## Execution Pipeline` section in README.

---

## [0.5.0] — 2025-04-30

### Added

- `Planner` — intent → validated `Program` in 1 LLM call.
- Benchmark suite BM1–BM11 + `benchmarks/run_all.py`.

---

## [0.4.0] — 2025-03-XX

### Added

- `max_steps`, `max_stalled_steps`, `max_tokens` budgets.
- `state_snapshots`, `on_error`, `max_retries`, `max_concurrency`.

---

## [0.3.0] — 2025-02-XX

### Added

- `max_concurrency` for `parallel` blocks.
- `retry` with exponential backoff (1 s, 2 s, 4 s … cap 30 s).

---

## [0.2.0] — 2025-01-XX

### Added

- `parallel` step type (`asyncio.gather`).
- `MockLLMAdapter`.

---

## [0.1.0] — 2025-01-XX

### Added

- `ExecutionVM` — FSM execution engine.
- `llm`, `tool`, `condition` step types.
- `LiteLLMAdapter` + cost tracking.
- `Trace` — full per-step log.
- Published to PyPI as `llm-nano-vm`.
