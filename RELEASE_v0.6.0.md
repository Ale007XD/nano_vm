# llm-nano-vm v0.6.0 — Release Notes

**Release date:** 2026-05-03  
**PyPI:** `pip install llm-nano-vm==0.6.0`

---

## Overview

v0.6.0 is a **foundation release for vault-class workloads** — transactional B2C systems
where execution guarantees must hold at the infrastructure level, not in the prompt.

Three primitives added: `StepResult` as a first-class VM contract, `BudgetInterrupt` as an
isolated signal, and `suspend/resume` via webhook events. Together they enable
[nano-vm-vault](https://github.com/Ale007XD/nano-vm-vault) to build a deterministic FSM
orchestrator on top of llm-nano-vm without patching the core.

---

## What's New

### `suspend` / `resume` via webhook events

`ExecutionVM` can now suspend mid-graph and resume from a persisted cursor when an
external event arrives (payment webhook, courier confirmation, etc.).

```python
# Suspend — tool returns sentinel "PENDING"
async def initiate_payment(**kwargs) -> str:
    await register_webhook_handler(order_id=kwargs["order_id"])
    return "PENDING"  # VM sees this, suspends, persists cursor


trace = await vm.run(program, context={"order_id": "123"})
assert trace.status == TraceStatus.SUSPENDED

# Resume — when webhook fires
trace = await vm.resume_with_program(
    program=program,
    trace_id=trace.trace_id,
    webhook_event={"type": "payment.confirmed", "order_id": "123"},
)
assert trace.status == TraceStatus.SUCCESS
```

**Cursor persistence:** inject a `CursorRepository` implementation.
`InMemoryCursorRepository` ships for tests and dry-run. Production requires
`SqliteCursorRepository(infrastructure.db)` — implement the `CursorRepository` Protocol.

```python
from nano_vm.vm import ExecutionVM, InMemoryCursorRepository

vm = ExecutionVM(llm=adapter, cursor_repo=InMemoryCursorRepository())
```

**`resume()` vs `resume_with_program()`:** `resume()` requires a Blueprint registry
(available at P8). Until then, pass the program explicitly via `resume_with_program()`.

### `BudgetInterrupt` — isolated signal, not control flow

Budget exhaustion is now a **system interrupt**, not a condition branch. The VM emits
`InterruptType.BUDGET` before touching the next step — the LLM cannot observe or
influence it.

```python
from nano_vm.vm import ExecutionVM, InterruptType


class InstrumentedVM(ExecutionVM):
    async def _emit_interrupt(self, interrupt_type: InterruptType) -> None:
        await notify_operator(f"interrupt: {interrupt_type.value}")


vm = InstrumentedVM(llm=adapter)
```

Override `_emit_interrupt()` in a subclass (standard Python inheritance, no magic).
The base implementation is a documented no-op hook.

### `VaultStepResult` + `VaultStepMetadata` — MCP-compatible contracts

New DTOs for vault integration. `status` is a plain string (`"SUCCESS" | "FAILED" | "PENDING"`),
not an enum — required for round-trip JSON serialization through MCP layer.

```python
from nano_vm.models import VaultStepResult, VaultStepMetadata

result = VaultStepResult(
    status="SUCCESS",
    data={"payment_id": "pay_123"},
    metadata=VaultStepMetadata(
        idempotency_key="order_1:pay_step:initiate_payment",
        execution_time_ms=142,
        tool_version="1.0.0",
        cached=False,
        trace_id=str(uuid4()),
    ),
)
```

`@model_validator` enforces `status ∈ {"SUCCESS", "FAILED", "PENDING"}` at construction time.

### `Trace.trace_id` — OTel propagation from day one

`Trace` now carries a `trace_id: str` (UUID4, `default_factory`). Propagated into
`VaultStepMetadata.trace_id` and `AuditEvent.trace_id`. OTel exporter is a separate
concern (P13) — the field is stable from this release.

```python
trace = await vm.run(program)
print(trace.trace_id)  # "3f2a1b4c-..."
```

---

## Benchmark — Stress Test v0.6.0

**10 000 FSM graphs × 5 runs · Python 3.12 · Mock adapter (CPU-bound)**

```
┌──────┬──────────────┬────────────┬────────┬────────┐
│ Run  │ Time (sec)   │ Speed      │ OK     │ Failed │
├──────┼──────────────┼────────────┼────────┼────────┤
│  1   │   0.70       │ 14 286 /s  │  8973  │  1027  │
│  2   │   0.70       │ 14 286 /s  │  8973  │  1027  │
│  3   │   0.69       │ 14 493 /s  │  8973  │  1027  │
│  4   │   0.70       │ 14 286 /s  │  8973  │  1027  │
│  5   │   0.70       │ 14 286 /s  │  8973  │  1027  │
├──────┼──────────────┼────────────┼────────┼────────┤
│ AVG  │   0.70       │ 14 327 /s  │   —    │   —    │
└──────┴──────────────┴────────────┴────────┴────────┘
```

**Determinism:** identical results across all 5 runs — dataset fixed before loop.
8973 / 10 000 = 89.73 % success rate matches `P(value ≤ 0.9) = 0.9` (1027 errors
from `unregistered_tool_to_force_error` triggered at `value > 0.9`).

**Failure isolation:** `VMError: Tool not found` on failed graphs is caught per-coroutine.
Event loop continues without interruption across 200 concurrent tasks.

**VM overhead:** near-zero at Mock adapter. In production, bottleneck is LLM API latency
and DB I/O — not the FSM core.

Previous benchmarks (double-execution safety, budget overhead, Planner determinism):
see [v0.5.0 release notes](https://github.com/Ale007XD/nano_vm/releases/tag/v0.5.0).

---

## Breaking Changes

None. All v0.5.0 public APIs are preserved. New parameters are keyword-optional with
safe defaults.

| Symbol | Change |
|--------|--------|
| `ExecutionVM.__init__` | `cursor_repo` kwarg added (default: `InMemoryCursorRepository()`) |
| `Trace` | `trace_id` field added (UUID4, auto-generated) |
| `TraceStatus` | `SUSPENDED` value added |

---

## Migration from v0.5.0

```bash
pip install llm-nano-vm==0.6.0
```

No code changes required for existing programs. `suspend/resume` is opt-in via
tool sentinel `"PENDING"` and `cursor_repo` injection.

---

## What's Next

`nano-vm-vault` Фаза 1 (P0–P9): transactional core building on v0.6.0 primitives.  
`nano-vm-mcp` v0.1.0: `run_program`, `get_trace`, SQLite WAL, SSE + Bearer auth — released.

---

## Links

- [nano-vm-vault spec v1.2.0](https://github.com/Ale007XD/nano-vm-vault)
- [nano-vm-mcp](https://github.com/Ale007XD/nano-vm-mcp)
- [Changelog](https://github.com/Ale007XD/nano_vm/blob/main/CHANGELOG.md)
