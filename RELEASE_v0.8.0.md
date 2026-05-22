# Release Notes — llm-nano-vm v0.8.0

**Released:** 2026-05-21  
**PyPI:** `pip install llm-nano-vm==0.8.0`  
**Tag:** `v0.8.0`  
**CI:** 432/432 passed · 0 violations

---

## Summary

v0.8.0 completes Sprint 5 DSL hardening. Two new `Step` fields close the gap between
nondeterministic LLM output and deterministic FSM execution:

- **`allowed_outputs`** — enforce exact output enum at the step level before conditions run.
- **`timeout_seconds` + `on_timeout`** — prevent a hung LLM call from stalling the FSM.

All v0.7.x programs are fully compatible. No breaking changes.

---

## New Features

### `Step.allowed_outputs` — LLM output enum guard

```python
{
    "id": "classify",
    "type": "llm",
    "prompt": "Classify the request. Reply ONLY with: refund / query / other",
    "output_key": "category",
    "allowed_outputs": ["refund", "query", "other"],
    "on_error": "skip",   # safe fallback → "refund" on unexpected output
}
```

After every LLM call the FSM checks the raw output string against the enum.
Behaviour is controlled by the existing `on_error` field:

| `on_error` | On mismatch |
| :--- | :--- |
| `fail` (default) | `VMError` → `trace.FAILED` |
| `skip` | output replaced with `allowed_outputs[0]` |
| `retry` | retry loop up to `max_retries`; `VMError` if exhausted |

Constraints enforced at `Program` construction time (`ValidationError`):
- Empty list is forbidden.
- Field is valid on `llm` steps only.

**Why it matters:** without `allowed_outputs`, an unexpected LLM string silently propagates
into conditions and branches, producing wrong paths instead of an explicit failure. With
`allowed_outputs`, the FSM fails fast and loudly at the correct step.

---

### `Step.timeout_seconds` + `Step.on_timeout` — per-step LLM timeout

```python
{
    "id": "analyze",
    "type": "llm",
    "prompt": "...",
    "allowed_outputs": ["approve", "reject"],
    "timeout_seconds": 5.0,
    "on_timeout": "fallback",   # → "approve" (allowed_outputs[0])
}
```

Wraps the LLM call in `asyncio.wait_for`. On expiry:

| `on_timeout` | Behaviour |
| :--- | :--- |
| `'fail'` (default) | `VMError` → `trace.FAILED` |
| `'fallback'` | output → `allowed_outputs[0]` if set, else `''` |

`allowed_outputs` and `timeout_seconds` are independent and fully composable — both can
be set on the same step (covered by test TO-07).

**Why it matters:** a payment or compliance pipeline cannot block indefinitely on a slow
LLM provider. `timeout_seconds` gives the FSM a hard ceiling; `on_timeout=fallback`
allows graceful degradation without halting the workflow.

---

## Changed

- **Step fields table** in README and DSL docs updated to include the three new fields.
- **Performance section** updated to v0.8.0 CI count (432/432).
- **Roadmap** — `allowed_outputs` and `timeout_seconds` moved from Upcoming → Done.

---

## Fixed

- **`VMError` containment clarified in docs.** `VMError` is caught inside
  `_execute_with_retry` and does not propagate to the caller. Failures are observable
  via `trace.status == TraceStatus.FAILED` and `trace.error`. Tests that use
  `pytest.raises(VMError)` at the top level will not fire; use `trace.status` instead.

---

## Test Coverage

| File | Tests | Status |
| :--- | :--- | :--- |
| `tests/test_v080_sprint5_allowed.py` | AO-01–09 (9) | ✅ PASS |
| `tests/test_v080_sprint5_timeout.py` | TO-01–07 (7) | ✅ PASS |
| Regression (all prior suites) | 416 | ✅ PASS |
| **Total** | **432** | **✅** |

Selected test IDs and what they cover:

| ID | Scenario |
| :--- | :--- |
| AO-01 | `allowed_outputs` match → `SUCCESS` |
| AO-02 | mismatch + `on_error=fail` → `FAILED` |
| AO-03 | mismatch + `on_error=skip` → `allowed_outputs[0]`, `SUCCESS` |
| AO-04 | mismatch + `on_error=retry` → retry loop → `FAILED` |
| AO-05 | empty list → `ValidationError` at construction |
| AO-06 | `allowed_outputs` on `tool` step → `ValidationError` |
| AO-07 | multi-value enum, second element match → `SUCCESS` |
| AO-08 | `None` (field absent) → no validation, `SUCCESS` |
| AO-09 | case-sensitive: `'Yes'` ≠ `'yes'` → `FAILED` |
| TO-01 | timeout + `on_timeout=fail` → `FAILED` |
| TO-02 | timeout + `on_timeout=fallback` + `allowed_outputs` → `allowed_outputs[0]`, `SUCCESS` |
| TO-03 | timeout + `on_timeout=fallback` without `allowed_outputs` → `''`, `SUCCESS` |
| TO-04 | no timeout, fast call → `SUCCESS` (regression) |
| TO-05 | `timeout_seconds=None` → no wrapping (regression) |
| TO-06 | `on_timeout` default is `'fail'` |
| TO-07 | `allowed_outputs` + `timeout_seconds` combined — both enforce independently |

---

## Migration

No changes required. All v0.7.x DSL programs run without modification.

To adopt the new fields in existing programs:

```python
# Before (v0.7.x) — unexpected LLM output silently propagates
{"id": "classify", "type": "llm", "prompt": "...", "output_key": "category"}

# After (v0.8.0) — fail fast on unexpected output
{
    "id": "classify",
    "type": "llm",
    "prompt": "Classify. Reply ONLY with: refund / query / other",
    "output_key": "category",
    "allowed_outputs": ["refund", "query", "other"],
    "on_error": "skip",
}
```

For LLM steps where latency is critical:

```python
{
    "id": "classify",
    "type": "llm",
    "prompt": "...",
    "allowed_outputs": ["approve", "reject"],
    "timeout_seconds": 10.0,
    "on_timeout": "fallback",   # safe degradation
}
```

---

## Known Limitations

- `allowed_outputs` is case-sensitive; exact string match only. Normalise via prompt
  (`Reply ONLY with: yes or no`) — do not use `.lower()` in conditions (raises
  `ASTEvalError` since v0.7.5).
- `timeout_seconds` applies to the LLM call only, not to tool steps.
- `'PENDING'` remains a reserved FSM suspend sentinel. Tools must not return it as a
  domain status.

---

## What's Next

**v0.8.x — DSL hardening:**
- `ProgramValidator` — static analysis: unreachable steps, missing branch targets,
  cycle detection.

**v0.8.x — observability:**
- OpenTelemetry span per FSM step.
- Incremental counters in `Trace`: `llm_calls`, `tool_calls`, `retries_total`.

**v0.9.x — gateway:**
- `nano-vm-mcp`: StateContext SQLite persistence (inter-session duplicate risk).
- `nano-vm-mcp`: `idempotency_store` (inter-session exactly-once guarantee).
- `nano-vm-mcp`: `GovernedToolExecutor` + circuit breaker.

---

## Checksums

Verify after download:

```bash
pip download llm-nano-vm==0.8.0 --no-deps -d /tmp/dist
sha256sum /tmp/dist/llm_nano_vm-0.8.0*.whl
```

Expected hash available on the [PyPI release page](https://pypi.org/project/llm-nano-vm/0.8.0/).

---

*llm-nano-vm is MIT licensed. © 2026 [@ale007xd](https://github.com/Ale007XD)*
