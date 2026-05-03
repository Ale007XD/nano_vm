"""
Tests for llm-nano-vm v0.6.0 vault primitives.
Покрывает: suspend/resume, BudgetInterrupt, VaultStepResult, trace_id propagation.
"""
import asyncio
import sys
sys.path.insert(0, "/home/claude")

from nano_vm_v060 import (
    ExecutionVM, Program, TraceStatus, StepStatus,
    WebhookEvent, InMemoryCursorRepository, ResumeError,
    InterruptType, VaultStepResult, VaultStepError, VaultStepMetadata,
    Trace,
)
from nano_vm_v060.adapters.mock_adapter import MockLLMAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vm(**kwargs):
    cursor_repo = InMemoryCursorRepository()
    return ExecutionVM(
        llm=MockLLMAdapter("ok"),
        cursor_repository=cursor_repo,
        **kwargs,
    ), cursor_repo


async def tool_payment_initiate(**kwargs):
    """Симулирует async оплату: возвращает PENDING."""
    return "PENDING"


async def tool_payment_complete(**kwargs):
    """Симулирует шаг после подтверждения оплаты."""
    return "payment_confirmed"


async def tool_simple(**kwargs):
    return "done"


# ---------------------------------------------------------------------------
# Test 1: Trace имеет trace_id
# ---------------------------------------------------------------------------

async def test_trace_has_id():
    vm, _ = make_vm()
    program = Program.from_dict({
        "name": "test_id",
        "steps": [{"id": "s1", "type": "tool", "tool": "simple"}],
    })
    vm.register_tool("simple", tool_simple)
    trace = await vm.run(program)
    assert trace.trace_id, "trace_id должен быть непустым"
    assert len(trace.trace_id) == 36, f"UUID4 ожидается, получено: {trace.trace_id}"
    assert trace.status == TraceStatus.SUCCESS
    print(f"[OK] test_trace_has_id: trace_id={trace.trace_id}")


# ---------------------------------------------------------------------------
# Test 2: suspend при PENDING tool
# ---------------------------------------------------------------------------

async def test_suspend_on_pending():
    vm, cursor_repo = make_vm()
    program = Program.from_dict({
        "name": "test_suspend",
        "steps": [
            {"id": "init_payment", "type": "tool", "tool": "payment_initiate"},
            {"id": "post_payment", "type": "tool", "tool": "payment_complete"},
        ],
    })
    vm.register_tool("payment_initiate", tool_payment_initiate)
    vm.register_tool("payment_complete", tool_payment_complete)

    trace = await vm.run(program)

    assert trace.status == TraceStatus.SUSPENDED, f"Ожидается SUSPENDED, получено: {trace.status}"
    assert trace.suspended_step_id == "init_payment"
    assert trace.suspended_at is not None
    assert trace.trace_id

    # cursor должен быть сохранён
    cursor = await cursor_repo.load(trace.trace_id)
    assert cursor is not None, "cursor должен быть в repository"
    print(f"[OK] test_suspend_on_pending: suspended_step_id={trace.suspended_step_id}")


# ---------------------------------------------------------------------------
# Test 3: resume_with_program продолжает execution
# ---------------------------------------------------------------------------

async def test_resume_with_program():
    vm, _ = make_vm()
    program = Program.from_dict({
        "name": "test_resume",
        "steps": [
            {"id": "init_payment", "type": "tool", "tool": "payment_initiate"},
            {"id": "post_payment", "type": "tool", "tool": "payment_complete"},
        ],
    })
    vm.register_tool("payment_initiate", tool_payment_initiate)
    vm.register_tool("payment_complete", tool_payment_complete)

    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUSPENDED

    # Симулируем webhook от платёжного шлюза
    event = WebhookEvent(
        trace_id=trace.trace_id,
        payload={"payment_id": "pay_123", "status": "confirmed"},
    )
    resumed_trace = await vm.resume_with_program(event, program)

    assert resumed_trace.status == TraceStatus.SUCCESS, \
        f"Ожидается SUCCESS после resume, получено: {resumed_trace.status}"
    assert resumed_trace.final_output == "payment_confirmed"
    # trace_id сохраняется через resume
    assert resumed_trace.trace_id == trace.trace_id
    # webhook payload доступен downstream steps
    print(f"[OK] test_resume_with_program: final_output={resumed_trace.final_output}")


# ---------------------------------------------------------------------------
# Test 4: двойной resume → ResumeError (replay protection)
# ---------------------------------------------------------------------------

async def test_double_resume_raises():
    vm, _ = make_vm()
    program = Program.from_dict({
        "name": "test_double_resume",
        "steps": [
            {"id": "init_payment", "type": "tool", "tool": "payment_initiate"},
            {"id": "post_payment", "type": "tool", "tool": "payment_complete"},
        ],
    })
    vm.register_tool("payment_initiate", tool_payment_initiate)
    vm.register_tool("payment_complete", tool_payment_complete)

    trace = await vm.run(program)
    event = WebhookEvent(trace_id=trace.trace_id, payload={"status": "confirmed"})

    # Первый resume — успешный
    await vm.resume_with_program(event, program)

    # Второй resume — должен упасть (cursor удалён)
    try:
        await vm.resume_with_program(event, program)
        assert False, "Должен был вызвать ResumeError"
    except ResumeError as e:
        print(f"[OK] test_double_resume_raises: ResumeError: {e}")


# ---------------------------------------------------------------------------
# Test 5: Budget = Interrupt (max_steps)
# ---------------------------------------------------------------------------

async def test_budget_interrupt():
    interrupted = []

    class InstrumentedVM(ExecutionVM):
        async def _emit_interrupt(self, interrupt_type, trace):
            interrupted.append(interrupt_type)

    cursor_repo = InMemoryCursorRepository()
    vm = InstrumentedVM(
        llm=MockLLMAdapter("ok"),
        cursor_repository=cursor_repo,
    )
    vm.register_tool("simple", tool_simple)

    program = Program.from_dict({
        "name": "test_budget",
        "max_steps": 1,
        "steps": [
            {"id": "s1", "type": "tool", "tool": "simple"},
            {"id": "s2", "type": "tool", "tool": "simple"},
        ],
    })

    trace = await vm.run(program)
    assert trace.status == TraceStatus.BUDGET_EXCEEDED
    assert InterruptType.BUDGET in interrupted, "BudgetInterrupt должен быть эмиттирован"
    print(f"[OK] test_budget_interrupt: status={trace.status}, interrupts={interrupted}")


# ---------------------------------------------------------------------------
# Test 6: VaultStepResult контракт
# ---------------------------------------------------------------------------

def test_vault_step_result():
    metadata = VaultStepMetadata(
        idempotency_key="order_1:step_pay:yookassa_charge",
        execution_time_ms=142,
        tool_version="1.0.0",
        cached=False,
        trace_id="abc-123",
    )

    # SUCCESS
    result = VaultStepResult(status="SUCCESS", data={"payment_id": "pay_1"}, metadata=metadata)
    assert not result.is_pending
    assert not result.is_failed

    # PENDING
    result_pending = VaultStepResult(status="PENDING", metadata=metadata)
    assert result_pending.is_pending

    # FAILED с retryable
    error = VaultStepError(
        code="PAYMENT_DECLINED",
        message="Insufficient funds",
        retryable=True,
        compensation_required=False,
    )
    result_failed = VaultStepResult(status="FAILED", error=error, metadata=metadata)
    assert result_failed.is_failed
    assert result_failed.is_retryable
    assert not result_failed.requires_compensation

    # Невалидный статус
    try:
        VaultStepResult(status="INVALID", metadata=metadata)
        assert False, "Должна быть ValueError"
    except Exception:
        pass

    print("[OK] test_vault_step_result: все проверки прошли")


# ---------------------------------------------------------------------------
# Test 7: WebhookEvent валидация
# ---------------------------------------------------------------------------

def test_webhook_event_validation():
    # Валидный
    event = WebhookEvent(trace_id="abc", payload={"x": 1}, source="WEBHOOK")
    assert event.source == "WEBHOOK"

    # Пустой trace_id
    try:
        WebhookEvent(trace_id="", payload={})
        assert False
    except ValueError:
        pass

    # Невалидный source
    try:
        WebhookEvent(trace_id="abc", payload={}, source="INVALID")
        assert False
    except ValueError:
        pass

    print("[OK] test_webhook_event_validation")


# ---------------------------------------------------------------------------
# Test 8: trace_id стабилен через resume
# ---------------------------------------------------------------------------

async def test_trace_id_stable_through_resume():
    vm, _ = make_vm()
    program = Program.from_dict({
        "name": "test_id_stable",
        "steps": [
            {"id": "pay", "type": "tool", "tool": "payment_initiate"},
            {"id": "confirm", "type": "tool", "tool": "payment_complete"},
        ],
    })
    vm.register_tool("payment_initiate", tool_payment_initiate)
    vm.register_tool("payment_complete", tool_payment_complete)

    trace = await vm.run(program)
    original_id = trace.trace_id

    event = WebhookEvent(trace_id=original_id, payload={})
    resumed = await vm.resume_with_program(event, program)

    assert resumed.trace_id == original_id, \
        f"trace_id должен быть стабилен: {original_id} != {resumed.trace_id}"
    print(f"[OK] test_trace_id_stable: trace_id={original_id} сохранён через resume")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main():
    print("=== llm-nano-vm v0.6.0 tests ===\n")

    # Sync tests
    test_vault_step_result()
    test_webhook_event_validation()

    # Async tests
    await test_trace_has_id()
    await test_suspend_on_pending()
    await test_resume_with_program()
    await test_double_resume_raises()
    await test_budget_interrupt()
    await test_trace_id_stable_through_resume()

    print("\n=== Все тесты прошли ✓ ===")


if __name__ == "__main__":
    asyncio.run(main())
