"""Тесты ExecutionVM с детерминированным mock-адаптером."""
import pytest
from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.models import StepStatus


class MockLLM:
    """Детерминированный адаптер для тестов."""

    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    async def complete(self, messages, **kwargs) -> str:
        return next(self._responses)


# ---------------------------------------------------------------------------
# Базовые тесты
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_llm_step():
    vm = ExecutionVM(llm=MockLLM(["Hello, World!"]))
    program = Program.from_dict({
        "name": "test",
        "steps": [{"id": "greet", "type": "llm", "prompt": "Say hello"}]
    })
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert trace.final_output == "Hello, World!"


@pytest.mark.asyncio
async def test_output_key_in_state():
    vm = ExecutionVM(llm=MockLLM(["Paris", "Paris is a city"]))
    program = Program.from_dict({
        "steps": [
            {"id": "s1", "type": "llm", "prompt": "Capital of France?", "output_key": "capital"},
            {"id": "s2", "type": "llm", "prompt": "Tell me about $capital"},
        ]
    })
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    # s2 получил $capital из state
    assert len(trace.steps) == 2


@pytest.mark.asyncio
async def test_tool_step():
    async def add(a: int, b: int) -> int:
        return a + b

    vm = ExecutionVM(llm=MockLLM([]), tools={"add": add})
    program = Program.from_dict({
        "steps": [{"id": "calc", "type": "tool", "tool": "add", "args": {"a": 2, "b": 3}}]
    })
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert trace.final_output == 5


@pytest.mark.asyncio
async def test_tool_not_registered():
    vm = ExecutionVM(llm=MockLLM([]))
    program = Program.from_dict({
        "steps": [{"id": "s1", "type": "tool", "tool": "unknown", "args": {}}]
    })
    trace = await vm.run(program)
    assert trace.status == TraceStatus.FAILED
    assert "не зарегистрирован" in trace.error


@pytest.mark.asyncio
async def test_condition_then_branch():
    vm = ExecutionVM(llm=MockLLM(["yes answer", "then branch"]))
    program = Program.from_dict({
        "steps": [
            {"id": "s1", "type": "llm", "prompt": "Q?", "output_key": "ans"},
            {
                "id": "check", "type": "condition",
                "condition": "'yes' in '$ans'",
                "then": "s3", "otherwise": "s4"
            },
            {"id": "s3", "type": "llm", "prompt": "Then path"},
            {"id": "s4", "type": "llm", "prompt": "Otherwise path"},
        ]
    })
    trace = await vm.run(program)
    step_ids = [r.step_id for r in trace.steps]
    assert "s3" in step_ids
    assert "s4" not in step_ids


@pytest.mark.asyncio
async def test_on_error_skip():
    async def fail_tool():
        raise RuntimeError("oops")

    vm = ExecutionVM(llm=MockLLM(["final"]), tools={"fail": fail_tool})
    program = Program.from_dict({
        "steps": [
            {"id": "s1", "type": "tool", "tool": "fail", "args": {}, "on_error": "skip"},
            {"id": "s2", "type": "llm", "prompt": "Continue"},
        ]
    })
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert trace.steps[0].status == StepStatus.SKIPPED
    assert trace.steps[1].status == StepStatus.SUCCESS


@pytest.mark.asyncio
async def test_context_variable_resolution():
    vm = ExecutionVM(llm=MockLLM(["answer"]))
    program = Program.from_dict({
        "steps": [{"id": "s1", "type": "llm", "prompt": "User said: $user_input"}]
    })
    trace = await vm.run(program, context={"user_input": "hello"})
    assert trace.status == TraceStatus.SUCCESS


@pytest.mark.asyncio
async def test_register_tool_after_init():
    vm = ExecutionVM(llm=MockLLM([]))
    vm.register_tool("ping", lambda: "pong")
    program = Program.from_dict({
        "steps": [{"id": "s1", "type": "tool", "tool": "ping", "args": {}}]
    })
    trace = await vm.run(program)
    assert trace.final_output == "pong"
