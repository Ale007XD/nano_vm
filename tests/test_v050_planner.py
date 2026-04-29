"""
tests/test_v050_planner.py
==========================
P5: Planner — 18 tests covering:
  - Happy path: valid intent → Program
  - available_tools / context_keys injection
  - JSON extraction: clean, fenced, embedded
  - Retry logic: JSON error → success, ValidationError → success
  - Feedback injection (error message propagates to next attempt)
  - Exhausted retries → PlannerError
  - LLMAdapter tuple return (LiteLLMAdapter-style)
  - PlannerError attributes (last_raw, attempts)
  - Minimal program (1 step)
  - Multi-step with condition
  - Parallel steps
  - _extract_json edge cases
"""

from __future__ import annotations

import json
import pytest

from nano_vm.models import Program, StepType, TraceStatus
from nano_vm.planner import Planner, PlannerError, _extract_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_program_json(**overrides) -> str:
    """Minimal valid program JSON string."""
    base = {
        "name": "test_program",
        "steps": [
            {
                "id": "step1",
                "type": "llm",
                "prompt": "Say hello",
                "output_key": "result",
            }
        ],
    }
    base.update(overrides)
    return json.dumps(base)


def _make_two_step_json() -> str:
    return json.dumps({
        "name": "classify_route",
        "description": "Classify and route",
        "steps": [
            {
                "id": "classify",
                "type": "llm",
                "prompt": "Classify: $user_input. Reply urgent or normal.",
                "output_key": "category",
            },
            {
                "id": "route",
                "type": "condition",
                "condition": "'urgent' in '$category'",
                "then": "handle_urgent",
                "otherwise": "handle_normal",
            },
            {"id": "handle_urgent", "type": "tool", "tool": "escalate"},
            {"id": "handle_normal", "type": "tool", "tool": "log"},
        ],
    })


def _make_parallel_json() -> str:
    return json.dumps({
        "name": "parallel_fetch",
        "steps": [
            {
                "id": "fetch",
                "type": "parallel",
                "output_key": "fetched",
                "parallel_steps": [
                    {"id": "weather", "type": "tool", "tool": "get_weather"},
                    {"id": "news", "type": "tool", "tool": "get_news"},
                ],
            },
            {
                "id": "briefing",
                "type": "llm",
                "prompt": "Summarize: $weather $news",
                "output_key": "result",
            },
        ],
    })


class MockAdapter:
    """Returns responses from a queue. Tracks all calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def complete(self, messages: list[dict], **kwargs) -> str:
        self.calls.append(messages)
        if not self._responses:
            raise RuntimeError("MockAdapter: response queue exhausted")
        return self._responses.pop(0)


class TupleAdapter:
    """Simulates LiteLLMAdapter tuple return: (str, dict|None)."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def complete(self, messages: list[dict], **kwargs) -> tuple[str, dict | None]:
        return self._response, {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_minimal_program():
    """Single llm step — minimal valid program."""
    adapter = MockAdapter([_make_program_json()])
    planner = Planner(llm=adapter)
    program = await planner.generate("Say hello")

    assert isinstance(program, Program)
    assert program.name == "test_program"
    assert len(program.steps) == 1
    assert program.steps[0].type == StepType.LLM


@pytest.mark.asyncio
async def test_generate_two_step_with_condition():
    adapter = MockAdapter([_make_two_step_json()])
    planner = Planner(llm=adapter)
    program = await planner.generate("Classify and route")

    assert len(program.steps) == 4
    assert program.steps[0].id == "classify"
    assert program.steps[1].type == StepType.CONDITION
    assert program.steps[1].then == "handle_urgent"
    assert program.steps[1].otherwise == "handle_normal"


@pytest.mark.asyncio
async def test_generate_parallel_steps():
    adapter = MockAdapter([_make_parallel_json()])
    planner = Planner(llm=adapter)
    program = await planner.generate("Fetch in parallel, then summarize")

    fetch = program.steps[0]
    assert fetch.type == StepType.PARALLEL
    assert len(fetch.parallel_steps) == 2
    assert fetch.parallel_steps[0].id == "weather"


@pytest.mark.asyncio
async def test_generate_returns_program_instance():
    adapter = MockAdapter([_make_program_json()])
    planner = Planner(llm=adapter)
    result = await planner.generate("anything")
    assert isinstance(result, Program)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_available_tools_in_prompt():
    adapter = MockAdapter([_make_program_json()])
    planner = Planner(llm=adapter)
    await planner.generate("do something", available_tools=["send_email", "save_db"])

    user_msg = adapter.calls[0][1]["content"]  # [system, user]
    assert "send_email" in user_msg
    assert "save_db" in user_msg


@pytest.mark.asyncio
async def test_context_keys_in_prompt():
    adapter = MockAdapter([_make_program_json()])
    planner = Planner(llm=adapter)
    await planner.generate("do something", context_keys=["user_id", "order_id"])

    user_msg = adapter.calls[0][1]["content"]
    assert "$user_id" in user_msg
    assert "$order_id" in user_msg


@pytest.mark.asyncio
async def test_intent_in_prompt():
    adapter = MockAdapter([_make_program_json()])
    planner = Planner(llm=adapter)
    await planner.generate("Classify and route urgent messages")

    user_msg = adapter.calls[0][1]["content"]
    assert "Classify and route urgent messages" in user_msg


@pytest.mark.asyncio
async def test_system_prompt_sent():
    adapter = MockAdapter([_make_program_json()])
    planner = Planner(llm=adapter)
    await planner.generate("test")

    system_msg = adapter.calls[0][0]
    assert system_msg["role"] == "system"
    assert "program generator" in system_msg["content"]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def test_extract_json_clean():
    data = _extract_json('{"name": "x", "steps": []}')
    assert data["name"] == "x"


def test_extract_json_fenced():
    raw = '```json\n{"name": "x"}\n```'
    data = _extract_json(raw)
    assert data["name"] == "x"


def test_extract_json_fenced_no_lang():
    raw = '```\n{"name": "y"}\n```'
    data = _extract_json(raw)
    assert data["name"] == "y"


def test_extract_json_embedded_in_text():
    raw = 'Here is the program:\n{"name": "z", "steps": []}\nDone.'
    data = _extract_json(raw)
    assert data["name"] == "z"


def test_extract_json_no_json_raises():
    with pytest.raises(ValueError, match="No JSON object"):
        _extract_json("This is just text with no JSON.")


def test_extract_json_invalid_json_raises():
    with pytest.raises(ValueError, match="Invalid JSON"):
        _extract_json('{"name": "x", broken}')


def test_extract_json_array_raises():
    with pytest.raises(ValueError, match="No JSON object found"):
        _extract_json("[1, 2, 3]")


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_on_json_error_then_success():
    """First response is invalid JSON, second is valid → succeeds on attempt 2."""
    adapter = MockAdapter(["not json at all", _make_program_json()])
    planner = Planner(llm=adapter, max_retries=2)
    program = await planner.generate("test")

    assert isinstance(program, Program)
    assert len(adapter.calls) == 2


@pytest.mark.asyncio
async def test_retry_on_validation_error_then_success():
    """First response fails pydantic validation, second is valid."""
    invalid = json.dumps({"name": "x", "steps": []})  # steps min_length=1 → ValidationError
    adapter = MockAdapter([invalid, _make_program_json()])
    planner = Planner(llm=adapter, max_retries=2)
    program = await planner.generate("test")

    assert isinstance(program, Program)
    assert len(adapter.calls) == 2


@pytest.mark.asyncio
async def test_feedback_injected_on_retry():
    """On retry, conversation must contain VALIDATION ERROR message."""
    invalid = json.dumps({"name": "x", "steps": []})
    adapter = MockAdapter([invalid, _make_program_json()])
    planner = Planner(llm=adapter, max_retries=2)
    await planner.generate("test")

    # Second call messages should contain the error feedback
    retry_messages = adapter.calls[1]
    assert any("VALIDATION ERROR" in m["content"] for m in retry_messages)


@pytest.mark.asyncio
async def test_exhausted_retries_raises_planner_error():
    """All attempts fail → PlannerError."""
    adapter = MockAdapter(["bad json", "also bad", "still bad"])
    planner = Planner(llm=adapter, max_retries=2)

    with pytest.raises(PlannerError) as exc_info:
        await planner.generate("test")

    assert exc_info.value.attempts == 3


@pytest.mark.asyncio
async def test_planner_error_last_raw():
    """PlannerError.last_raw contains the last LLM response."""
    adapter = MockAdapter(["bad json 1", "bad json 2", "bad json 3"])
    planner = Planner(llm=adapter, max_retries=2)

    with pytest.raises(PlannerError) as exc_info:
        await planner.generate("test")

    assert exc_info.value.last_raw == "bad json 3"


# ---------------------------------------------------------------------------
# Adapter compatibility
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tuple_adapter_compatibility():
    """LiteLLMAdapter returns tuple[str, dict] — Planner must handle it."""
    adapter = TupleAdapter(_make_program_json())
    planner = Planner(llm=adapter)
    program = await planner.generate("test")
    assert isinstance(program, Program)


@pytest.mark.asyncio
async def test_str_adapter_compatibility():
    """Standard Protocol adapter returns str — also works."""
    adapter = MockAdapter([_make_program_json()])
    planner = Planner(llm=adapter)
    program = await planner.generate("test")
    assert isinstance(program, Program)
