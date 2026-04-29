"""Тесты Planner с mock-адаптером."""

import pytest

from nano_vm import Planner, PlannerError, Program


class MockLLM:
    def __init__(self, response: str):
        self._response = response

    async def complete(self, messages, **kwargs) -> str:
        return self._response


VALID_PROGRAM_JSON = """
{
  "name": "test_plan",
  "steps": [
    {"id": "s1", "type": "llm", "prompt": "Do something with $user_input"}
  ]
}
"""

PROGRAM_IN_MARKDOWN = """
```json
{
  "name": "wrapped",
  "steps": [
    {"id": "s1", "type": "llm", "prompt": "Hello"}
  ]
}
```
"""


@pytest.mark.asyncio
async def test_generate_valid_program():
    planner = Planner(llm=MockLLM(VALID_PROGRAM_JSON))
    program = await planner.generate("Do something")
    assert isinstance(program, Program)
    assert program.name == "test_plan"


@pytest.mark.asyncio
async def test_generate_from_markdown_block():
    planner = Planner(llm=MockLLM(PROGRAM_IN_MARKDOWN))
    program = await planner.generate("Do something")
    assert program.name == "wrapped"


@pytest.mark.asyncio
async def test_generate_invalid_json_raises():
    # D1: сообщение PlannerError — английское, соответствует planner.py
    planner = Planner(llm=MockLLM("not a json at all"))
    with pytest.raises(PlannerError, match="Failed to generate a valid Program"):
        await planner.generate("Do something")


@pytest.mark.asyncio
async def test_generate_invalid_structure_raises():
    planner = Planner(llm=MockLLM('{"name": "x", "steps": []}'))
    with pytest.raises(PlannerError):
        await planner.generate("Do something")


@pytest.mark.asyncio
async def test_tools_in_system_prompt():
    """Проверить что инструменты попадают в промпт.

    D2: tools — не параметр конструктора, передаются в generate(available_tools=).
    Planner кладёт их в user-сообщение (не system), поэтому ищем в любом сообщении.
    """
    received_messages = []

    class CaptureLLM:
        async def complete(self, messages, **kwargs):
            received_messages.extend(messages)
            return VALID_PROGRAM_JSON

    planner = Planner(llm=CaptureLLM())
    await planner.generate("Find and email results", available_tools=["search", "send_email"])

    full_content = " ".join(m["content"] for m in received_messages)
    assert "search" in full_content
    assert "send_email" in full_content
