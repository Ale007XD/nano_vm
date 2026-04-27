"""
nano_vm.adapters.mock_adapter
=============================
Deterministic LLM adapter for testing and local development.

Usage:
    from nano_vm.adapters import MockLLMAdapter

    # Always returns the same string
    vm = ExecutionVM(llm=MockLLMAdapter("yes"))

    # Per-call sequence (cycles when exhausted)
    vm = ExecutionVM(llm=MockLLMAdapter(["yes", "no", "yes"]))

    # Per-prompt mapping
    vm = ExecutionVM(llm=MockLLMAdapter({"classify": "SAFE", "handle": "done"}))
    # key matched via substring of last user message

Properties:
    - Fully synchronous internally, exposed as async (matches LLMAdapter protocol)
    - call_count: total number of complete() calls made
    - calls: list of message lists passed to complete()
    - No network, no API key, no latency
"""

from __future__ import annotations

from typing import Any


class MockLLMAdapter:
    """
    Deterministic LLM adapter for testing.

    Args:
        response: str | list[str] | dict[str, str]
            - str: always return this value
            - list[str]: return items in order, cycling when exhausted
            - dict[str, str]: match substring of last user message to response;
              falls back to "__default__" key or empty string

    Example:
        adapter = MockLLMAdapter(["SAFE", "approved"])
        vm = ExecutionVM(llm=adapter)
        trace = await vm.run(program, context={"user_input": "test"})
        assert adapter.call_count == 2
    """

    def __init__(self, response: str | list[str] | dict[str, str] = "") -> None:
        self._response = response
        self._call_index = 0
        self.calls: list[list[dict[str, str]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        self.calls.append(messages)
        result = self._resolve(messages)
        self._call_index += 1
        return result

    def _resolve(self, messages: list[dict[str, str]]) -> str:
        if isinstance(self._response, str):
            return self._response

        if isinstance(self._response, list):
            if not self._response:
                return ""
            return self._response[self._call_index % len(self._response)]

        if isinstance(self._response, dict):
            last_content = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_content = msg.get("content", "")
                    break
            for key, value in self._response.items():
                if key != "__default__" and key in last_content:
                    return value
            return self._response.get("__default__", "")

        return ""

    def reset(self) -> None:
        """Reset call history and index. Useful between test cases."""
        self._call_index = 0
        self.calls.clear()
