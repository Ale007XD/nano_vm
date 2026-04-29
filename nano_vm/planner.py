"""
nano_vm.planner
===============
P5: P-planner — converts natural language intent into a validated Program DSL.

One LLM call (+ up to max_retries on ValidationError).
Nondeterminism is contained here; ExecutionVM runs deterministically.

Usage:
    from nano_vm import Planner, ExecutionVM
    from nano_vm.adapters import LiteLLMAdapter

    adapter = LiteLLMAdapter("openai/gpt-4o-mini")
    planner = Planner(llm=adapter)

    program = await planner.generate(
        "Fetch the latest news, summarize each article, then classify by topic"
    )
    trace = await ExecutionVM(llm=adapter).run(program)
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from .models import Program

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PlannerError(Exception):
    """Raised when Planner fails to produce a valid Program after all retries."""

    def __init__(self, message: str, last_raw: str = "", attempts: int = 0) -> None:
        super().__init__(message)
        self.last_raw = last_raw   # last LLM response for debugging
        self.attempts = attempts


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a program generator for llm-nano-vm.
Your task: convert the user's intent into a valid JSON program.

Output ONLY a JSON object — no markdown, no explanation, no code fences.

---
PROGRAM SCHEMA:
{
  "name": "<snake_case identifier>",
  "description": "<one sentence>",
  "steps": [ <step>, ... ]
}

STEP TYPES:

llm step:
  {"id": "<id>", "type": "llm", "prompt": "<prompt with $variables>", "output_key": "<key>"}

tool step:
  {"id": "<id>", "type": "tool", "tool": "<tool_name>", "args": {}, "output_key": "<key>"}

condition step:
  {"id": "<id>", "type": "condition", "condition": "'<value>' in '$<key>'", "then": "<step_id>", "otherwise": "<step_id>"}

parallel step:
  {"id": "<id>", "type": "parallel", "output_key": "<key>", "parallel_steps": [<llm or tool steps only>]}

RULES:
- Step ids must be unique, snake_case.
- llm steps require: prompt, output_key.
- tool steps require: tool name. Use tool names from the available_tools list if provided.
- condition steps require: condition expression, at least one of: then, otherwise.
- parallel sub-steps may only be llm or tool type (no nested parallel or condition).
- Use $variable_name to reference initial context. Use $step_id to reference a previous step output.
- Keep steps minimal — only what the user's intent requires.

---
EXAMPLE 1 — simple two-step pipeline:
User intent: "Classify the user's message as urgent or not, then route accordingly"
Output:
{
  "name": "classify_and_route",
  "description": "Classify user message and route based on urgency",
  "steps": [
    {
      "id": "classify",
      "type": "llm",
      "prompt": "Classify this message as 'urgent' or 'not_urgent'. Reply with one word only.\\nMessage: $user_input",
      "output_key": "classification"
    },
    {
      "id": "route",
      "type": "condition",
      "condition": "'urgent' in '$classification'",
      "then": "handle_urgent",
      "otherwise": "handle_normal"
    },
    {"id": "handle_urgent", "type": "tool", "tool": "escalate"},
    {"id": "handle_normal", "type": "tool", "tool": "log_message"}
  ]
}

EXAMPLE 2 — parallel fetch then summarize:
User intent: "Fetch weather and news in parallel, then write a morning briefing"
Output:
{
  "name": "morning_briefing",
  "description": "Fetch weather and news in parallel, then summarize",
  "steps": [
    {
      "id": "fetch",
      "type": "parallel",
      "output_key": "fetched",
      "parallel_steps": [
        {"id": "weather", "type": "tool", "tool": "get_weather"},
        {"id": "news",    "type": "tool", "tool": "get_news"}
      ]
    },
    {
      "id": "briefing",
      "type": "llm",
      "prompt": "Write a short morning briefing.\\nWeather: $weather\\nNews: $news",
      "output_key": "result"
    }
  ]
}
---

If a previous attempt failed validation, a VALIDATION ERROR will be shown.
Fix only what the error describes. Do not change unrelated parts of the program.
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """
    Converts natural language intent → validated Program in one LLM call.

    Args:
        llm:         Any LLMAdapter-compatible object (Protocol: async complete()).
        max_retries: Max additional attempts after first failure (default 2 → up to 3 total).
        temperature: Passed to adapter. Default 0.0 for reproducibility.
    """

    def __init__(
        self,
        llm: Any,
        max_retries: int = 2,
        temperature: float = 0.0,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._temperature = temperature

    async def generate(
        self,
        intent: str,
        available_tools: list[str] | None = None,
        context_keys: list[str] | None = None,
    ) -> Program:
        """
        Generate a Program from natural language intent.

        Args:
            intent:          What the program should do (plain text).
            available_tools: Optional list of registered tool names — injected into prompt.
            context_keys:    Optional list of expected context variable names.

        Returns:
            Validated Program ready for ExecutionVM.run().

        Raises:
            PlannerError: If all retries fail (ValidationError or JSON parse error).
        """
        user_prompt = self._build_user_prompt(intent, available_tools, context_keys)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        last_raw = ""
        last_error = ""

        for attempt in range(1 + self._max_retries):
            if attempt > 0:
                # Append validation error as assistant/user feedback pair
                messages = self._append_feedback(messages, last_raw, last_error)

            raw = await self._call_llm(messages)
            last_raw = raw

            # Parse JSON
            try:
                data = _extract_json(raw)
            except ValueError as exc:
                last_error = f"JSON parse error: {exc}. Raw output was not valid JSON."
                continue

            # Validate against Program schema
            try:
                return Program.from_dict(data)
            except ValidationError as exc:
                last_error = f"Schema validation error: {exc}"
                continue

        raise PlannerError(
            f"Failed to generate a valid Program after {1 + self._max_retries} attempts. "
            f"Last error: {last_error}",
            last_raw=last_raw,
            attempts=1 + self._max_retries,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call adapter, handle both str and tuple[str, ...] return signatures."""
        result = await self._llm.complete(messages, temperature=self._temperature)
        # LiteLLMAdapter returns tuple[str, dict|None]; Protocol returns str
        if isinstance(result, tuple):
            return result[0]
        return result

    @staticmethod
    def _build_user_prompt(
        intent: str,
        available_tools: list[str] | None,
        context_keys: list[str] | None,
    ) -> str:
        parts = [f"User intent: {intent}"]
        if available_tools:
            parts.append(f"Available tools: {', '.join(available_tools)}")
        if context_keys:
            parts.append(f"Context variables available: {', '.join('$' + k for k in context_keys)}")
        return "\n".join(parts)

    @staticmethod
    def _append_feedback(
        messages: list[dict[str, str]],
        last_raw: str,
        error: str,
    ) -> list[dict[str, str]]:
        """Inject previous failed output + validation error back into conversation."""
        return [
            *messages,
            {"role": "assistant", "content": last_raw},
            {
                "role": "user",
                "content": (
                    f"VALIDATION ERROR: {error}\n\n"
                    "Fix the JSON and return a corrected program. "
                    "Output only the corrected JSON object."
                ),
            },
        ]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def _extract_json(raw: str) -> dict[str, Any]:
    """
    Extract a JSON object from LLM output.

    Handles:
    - Clean JSON output (ideal case)
    - JSON wrapped in ```json ... ``` or ``` ... ``` fences
    - Leading/trailing whitespace or explanation text

    Raises ValueError if no valid JSON object found.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        # Find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM output")
        text = text[start : end + 1]

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result).__name__}")

    return result
