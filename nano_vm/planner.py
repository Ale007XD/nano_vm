"""
nano_vm.planner
===============
Planner — единственное место недетерминизма.
Принимает свободный текст, делает один LLM-вызов, возвращает Program.

Исполнение Program — всегда детерминировано (ExecutionVM).
"""

from __future__ import annotations

import json
import re
from typing import Any

from .adapters.base import LLMAdapter
from .models import Program

SYSTEM_PROMPT = """\
Ты — планировщик задач. Твоя цель: преобразовать запрос пользователя в JSON-программу.

Формат программы:
{{
  "name": "краткое название",
  "description": "что делает программа",
  "steps": [
    {{
      "id": "step_1",
      "type": "llm",
      "prompt": "текст промпта, можно использовать $переменные из context",
      "output_key": "имя_ключа"
    }},
    {{
      "id": "step_2",
      "type": "tool",
      "tool": "имя_инструмента",
      "args": {{"arg1": "$step_1.output"}}
    }},
    {{
      "id": "step_3",
      "type": "condition",
      "condition": "$step_2.output == 'yes'",
      "then": "step_4",
      "otherwise": "step_5"
    }}
  ]
}}

Правила:
- Используй только шаги типов: llm, tool, condition
- Ссылки на предыдущие шаги: $step_id.output
- Ссылки на входные данные: $имя_переменной
- Отвечай ТОЛЬКО валидным JSON, без комментариев и markdown-блоков
- Доступные инструменты: {tools}
"""


class PlannerError(Exception):
    """Ошибка генерации программы."""


class Planner:
    """
    Генерирует Program из свободного текста пользователя.

    Args:
        llm:   адаптер языковой модели
        tools: список имён доступных инструментов (для системного промпта)

    Пример:
        planner = Planner(llm=adapter, tools=["search", "send_email"])
        program = await planner.generate("Найди последние новости по AI и отправь на почту")
        trace = await vm.run(program, context={})
    """

    def __init__(
        self,
        llm: LLMAdapter,
        tools: list[str] | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools or []

    async def generate(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
    ) -> Program:
        """
        Сгенерировать Program из запроса пользователя.

        Args:
            user_input: свободный текст запроса
            context:    доступные переменные (используются для подсказки в промпте)

        Returns:
            Program — готова к исполнению в ExecutionVM.

        Raises:
            PlannerError: если LLM вернул невалидный JSON или невалидную программу.
        """
        system = SYSTEM_PROMPT.format(tools=", ".join(self._tools) if self._tools else "нет")

        user_msg = user_input
        if context:
            keys = ", ".join(f"${k}" for k in context.keys())
            user_msg = f"{user_input}\n\nДоступные переменные в context: {keys}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

        raw = await self._llm.complete(messages)
        return self._parse_program(raw)

    # ------------------------------------------------------------------
    # Парсинг ответа
    # ------------------------------------------------------------------

    def _parse_program(self, raw: str) -> Program:
        """Извлечь JSON из ответа LLM и валидировать как Program."""
        json_str = self._extract_json(raw)
        if not json_str:
            raise PlannerError(f"LLM не вернул валидный JSON. Ответ:\n{raw[:500]}")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise PlannerError(f"Ошибка парсинга JSON: {exc}\nСырой ответ:\n{raw[:500]}") from exc

        try:
            return Program.model_validate(data)
        except Exception as exc:
            raise PlannerError(f"Невалидная структура программы: {exc}") from exc

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """
        Извлечь JSON из текста.
        Обрабатывает как чистый JSON, так и JSON в markdown-блоках.
        """
        text = text.strip()

        # Убрать ```json ... ``` или ``` ... ```
        md_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if md_match:
            return md_match.group(1).strip()

        # Найти первый { ... } блок
        brace_match = re.search(r"\{[\s\S]+\}", text)
        if brace_match:
            return brace_match.group(0)

        return None
