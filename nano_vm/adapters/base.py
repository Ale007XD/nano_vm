"""
nano_vm.adapters.base
=====================
Контракт адаптера. VM знает только этот интерфейс.
Реализуй его — и любой LLM-провайдер будет работать с nano-vm.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMAdapter(Protocol):
    """
    Минимальный контракт для LLM-адаптера.

    complete() возвращает либо строку (legacy / кастомные адаптеры),
    либо tuple[str, dict | None] (встроенные адаптеры с usage-данными).
    ExecutionVM обрабатывает оба варианта через isinstance(result, tuple).

    Пример кастомной реализации (минимальная — только str):

        class MyAdapter:
            async def complete(self, messages, **kwargs) -> str:
                # твой HTTP-клиент
                return "ответ"

        vm = ExecutionVM(llm=MyAdapter())

    Пример с usage:

        class MyAdapter:
            async def complete(
                self, messages, **kwargs
            ) -> tuple[str, dict[str, Any] | None]:
                text = ...
                usage = {"prompt_tokens": 10, "completion_tokens": 5,
                         "total_tokens": 15, "cost_usd": None}
                return text, usage
    """

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str | tuple[str, dict[str, Any] | None]:
        """
        Отправить список сообщений в LLM, вернуть ответ.

        Args:
            messages: список dict с ключами 'role' и 'content'
                      [{"role": "user", "content": "..."}]
            **kwargs: дополнительные параметры (temperature, max_tokens и т.д.)

        Returns:
            str — текст ответа (legacy / кастомные адаптеры).
            tuple[str, dict | None] — текст + usage_dict с ключами
            prompt_tokens, completion_tokens, total_tokens, cost_usd.
        """
        ...
