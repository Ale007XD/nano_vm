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

    Пример кастомной реализации:

        class MyAdapter:
            async def complete(self, messages, **kwargs) -> str:
                # твой HTTP-клиент
                return "ответ"

        vm = ExecutionVM(llm=MyAdapter())
    """

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Отправить список сообщений в LLM, вернуть текст ответа.

        Args:
            messages: список dict с ключами 'role' и 'content'
                      [{"role": "user", "content": "..."}]
            **kwargs: дополнительные параметры (temperature, max_tokens и т.д.)

        Returns:
            Текст ответа модели (строка).
        """
        ...
