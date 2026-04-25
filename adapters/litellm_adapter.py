"""
nano_vm.adapters.litellm_adapter
=================================
Адаптер на основе litellm — единый интерфейс ко всем провайдерам.

Установка:
    pip install nano-vm[litellm]

Поддерживаемые провайдеры (примеры model string):
    Groq:        "groq/llama-3.3-70b-versatile"
    Anthropic:   "anthropic/claude-sonnet-4-20250514"
    OpenAI:      "openai/gpt-4o"
    OpenRouter:  "openrouter/llama-3.3-70b-instruct:free"
    Ollama:      "ollama/llama3"
"""

from __future__ import annotations

from typing import Any

try:
    import litellm
    from litellm import acompletion
except ImportError:
    raise ImportError(
        "litellm не установлен. Выполни: pip install nano-vm[litellm]"
    )


class LiteLLMAdapter:
    """
    LLM-адаптер через litellm.

    Args:
        model:       строка провайдер/модель, например "groq/llama-3.3-70b-versatile"
        fallbacks:   список резервных моделей, litellm переключается автоматически
        timeout:     таймаут запроса в секундах
        max_retries: количество повторов при ошибке провайдера
        temperature: температура генерации (0.0 — детерминировано)
        **kwargs:    любые дополнительные параметры litellm.acompletion
    """

    def __init__(
        self,
        model: str,
        fallbacks: list[str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.fallbacks = fallbacks
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self._extra = kwargs

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Вызов LLM через litellm, возвращает текст ответа."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "num_retries": self.max_retries,
            **self._extra,
            **kwargs,  # kwargs из вызова имеют приоритет
        }

        if self.fallbacks:
            params["fallbacks"] = self.fallbacks

        response = await acompletion(**params)
        return response.choices[0].message.content

    def __repr__(self) -> str:
        parts = [f"model={self.model!r}"]
        if self.fallbacks:
            parts.append(f"fallbacks={self.fallbacks!r}")
        return f"LiteLLMAdapter({', '.join(parts)})"
