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
    Vibecode:    "openai/claude-sonnet-4.6"  (stream=True required)
"""

from __future__ import annotations

from typing import Any

try:
    from litellm import acompletion
except ImportError:
    raise ImportError("litellm не установлен. Выполни: pip install nano-vm[litellm]")


class LiteLLMAdapter:
    """
    LLM-адаптер через litellm.

    Args:
        model:       строка провайдер/модель, например "groq/llama-3.3-70b-versatile"
        fallbacks:   список резервных моделей, litellm переключается автоматически
        timeout:     таймаут запроса в секундах
        max_retries: количество повторов при ошибке провайдера
        temperature: температура генерации (0.0 — детерминировано)
        stream:      включить streaming (обязательно для Vibecode/прокси с таймаутом)
        max_tokens:  максимальное количество токенов в ответе (None = дефолт провайдера)
        **kwargs:    любые дополнительные параметры litellm.acompletion
    """

    def __init__(
        self,
        model: str,
        fallbacks: list[str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        stream: bool = False,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.fallbacks = fallbacks
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.stream = stream
        self.max_tokens = max_tokens
        self._extra = kwargs

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Вызов LLM через litellm. Возвращает текст ответа."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "num_retries": self.max_retries,
            "stream": self.stream,
            **self._extra,
            **kwargs,
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        if self.fallbacks:
            params["fallbacks"] = self.fallbacks

        response = await acompletion(**params)

        # Stream mode: собираем чанки в строку
        if self.stream or params.get("stream"):
            text_parts: list[str] = []
            async for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    text_parts.append(delta.content)
            return "".join(text_parts)

        # Non-stream mode
        text: str = response.choices[0].message.content
        return text

    def __repr__(self) -> str:
        parts = [f"model={self.model!r}"]
        if self.fallbacks:
            parts.append(f"fallbacks={self.fallbacks!r}")
        if self.stream:
            parts.append("stream=True")
        if self.max_tokens is not None:
            parts.append(f"max_tokens={self.max_tokens}")
        return f"LiteLLMAdapter({', '.join(parts)})"
