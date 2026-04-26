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
    ) -> tuple[str, dict | None]:
        """Вызов LLM через litellm. Возвращает (text, usage_dict) или (text, None)."""
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
        text = response.choices[0].message.content

        usage_dict: dict | None = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(raw_usage, "completion_tokens", 0) or 0
            total_tokens = getattr(raw_usage, "total_tokens", 0) or (
                prompt_tokens + completion_tokens
            )
            cost_usd: float | None = None
            hidden = getattr(response, "_hidden_params", {}) or {}
            if "response_cost" in hidden and hidden["response_cost"] is not None:
                cost_usd = float(hidden["response_cost"])
            usage_dict = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
            }

        return text, usage_dict

    def __repr__(self) -> str:
        parts = [f"model={self.model!r}"]
        if self.fallbacks:
            parts.append(f"fallbacks={self.fallbacks!r}")
        return f"LiteLLMAdapter({', '.join(parts)})"
