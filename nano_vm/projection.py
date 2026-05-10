"""
nano_vm.projection
==================
Sprint 2 — RFC v0.7.0: Projection Layer.

Архитектура:
  ProjectionTarget  — enum целевых представлений: LLM | TRACE | TOOL
  AbstractProjectionLayer — ABC; реализации подменяют project()
  DeterministicSanitizer  — конкретная реализация: regex + field rules
  project()               — convenience function (singleton sanitizer)

Инварианты:
  - project() — pure function: нет I/O, нет глобального состояния.
  - Одинаковый (state, target, policy) → одинаковый результат.
  - Tombstone CapabilityRef → всегда "[REDACTED_TOMBSTONE]", независимо от target.
  - LLM target — максимальная фильтрация (только разрешённые поля + маскировка PII).
  - TRACE target — полный аудит (tombstone маскировка, остальное сохраняется).
  - TOOL target — только поля из policy.tool_capabilities для запрашиваемого tool.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from .models import CapabilityRef, PolicySnapshot, StateContext

# ---------------------------------------------------------------------------
# ProjectionTarget
# ---------------------------------------------------------------------------


class ProjectionTarget(str, Enum):
    """Целевое представление StateContext."""

    LLM = "LLM"  # промпт для LLM: максимальная фильтрация PII
    TRACE = "TRACE"  # аудит-лог: tombstone-маскировка, полная структура
    TOOL = "TOOL"  # аргументы tool-вызова: только capabilities из PolicySnapshot


# ---------------------------------------------------------------------------
# Sentinel для tombstone
# ---------------------------------------------------------------------------

_TOMBSTONE_SENTINEL = "[REDACTED_TOMBSTONE]"
_PII_SENTINEL = "[REDACTED]"

# ---------------------------------------------------------------------------
# Правила PII по умолчанию (regex → sentinel)
# ---------------------------------------------------------------------------

_DEFAULT_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # email
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), _PII_SENTINEL),
    # phone (E.164 и локальные)
    (re.compile(r"\+?\d[\d\s\-().]{7,}\d"), _PII_SENTINEL),
    # IPv4
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), _PII_SENTINEL),
]

# Поля StateContext.data которые всегда маскируются в LLM target
_SENSITIVE_FIELD_PREFIXES = (
    "password",
    "secret",
    "token",
    "api_key",
    "auth",
    "ssn",
    "credit_card",
    "card_number",
    "cvv",
    "__webhook__",  # payload от внешней системы — не в LLM промпт
)


# ---------------------------------------------------------------------------
# AbstractProjectionLayer
# ---------------------------------------------------------------------------


class AbstractProjectionLayer(ABC):
    """
    ABC для ProjectionLayer.

    Реализации обязаны определить project().
    Вызов project() должен быть детерминированным pure function.
    """

    @abstractmethod
    def project(
        self,
        state: StateContext,
        target: ProjectionTarget,
        policy: PolicySnapshot | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Проецирует StateContext в словарь для целевого представления.

        Args:
            state:     исходный StateContext (immutable).
            target:    ProjectionTarget.LLM | TRACE | TOOL.
            policy:    PolicySnapshot для target=TOOL (проверка capabilities).
                       Для LLM/TRACE может быть None.
            tool_name: имя tool для target=TOOL (фильтр по tool_capabilities).

        Returns:
            dict[str, Any] — отфильтрованное представление.
            Никогда не мутирует state.
        """

    def project_for_llm(
        self, state: StateContext, policy: PolicySnapshot | None = None
    ) -> dict[str, Any]:
        """Удобный алиас: project(state, LLM, policy)."""
        return self.project(state, ProjectionTarget.LLM, policy=policy)

    def project_for_trace(self, state: StateContext) -> dict[str, Any]:
        """Удобный алиас: project(state, TRACE)."""
        return self.project(state, ProjectionTarget.TRACE)

    def project_for_tool(
        self,
        state: StateContext,
        tool_name: str,
        policy: PolicySnapshot | None = None,
    ) -> dict[str, Any]:
        """Удобный алиас: project(state, TOOL, policy, tool_name)."""
        return self.project(
            state, ProjectionTarget.TOOL, policy=policy, tool_name=tool_name
        )


# ---------------------------------------------------------------------------
# DeterministicSanitizer
# ---------------------------------------------------------------------------


class DeterministicSanitizer(AbstractProjectionLayer):
    """
    Конкретная реализация ProjectionLayer.

    Правила фильтрации:
      LLM:
        - CapabilityRef → secure_hash() (tombstone → _TOMBSTONE_SENTINEL)
        - Поля с _SENSITIVE_FIELD_PREFIXES → _PII_SENTINEL
        - Строковые значения → regex PII scan (_DEFAULT_PII_PATTERNS)
        - __step_outputs__ включаются (нужны для $var резолвинга в промптах)

      TRACE:
        - CapabilityRef → secure_hash() (tombstone → _TOMBSTONE_SENTINEL)
        - Остальные поля — без изменений (полный аудит)
        - __webhook__ включается (нужен для forensic replay)

      TOOL:
        - Если policy is None или tool_name is None → возвращает state.data as-is
        - Иначе: только поля из policy.tool_capabilities[tool_name]
        - CapabilityRef → secure_hash() (tombstone → _TOMBSTONE_SENTINEL)

    Args:
        extra_pii_patterns: дополнительные regex-паттерны для LLM target.
        extra_sensitive_prefixes: дополнительные префиксы чувствительных полей.
    """

    def __init__(
        self,
        extra_pii_patterns: list[tuple[re.Pattern[str], str]] | None = None,
        extra_sensitive_prefixes: tuple[str, ...] | None = None,
    ) -> None:
        self._pii_patterns = list(_DEFAULT_PII_PATTERNS)
        if extra_pii_patterns:
            self._pii_patterns.extend(extra_pii_patterns)
        self._sensitive_prefixes = _SENSITIVE_FIELD_PREFIXES
        if extra_sensitive_prefixes:
            self._sensitive_prefixes = (
                _SENSITIVE_FIELD_PREFIXES + extra_sensitive_prefixes
            )

    # ------------------------------------------------------------------
    # project() — главная точка входа
    # ------------------------------------------------------------------

    def project(
        self,
        state: StateContext,
        target: ProjectionTarget,
        policy: PolicySnapshot | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        if target == ProjectionTarget.LLM:
            return self._project_llm(state)
        if target == ProjectionTarget.TRACE:
            return self._project_trace(state)
        if target == ProjectionTarget.TOOL:
            return self._project_tool(state, policy, tool_name)
        raise ValueError(f"Unknown ProjectionTarget: {target}")

    # ------------------------------------------------------------------
    # LLM projection
    # ------------------------------------------------------------------

    def _project_llm(self, state: StateContext) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for key, value in state.data.items():
            # CapabilityRef проверяется первым: tombstone имеет приоритет
            # над sensitive-prefix редактированием (иначе ssn_ref → PII_SENTINEL,
            # а не TOMBSTONE_SENTINEL, что ломает hash chain инвариант).
            if isinstance(value, CapabilityRef):
                result[key] = self._sanitize_value_llm(value)
            elif self._is_sensitive_field(key):
                result[key] = _PII_SENTINEL
            else:
                result[key] = self._sanitize_value_llm(value)

        # step_outputs нужны для $var резолвинга — включаем с sanitize
        result["__step_outputs__"] = {
            k: self._sanitize_value_llm(v) for k, v in state.step_outputs.items()
        }

        return result

    # ------------------------------------------------------------------
    # TRACE projection
    # ------------------------------------------------------------------

    def _project_trace(self, state: StateContext) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for key, value in state.data.items():
            result[key] = self._sanitize_value_trace(value)

        result["__step_outputs__"] = {
            k: self._sanitize_value_trace(v) for k, v in state.step_outputs.items()
        }

        return result

    # ------------------------------------------------------------------
    # TOOL projection
    # ------------------------------------------------------------------

    def _project_tool(
        self,
        state: StateContext,
        policy: PolicySnapshot | None,
        tool_name: str | None,
    ) -> dict[str, Any]:
        # Без policy или tool_name — возвращаем всё data (backward compat)
        if policy is None or tool_name is None:
            return {
                k: self._sanitize_value_trace(v) for k, v in state.data.items()
            }

        allowed_caps = policy.tool_capabilities.get(tool_name, [])

        # Фильтруем: только ключи, совпадающие с capabilities
        result: dict[str, Any] = {}
        for cap in allowed_caps:
            if cap in state.data:
                result[cap] = self._sanitize_value_trace(state.data[cap])
            # capability может быть составным: "email.address" → ищем в step_outputs
            if "." in cap:
                step_id, field = cap.split(".", 1)
                step_out = state.step_outputs.get(step_id)
                if isinstance(step_out, dict) and field in step_out:
                    result[cap] = self._sanitize_value_trace(step_out[field])

        return result

    # ------------------------------------------------------------------
    # Value sanitizers
    # ------------------------------------------------------------------

    def _sanitize_value_llm(self, value: Any) -> Any:
        """LLM: CapabilityRef → hash, строки → PII scan, dict/list — рекурсивно."""
        if isinstance(value, CapabilityRef):
            return _TOMBSTONE_SENTINEL if value.is_tombstone else value.secure_hash()
        if isinstance(value, str):
            return self._redact_pii(value)
        if isinstance(value, dict):
            return {k: self._sanitize_value_llm(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_value_llm(v) for v in value]
        return value

    def _sanitize_value_trace(self, value: Any) -> Any:
        """TRACE/TOOL: CapabilityRef → hash (tombstone sentinel).

        Остальное без изменений.
        """
        if isinstance(value, CapabilityRef):
            return _TOMBSTONE_SENTINEL if value.is_tombstone else value.secure_hash()
        if isinstance(value, dict):
            return {k: self._sanitize_value_trace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_value_trace(v) for v in value]
        return value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_sensitive_field(self, key: str) -> bool:
        """True если ключ начинается с чувствительного префикса."""
        key_lower = key.lower()
        return any(key_lower.startswith(prefix) for prefix in self._sensitive_prefixes)

    def _redact_pii(self, text: str) -> str:
        """Применяет все PII-паттерны к строке."""
        for pattern, sentinel in self._pii_patterns:
            text = pattern.sub(sentinel, text)
        return text


# ---------------------------------------------------------------------------
# Module-level convenience: singleton sanitizer + project()
# ---------------------------------------------------------------------------

_default_sanitizer = DeterministicSanitizer()


def project(
    state: StateContext,
    target: ProjectionTarget,
    policy: PolicySnapshot | None = None,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function: проецирует StateContext через DeterministicSanitizer.

    Эквивалент: DeterministicSanitizer().project(state, target, policy, tool_name)

    Pure function — детерминирована, нет I/O.
    """
    return _default_sanitizer.project(state, target, policy=policy, tool_name=tool_name)
