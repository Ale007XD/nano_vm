"""
nano_vm.projection — Sprint 2: Projection Layer
================================================
AbstractProjectionLayer — ABC for all projection targets.
DeterministicSanitizer  — concrete regex + field-rule implementation.
ProjectionTarget        — enum of valid targets (LLM, TRACE, TOOL).
project()               — module-level convenience function.

API contract (from test_v070_sprint2_projection.py):
- sanitizer.project(state: StateContext, target, *, policy=None, tool_name=None)
- sanitizer.project_for_llm(state)
- sanitizer.project_for_trace(state)
- sanitizer.project_for_tool(state, tool_name, *, policy=None)
- DeterministicSanitizer(extra_pii_patterns=..., extra_sensitive_prefixes=...)
- module-level project(state, target, *, policy=None, tool_name=None)
- _PII_SENTINEL / _TOMBSTONE_SENTINEL — stable string aliases for assertions

Design invariants (from RFC / AGENTS.md):
- project() is a pure function: same input -> same output.
- CapabilityRef values are never passed through raw.
- No I/O, no global state, no eval() in this module.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from nano_vm.contracts import CapabilityRef, PolicySnapshot

# ---------------------------------------------------------------------------
# Constants & sentinels
# ---------------------------------------------------------------------------

TOMBSTONE_PLACEHOLDER = "[REDACTED_TOMBSTONE]"
MASKED_PLACEHOLDER = "[REDACTED]"

# Stable string aliases exported for test assertions.
# String aliases (not object() sentinels) — they ARE the output values,
# so tests assert result == _PII_SENTINEL without hardcoding the literal.
_PII_SENTINEL = MASKED_PLACEHOLDER
_TOMBSTONE_SENTINEL = TOMBSTONE_PLACEHOLDER

# ---------------------------------------------------------------------------
# Regex patterns for PII detection (LLM target only)
# ---------------------------------------------------------------------------

_DEFAULT_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        MASKED_PLACEHOLDER,
    ),
    (
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        MASKED_PLACEHOLDER,
    ),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), MASKED_PLACEHOLDER),
    (
        re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),
        MASKED_PLACEHOLDER,
    ),
]

_DEFAULT_SENSITIVE_PREFIXES: tuple[str, ...] = (
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "access_token", "refresh_token", "private_key", "credential",
    "ssn", "credit_card", "card_number",
)

# StateContext field always redacted for LLM target.
_WEBHOOK_FIELD = "__webhook__"


# ---------------------------------------------------------------------------
# ProjectionTarget
# ---------------------------------------------------------------------------


class ProjectionTarget(str, Enum):
    """Which consumer will receive the projected data.

    LLM   -- minimum needed by the model; PII redacted, refs hashed.
    TRACE -- audit log; refs hashed, tombstones marked, PII preserved.
    TOOL  -- capability-filtered view; refs resolved via JIT provider.
    """

    LLM = "LLM"
    TRACE = "TRACE"
    TOOL = "TOOL"


# ---------------------------------------------------------------------------
# AbstractProjectionLayer
# ---------------------------------------------------------------------------


class AbstractProjectionLayer(ABC):
    """Base class for all projection implementations.

    Subclasses must implement :meth:`project`.
    Convenience methods ``project_for_llm``, ``project_for_trace``, and
    ``project_for_tool`` delegate to :meth:`project` so subclasses get them
    for free.
    """

    @abstractmethod
    def project(
        self,
        state: Any,
        target: ProjectionTarget,
        policy: PolicySnapshot | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        """Project *state* for *target*.

        Parameters
        ----------
        state:     StateContext or compatible mapping.
        target:    Projection target.
        policy:    Required for TOOL target capability filtering.
        tool_name: Required for TOOL target capability filtering.
        """

    def project_for_llm(self, state: Any) -> dict[str, Any]:
        """Project *state* for LLM consumption."""
        return self.project(state, ProjectionTarget.LLM)

    def project_for_trace(self, state: Any) -> dict[str, Any]:
        """Project *state* for audit trace storage."""
        return self.project(state, ProjectionTarget.TRACE)

    def project_for_tool(
        self,
        state: Any,
        tool_name: str | None = None,
        *,
        policy: PolicySnapshot | None = None,
    ) -> dict[str, Any]:
        """Project *state* for tool execution (capability-filtered)."""
        return self.project(
            state, ProjectionTarget.TOOL, policy=policy, tool_name=tool_name
        )


# ---------------------------------------------------------------------------
# DeterministicSanitizer
# ---------------------------------------------------------------------------


class DeterministicSanitizer(AbstractProjectionLayer):
    """Concrete projection layer: regex + field-rule based sanitisation.

    Parameters
    ----------
    extra_pii_patterns:
        Additional ``(compiled_pattern, replacement_str)`` tuples appended to
        the default PII pattern list. Applied during LLM projection only.
    extra_sensitive_prefixes:
        Additional field-name prefixes treated as sensitive for LLM projection.

    Behaviour per target
    --------------------
    LLM
        - ``__webhook__`` field -> ``[REDACTED]``
        - CapabilityRef (live) -> ``secure_hash()``
        - CapabilityRef (tombstone) -> ``[REDACTED_TOMBSTONE]``
        - Sensitive field name -> ``[REDACTED]``
        - Plain string -> regex PII scan
        - ``step_outputs`` included under ``__step_outputs__`` key

    TRACE
        - CapabilityRef (live) -> ``secure_hash()``
        - CapabilityRef (tombstone) -> ``[REDACTED_TOMBSTONE]``
        - All other values pass through unchanged
        - ``step_outputs`` included under ``__step_outputs__`` key
        - ``__webhook__`` field included as-is

    TOOL
        - Only keys matching ``policy.tool_capabilities[tool_name]`` returned
        - If ``policy`` or ``tool_name`` is None -> all data returned
        - If ``tool_name`` not in policy -> empty dict
        - CapabilityRef (tombstone) -> ``[REDACTED_TOMBSTONE]``
        - CapabilityRef (live) -> ``ref_id`` URI (JIT resolution by caller)
    """

    def __init__(
        self,
        extra_pii_patterns: list[tuple[re.Pattern[str], str]] | None = None,
        extra_sensitive_prefixes: tuple[str, ...] | None = None,
    ) -> None:
        self._pii_patterns = list(_DEFAULT_PII_PATTERNS)
        if extra_pii_patterns:
            self._pii_patterns.extend(extra_pii_patterns)
        self._sensitive_prefixes: frozenset[str] = frozenset(
            _DEFAULT_SENSITIVE_PREFIXES
        ) | frozenset(extra_sensitive_prefixes or ())

    # ------------------------------------------------------------------
    # AbstractProjectionLayer implementation
    # ------------------------------------------------------------------

    def project(
        self,
        state: Any,
        target: ProjectionTarget,
        policy: PolicySnapshot | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        """Project *state* for *target*. Returns a new dict — input is unchanged."""
        data: dict[str, Any] = dict(getattr(state, "data", state) or {})
        step_outputs: dict[str, Any] = dict(getattr(state, "step_outputs", None) or {})

        if target == ProjectionTarget.LLM:
            return self._project_llm(data, step_outputs)
        if target == ProjectionTarget.TRACE:
            return self._project_trace(data, step_outputs)
        # TOOL
        return self._project_tool(data, policy=policy, tool_name=tool_name)

    # ------------------------------------------------------------------
    # Target implementations
    # ------------------------------------------------------------------

    def _project_llm(
        self,
        data: dict[str, Any],
        step_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in data.items():
            if key == _WEBHOOK_FIELD:
                result[key] = MASKED_PLACEHOLDER
            else:
                result[key] = self._sanitize_llm(value, field_name=key)
        if step_outputs:
            result["__step_outputs__"] = {
                k: self._sanitize_llm(v, field_name=k) for k, v in step_outputs.items()
            }
        return result

    def _project_trace(
        self,
        data: dict[str, Any],
        step_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {k: _sanitize_trace(v) for k, v in data.items()}
        if step_outputs:
            result["__step_outputs__"] = {
                k: _sanitize_trace(v) for k, v in step_outputs.items()
            }
        return result

    def _project_tool(
        self,
        data: dict[str, Any],
        *,
        policy: PolicySnapshot | None,
        tool_name: str | None,
    ) -> dict[str, Any]:
        if policy is None or tool_name is None:
            return {k: _sanitize_tool(v) for k, v in data.items()}
        if not policy.allows_tool(tool_name):
            return {}
        allowed = set(policy.required_capabilities(tool_name))
        return {k: _sanitize_tool(v) for k, v in data.items() if k in allowed}

    # ------------------------------------------------------------------
    # Leaf-value helper (LLM — instance method: needs self._sensitive_prefixes)
    # ------------------------------------------------------------------

    def _sanitize_llm(self, value: Any, *, field_name: str = "") -> Any:
        if isinstance(value, CapabilityRef):
            return TOMBSTONE_PLACEHOLDER if value.is_tombstone else value.secure_hash()
        if isinstance(value, dict):
            return {k: self._sanitize_llm(v, field_name=k) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_llm(item) for item in value]
        fname = field_name.lower()
        if fname in self._sensitive_prefixes or any(
            fname.startswith(p) for p in self._sensitive_prefixes
        ):
            return MASKED_PLACEHOLDER
        if not isinstance(value, str):
            return value
        result = value
        for pattern, replacement in self._pii_patterns:
            result = pattern.sub(replacement, result)
        return result


# ---------------------------------------------------------------------------
# Module-level pure helpers (TRACE / TOOL — no instance state needed)
# ---------------------------------------------------------------------------


def _sanitize_trace(value: Any) -> Any:
    if isinstance(value, CapabilityRef):
        return TOMBSTONE_PLACEHOLDER if value.is_tombstone else value.secure_hash()
    if isinstance(value, dict):
        return {k: _sanitize_trace(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_trace(item) for item in value]
    return value


def _sanitize_tool(value: Any) -> Any:
    if isinstance(value, CapabilityRef):
        return TOMBSTONE_PLACEHOLDER if value.is_tombstone else value.ref_id
    return value


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_sanitizer = DeterministicSanitizer()


def project(
    state: Any,
    target: ProjectionTarget,
    *,
    policy: PolicySnapshot | None = None,
    tool_name: str | None = None,
) -> dict[str, Any]:
    """Project *state* for *target* using the default DeterministicSanitizer."""
    return _default_sanitizer.project(state, target, policy=policy, tool_name=tool_name)
