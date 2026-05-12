"""
nano_vm.projection — Sprint 2: Projection Layer
================================================
AbstractProjectionLayer — protocol / base class for all projection targets.
DeterministicSanitizer  — concrete regex + field-rule implementation.
ProjectionTarget        — enum of valid targets (LLM, TRACE, TOOL).

Design invariants (from RFC / AGENTS.md):
- project(state, target) is a pure function: same input → same output.
- CapabilityRef values are NEVER passed through raw — always masked or tombstoned.
- The TOMBSTONE constant is stable across runs to preserve hash chains.
- No I/O, no global state, no eval() in this module.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from nano_vm.contracts import CapabilityRef

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOMBSTONE_PLACEHOLDER = "[REDACTED_TOMBSTONE]"
MASKED_PLACEHOLDER = "[REDACTED]"

# Regex patterns used by DeterministicSanitizer for plain-string PII detection.
# These are intentionally conservative: false positives are safe, false negatives are not.
_PII_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # US phone
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"),  # CC#
]


# ---------------------------------------------------------------------------
# ProjectionTarget
# ---------------------------------------------------------------------------


class ProjectionTarget(str, Enum):
    """Which consumer will receive the projected data.

    LLM   — stripped to the minimum the model needs; no ref_ids, no hashes.
    TRACE — audit log; includes secure_hash() fingerprints but no plaintext.
    TOOL  — JIT access permitted; raw values resolved via RemoteProjectionProvider.
    """

    LLM = "LLM"
    TRACE = "TRACE"
    TOOL = "TOOL"


# ---------------------------------------------------------------------------
# AbstractProjectionLayer
# ---------------------------------------------------------------------------


class AbstractProjectionLayer(ABC):
    """Base class / protocol for all projection implementations.

    Subclasses must implement :meth:`project_value` which is called for every
    leaf value encountered during recursive projection of a state dict/list.
    :meth:`project` provides the recursive traversal so subclasses only need
    to handle scalar values.
    """

    @abstractmethod
    def project_value(
        self,
        value: Any,
        *,
        target: ProjectionTarget,
        field_name: str = "",
    ) -> Any:
        """Transform a single leaf value for the given target.

        Parameters
        ----------
        value:      The raw value (may be a CapabilityRef or a plain scalar).
        target:     Projection target (LLM / TRACE / TOOL).
        field_name: Optional field name hint for field-rule matching.
        """

    def project(
        self,
        state: dict[str, Any] | list[Any] | Any,
        *,
        target: ProjectionTarget,
        field_name: str = "",
    ) -> dict[str, Any] | list[Any] | Any:
        """Recursively project *state* for *target*.

        Traverses dicts and lists depth-first; delegates leaf values to
        :meth:`project_value`.
        """
        if isinstance(state, dict):
            return {
                k: self.project(v, target=target, field_name=k)
                for k, v in state.items()
            }
        if isinstance(state, list):
            return [
                self.project(item, target=target, field_name=field_name)
                for item in state
            ]
        return self.project_value(state, target=target, field_name=field_name)


# ---------------------------------------------------------------------------
# DeterministicSanitizer
# ---------------------------------------------------------------------------

# Field names that are always treated as sensitive regardless of value type.
_SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "password", "passwd", "secret", "token", "api_key", "apikey",
        "access_token", "refresh_token", "private_key", "credential",
        "ssn", "credit_card", "card_number",
    }
)


class DeterministicSanitizer(AbstractProjectionLayer):
    """Concrete projection layer: regex + field-rule based sanitisation.

    Behaviour per target:

    LLM
        CapabilityRef → ``[REDACTED]`` (tombstoned → ``[REDACTED_TOMBSTONE]``).
        Plain strings → regex scan; PII patterns replaced with ``[REDACTED]``.
        Sensitive field names → ``[REDACTED]`` regardless of value.

    TRACE
        CapabilityRef → ``secure_hash()`` result (preserves hash chain).
        Tombstoned refs → ``[REDACTED_TOMBSTONE]`` (stable constant).
        Plain strings → field-name check only (no regex — too slow for hot log path).

    TOOL
        CapabilityRef → ``ref_id`` URI (JIT access marker; actual resolution is
        done by ``RemoteProjectionProvider`` in nano_vm_mcp, not here).
        Tombstoned refs → ``[REDACTED_TOMBSTONE]`` (access denied).
        Plain strings → pass-through (tool receives raw context values).

    The sanitiser is stateless — all methods are pure functions.
    """

    def project_value(
        self,
        value: Any,
        *,
        target: ProjectionTarget,
        field_name: str = "",
    ) -> Any:
        # --- CapabilityRef handling (always takes priority) ---------------
        if isinstance(value, CapabilityRef):
            return self._project_ref(value, target=target)

        # --- Non-string scalars: field-name check only --------------------
        if not isinstance(value, str):
            if field_name.lower() in _SENSITIVE_FIELD_NAMES and target in (
                ProjectionTarget.LLM,
                ProjectionTarget.TRACE,
            ):
                return MASKED_PLACEHOLDER
            return value

        # --- String values ------------------------------------------------
        return self._project_string(value, target=target, field_name=field_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _project_ref(ref: CapabilityRef, *, target: ProjectionTarget) -> str:
        if ref.is_tombstone:
            return TOMBSTONE_PLACEHOLDER

        if target == ProjectionTarget.LLM:
            return MASKED_PLACEHOLDER

        if target == ProjectionTarget.TRACE:
            # Return the secure hash — preserves chain integrity without
            # exposing the URI or plaintext.
            return ref.secure_hash()

        if target == ProjectionTarget.TOOL:
            # Signal to RemoteProjectionProvider that JIT resolution is needed.
            return ref.ref_id

        # Unreachable — ProjectionTarget is a closed enum.
        return MASKED_PLACEHOLDER  # pragma: no cover

    @staticmethod
    def _project_string(
        value: str,
        *,
        target: ProjectionTarget,
        field_name: str,
    ) -> str:
        # Sensitive field name → always redact for LLM and TRACE targets.
        if field_name.lower() in _SENSITIVE_FIELD_NAMES and target in (
            ProjectionTarget.LLM,
            ProjectionTarget.TRACE,
        ):
            return MASKED_PLACEHOLDER

        # LLM target: run full regex scan for PII patterns.
        if target == ProjectionTarget.LLM:
            result = value
            for pattern in _PII_PATTERNS:
                result = pattern.sub(MASKED_PLACEHOLDER, result)
            return result

        # TRACE / TOOL: no regex scan (performance + determinism).
        # Field-name check already handled above for TRACE.
        return value
