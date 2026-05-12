"""
nano_vm.contracts — Sprint 1: Shared Contracts
===============================================
CapabilityRef   — replaces raw PII in CanonicalState.
PolicySnapshot  — immutable rule snapshot per session (frozen Pydantic model).
GovernanceEnvelope — wrapper for outgoing MCP data.

Design invariants (from RFC / AGENTS.md):
- CapabilityRef.secure_hash() is a pure function: sha256(ref_id + salt) or "TOMBSTONE".
- PolicySnapshot is frozen — no mutation after construction.
- GovernanceEnvelope carries a canonical_snapshot_hash for audit trail integrity.
- No I/O, no global state, no eval() anywhere in this module.
"""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# CapabilityRef
# ---------------------------------------------------------------------------


class CapabilityRef(BaseModel):
    """Opaque reference that replaces raw PII values in CanonicalState.

    The ref_id is a URI pointing to the actual secret in an external vault
    (e.g. ``vault://secret/123``).  The salt makes the hash unguessable even
    when the URI scheme is predictable.

    ``secure_hash()`` is intentionally a pure function with no side-effects so
    that it can be called arbitrarily during Merkle/Delta hash-chain computation
    without risk of accidental state mutation.
    """

    ref_id: str = Field(
        ...,
        description="URI identifying the secret in an external vault, e.g. 'vault://secret/123'.",
    )
    salt: str = Field(
        ...,
        description="Per-ref random salt used in salted hashing to prevent URI enumeration.",
    )
    is_tombstone: bool = Field(
        default=False,
        description="When True the ref has been GDPR-erased. secure_hash() returns 'TOMBSTONE'.",
    )

    model_config = {"frozen": True}

    def secure_hash(self) -> str:
        """Return the canonical hash for this ref.

        Returns
        -------
        str
            ``"TOMBSTONE"`` if ``is_tombstone`` is ``True``, otherwise the
            hex-encoded SHA-256 digest of ``ref_id + salt``.  The result is
            stable across Python versions because we encode to UTF-8 before
            hashing.
        """
        if self.is_tombstone:
            return "TOMBSTONE"
        payload = (self.ref_id + self.salt).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def tombstone(self) -> CapabilityRef:
        """Return a new CapabilityRef with is_tombstone=True (model is frozen).

        Triggered by the E_gdpr_erase system event.  The original ref_id and
        salt are preserved so that the hash chain can be verified post-erasure.
        """
        return self.model_copy(update={"is_tombstone": True})


# ---------------------------------------------------------------------------
# PolicySnapshot
# ---------------------------------------------------------------------------


class PolicySnapshot(BaseModel):
    """Immutable policy snapshot captured at session start.

    ``policy_hash`` is the SHA-256 of the serialised config dict that produced
    this snapshot.  It is stored in every ``GovernanceEnvelope`` so that
    auditors can verify which policy version governed a given execution step.

    ``tool_capabilities`` maps tool names to the capability strings they
    require, e.g.::

        {"send_email": ["email.read_raw", "email.send"]}

    The ``GovernedToolExecutor`` (Sprint 2 / nano_vm_mcp) checks membership in
    this dict before allowing any tool call.
    """

    policy_id: str = Field(..., description="Unique identifier for the policy definition.")
    version: str = Field(..., description="Semantic version string, e.g. '1.0.0'.")
    policy_hash: str = Field(
        ...,
        description="SHA-256 hex digest of the serialised policy config.",
    )
    tool_capabilities: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Maps tool_name → list[capability_string] required to invoke the tool.",
    )

    model_config = {"frozen": True}

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def allows_tool(self, tool_name: str) -> bool:
        """Return True if ``tool_name`` appears in the capability registry."""
        return tool_name in self.tool_capabilities

    def required_capabilities(self, tool_name: str) -> list[str]:
        """Return capability list for ``tool_name``, or empty list if absent."""
        return list(self.tool_capabilities.get(tool_name, []))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        policy_id: str,
        version: str,
    ) -> PolicySnapshot:
        """Build a PolicySnapshot from a raw config dict.

        The ``policy_hash`` is computed deterministically from the JSON
        representation of *config* (keys sorted, no whitespace).

        Parameters
        ----------
        config:
            Raw policy configuration dict.  Must contain at minimum a
            ``"tool_capabilities"`` key.
        policy_id:
            Unique identifier for the policy.
        version:
            Semantic version string.
        """
        import json

        serialised = json.dumps(config, sort_keys=True, separators=(",", ":"))
        policy_hash = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
        return cls(
            policy_id=policy_id,
            version=version,
            policy_hash=policy_hash,
            tool_capabilities=config.get("tool_capabilities", {}),
        )

    @model_validator(mode="after")
    def _validate_policy_hash_format(self) -> PolicySnapshot:
        if len(self.policy_hash) != 64 or not all(  # noqa: PLR2004
            c in "0123456789abcdef" for c in self.policy_hash
        ):
            raise ValueError(
                f"policy_hash must be a 64-char lowercase hex string, got: {self.policy_hash!r}"
            )
        return self


# ---------------------------------------------------------------------------
# GovernanceEnvelope
# ---------------------------------------------------------------------------


class GovernanceEnvelope(BaseModel):
    """Wrapper for all outgoing MCP data from the Gateway.

    Every response from ``POST /mcp/session/{execution_id}/step`` is wrapped in
    a GovernanceEnvelope.  The ``canonical_snapshot_hash`` ties the payload to
    a specific state in the Merkle/Delta hash chain, enabling deterministic
    replay verification.

    ``payload`` holds the *projected* data (already sanitised by the
    ``ProjectionLayer`` for the TRACE target) — never raw PII.
    """

    execution_id: str = Field(..., description="UUID identifying the execution session.")
    step_id: int = Field(..., description="Zero-based index of the step within the execution.")
    policy_hash: str = Field(
        ...,
        description="SHA-256 of the PolicySnapshot that governed this step.",
    )
    canonical_snapshot_hash: str = Field(
        ...,
        description="SHA-256 of the CanonicalState after this step (Merkle/Delta chain node).",
    )
    payload: dict[str, Any] | list[Any] = Field(
        ...,
        description=(
            "Projected (sanitised) step result"
            "— safe for TRACE storage and external delivery.",
        ),
    )

    model_config = {"frozen": True}

    def verify_policy(self, snapshot: PolicySnapshot) -> bool:
        """Return True if this envelope was produced under *snapshot*."""
        return self.policy_hash == snapshot.policy_hash
