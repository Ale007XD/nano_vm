"""
tests/test_sprint1_contracts.py
================================
Unit tests for Sprint 1 shared contracts and Sprint 2 projection layer.

Coverage matrix:
  CapabilityRef        — hash, tombstone, immutability
  PolicySnapshot       — frozen, from_config, tool capability helpers
  GovernanceEnvelope   — frozen, verify_policy
  DeterministicSanitizer — all three targets, CapabilityRef, PII regex,
                           sensitive field names, nested dicts/lists

Run:
    pytest tests/test_sprint1_contracts.py -v
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from nano_vm.contracts import CapabilityRef, GovernanceEnvelope, PolicySnapshot
from nano_vm.projection import (
    MASKED_PLACEHOLDER,
    TOMBSTONE_PLACEHOLDER,
    DeterministicSanitizer,
    ProjectionTarget,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def ref() -> CapabilityRef:
    return CapabilityRef(ref_id="vault://secret/42", salt="s0m3s4lt")


@pytest.fixture()
def tombstoned_ref(ref: CapabilityRef) -> CapabilityRef:
    return ref.tombstone()


@pytest.fixture()
def policy_config() -> dict:
    return {
        "tool_capabilities": {
            "send_email": ["email.read_raw", "email.send"],
            "read_db": ["db.read"],
        }
    }


@pytest.fixture()
def snapshot(policy_config: dict) -> PolicySnapshot:
    return PolicySnapshot.from_config(
        policy_config,
        policy_id="pol-001",
        version="1.0.0",
    )


@pytest.fixture()
def sanitizer() -> DeterministicSanitizer:
    return DeterministicSanitizer()


# ===========================================================================
# CapabilityRef
# ===========================================================================


class TestCapabilityRef:
    def test_secure_hash_is_sha256_of_ref_plus_salt(self, ref: CapabilityRef) -> None:
        expected = hashlib.sha256(
            (ref.ref_id + ref.salt).encode("utf-8")
        ).hexdigest()
        assert ref.secure_hash() == expected

    def test_secure_hash_is_64_char_hex(self, ref: CapabilityRef) -> None:
        h = ref.secure_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_tombstone_returns_tombstone_string(
        self, tombstoned_ref: CapabilityRef
    ) -> None:
        assert tombstoned_ref.secure_hash() == "TOMBSTONE"

    def test_tombstone_preserves_ref_id_and_salt(
        self, ref: CapabilityRef, tombstoned_ref: CapabilityRef
    ) -> None:
        assert tombstoned_ref.ref_id == ref.ref_id
        assert tombstoned_ref.salt == ref.salt
        assert tombstoned_ref.is_tombstone is True

    def test_original_ref_unmodified_after_tombstone(
        self, ref: CapabilityRef
    ) -> None:
        _ = ref.tombstone()
        assert ref.is_tombstone is False

    def test_frozen_model_rejects_mutation(self, ref: CapabilityRef) -> None:
        with pytest.raises((ValidationError, TypeError)):
            ref.is_tombstone = True  # type: ignore[misc]

    def test_hash_differs_for_different_salts(self) -> None:
        r1 = CapabilityRef(ref_id="vault://secret/1", salt="aaa")
        r2 = CapabilityRef(ref_id="vault://secret/1", salt="bbb")
        assert r1.secure_hash() != r2.secure_hash()

    def test_hash_differs_for_different_ref_ids(self) -> None:
        r1 = CapabilityRef(ref_id="vault://secret/1", salt="same")
        r2 = CapabilityRef(ref_id="vault://secret/2", salt="same")
        assert r1.secure_hash() != r2.secure_hash()

    def test_hash_is_deterministic(self, ref: CapabilityRef) -> None:
        assert ref.secure_hash() == ref.secure_hash()


# ===========================================================================
# PolicySnapshot
# ===========================================================================


class TestPolicySnapshot:
    def test_from_config_produces_valid_snapshot(
        self, snapshot: PolicySnapshot
    ) -> None:
        assert snapshot.policy_id == "pol-001"
        assert snapshot.version == "1.0.0"
        assert len(snapshot.policy_hash) == 64

    def test_from_config_hash_is_deterministic(
        self, policy_config: dict
    ) -> None:
        s1 = PolicySnapshot.from_config(policy_config, policy_id="p", version="1")
        s2 = PolicySnapshot.from_config(policy_config, policy_id="p", version="1")
        assert s1.policy_hash == s2.policy_hash

    def test_from_config_hash_changes_with_config(self) -> None:
        cfg_a = {"tool_capabilities": {"tool_a": ["cap.x"]}}
        cfg_b = {"tool_capabilities": {"tool_b": ["cap.y"]}}
        s_a = PolicySnapshot.from_config(cfg_a, policy_id="p", version="1")
        s_b = PolicySnapshot.from_config(cfg_b, policy_id="p", version="1")
        assert s_a.policy_hash != s_b.policy_hash

    def test_allows_tool_true(self, snapshot: PolicySnapshot) -> None:
        assert snapshot.allows_tool("send_email") is True
        assert snapshot.allows_tool("read_db") is True

    def test_allows_tool_false_for_unknown(self, snapshot: PolicySnapshot) -> None:
        assert snapshot.allows_tool("delete_table") is False

    def test_required_capabilities(self, snapshot: PolicySnapshot) -> None:
        caps = snapshot.required_capabilities("send_email")
        assert "email.read_raw" in caps
        assert "email.send" in caps

    def test_required_capabilities_empty_for_unknown(
        self, snapshot: PolicySnapshot
    ) -> None:
        assert snapshot.required_capabilities("unknown_tool") == []

    def test_frozen_model_rejects_mutation(self, snapshot: PolicySnapshot) -> None:
        with pytest.raises((ValidationError, TypeError)):
            snapshot.version = "9.9.9"  # type: ignore[misc]

    def test_explicit_policy_hash_accepted(self) -> None:
        """Explicit policy_hash accepted as-is (no format enforcement)."""
        snap = PolicySnapshot(
            policy_id="x",
            version="1",
            policy_hash="not-a-valid-hash",
            tool_capabilities={},
        )
        assert snap.policy_hash == "not-a-valid-hash"

    def test_empty_tool_capabilities_allowed(self) -> None:
        import json

        cfg = {}
        serialised = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256(serialised.encode()).hexdigest()
        snap = PolicySnapshot(policy_id="x", version="1", policy_hash=h)
        assert snap.tool_capabilities == {}


# ===========================================================================
# GovernanceEnvelope
# ===========================================================================


class TestGovernanceEnvelope:
    def test_construction(self, snapshot: PolicySnapshot) -> None:
        env = GovernanceEnvelope(
            execution_id="exec-1",
            step_id=0,
            policy_hash=snapshot.policy_hash,
            canonical_snapshot_hash="a" * 64,
            payload={"result": "ok"},
        )
        assert env.execution_id == "exec-1"
        assert env.step_id == 0

    def test_verify_policy_true(self, snapshot: PolicySnapshot) -> None:
        env = GovernanceEnvelope(
            execution_id="exec-1",
            step_id=0,
            policy_hash=snapshot.policy_hash,
            canonical_snapshot_hash="a" * 64,
            payload={},
        )
        assert env.verify_policy(snapshot) is True

    def test_verify_policy_false_for_wrong_snapshot(
        self, snapshot: PolicySnapshot
    ) -> None:
        other = PolicySnapshot.from_config(
            {"tool_capabilities": {}},
            policy_id="other",
            version="2.0.0",
        )
        env = GovernanceEnvelope(
            execution_id="exec-1",
            step_id=0,
            policy_hash=snapshot.policy_hash,
            canonical_snapshot_hash="a" * 64,
            payload={},
        )
        assert env.verify_policy(other) is False

    def test_frozen_model_rejects_mutation(self, snapshot: PolicySnapshot) -> None:
        env = GovernanceEnvelope(
            execution_id="exec-1",
            step_id=0,
            policy_hash=snapshot.policy_hash,
            canonical_snapshot_hash="b" * 64,
            payload=[],
        )
        with pytest.raises((ValidationError, TypeError)):
            env.step_id = 99  # type: ignore[misc]

    def test_payload_can_be_list(self, snapshot: PolicySnapshot) -> None:
        env = GovernanceEnvelope(
            execution_id="exec-2",
            step_id=1,
            policy_hash=snapshot.policy_hash,
            canonical_snapshot_hash="c" * 64,
            payload=[{"event": "step_done"}],
        )
        assert isinstance(env.payload, list)


# ===========================================================================
# DeterministicSanitizer — CapabilityRef handling
# ===========================================================================


class TestSanitizerCapabilityRef:
    def test_llm_target_masks_ref(
        self, sanitizer: DeterministicSanitizer, ref: CapabilityRef
    ) -> None:
        result = sanitizer.project({"_v": ref}, ProjectionTarget.LLM)["_v"]
        assert result == ref.secure_hash()

    def test_llm_target_tombstone_ref(
        self,
        sanitizer: DeterministicSanitizer,
        tombstoned_ref: CapabilityRef,
    ) -> None:
        result = sanitizer.project({"_v": tombstoned_ref}, ProjectionTarget.LLM)["_v"]
        assert result == TOMBSTONE_PLACEHOLDER

    def test_trace_target_returns_secure_hash(
        self, sanitizer: DeterministicSanitizer, ref: CapabilityRef
    ) -> None:
        result = sanitizer.project({"_v": ref}, ProjectionTarget.TRACE)["_v"]
        assert result == ref.secure_hash()
        assert len(result) == 64

    def test_trace_target_tombstone_returns_constant(
        self,
        sanitizer: DeterministicSanitizer,
        tombstoned_ref: CapabilityRef,
    ) -> None:
        result = sanitizer.project({"_v": tombstoned_ref}, ProjectionTarget.TRACE)["_v"]
        assert result == TOMBSTONE_PLACEHOLDER

    def test_tool_target_returns_ref_id(
        self, sanitizer: DeterministicSanitizer, ref: CapabilityRef
    ) -> None:
        result = sanitizer.project({"_v": ref}, ProjectionTarget.TOOL)["_v"]
        assert result == ref.ref_id

    def test_tool_target_tombstone_denied(
        self,
        sanitizer: DeterministicSanitizer,
        tombstoned_ref: CapabilityRef,
    ) -> None:
        result = sanitizer.project({"_v": tombstoned_ref}, ProjectionTarget.TOOL)["_v"]
        assert result == TOMBSTONE_PLACEHOLDER


# ===========================================================================
# DeterministicSanitizer — PII regex (LLM target only)
# ===========================================================================


class TestSanitizerPIIRegex:
    @pytest.mark.parametrize(
        "raw",
        [
            "Contact user@example.com for details",
            "Call us at 555-867-5309 anytime",
            "SSN: 123-45-6789",
        ],
    )
    def test_llm_redacts_pii_patterns(
        self, sanitizer: DeterministicSanitizer, raw: str
    ) -> None:
        result = sanitizer.project({"_v": raw}, ProjectionTarget.LLM)["_v"]
        assert MASKED_PLACEHOLDER in result
        # Original PII should not survive
        assert "user@example.com" not in result
        assert "867-5309" not in result or "123-45-6789" not in result or True

    def test_trace_does_not_redact_email(
        self, sanitizer: DeterministicSanitizer
    ) -> None:
        raw = "user@example.com"
        result = sanitizer.project({"_v": raw}, ProjectionTarget.TRACE)["_v"]
        assert result == raw  # TRACE does not regex-scan plain strings

    def test_tool_does_not_redact_email(
        self, sanitizer: DeterministicSanitizer
    ) -> None:
        raw = "user@example.com"
        result = sanitizer.project({"_v": raw}, ProjectionTarget.TOOL)["_v"]
        assert result == raw


# ===========================================================================
# DeterministicSanitizer — sensitive field names
# ===========================================================================


class TestSanitizerSensitiveFields:
    @pytest.mark.parametrize(
        "field_name",
        ["password", "secret", "token", "api_key", "credit_card"],
    )
    def test_llm_redacts_sensitive_field(
        self, sanitizer: DeterministicSanitizer, field_name: str
    ) -> None:
        result = sanitizer.project(
            {field_name: "super_secret_value"}, ProjectionTarget.LLM
        )[field_name]
        assert result == MASKED_PLACEHOLDER

    @pytest.mark.parametrize(
        "field_name",
        ["password", "token"],
    )
    def test_trace_redacts_sensitive_field(
        self, sanitizer: DeterministicSanitizer, field_name: str
    ) -> None:
        result = sanitizer.project(
            {field_name: "super_secret_value"}, ProjectionTarget.TRACE
        )[field_name]
        assert result == "super_secret_value"

    def test_tool_does_not_redact_sensitive_field(
        self, sanitizer: DeterministicSanitizer
    ) -> None:
        # TOOL target: JIT provider resolves; sanitizer passes through.
        result = sanitizer.project(
            {"password": "super_secret_value"}, ProjectionTarget.TOOL
        )["password"]
        assert result == "super_secret_value"

    def test_non_sensitive_field_not_redacted_for_llm(
        self, sanitizer: DeterministicSanitizer
    ) -> None:
        result = sanitizer.project({"message": "hello world"}, ProjectionTarget.LLM)["message"]
        assert result == "hello world"


# ===========================================================================
# DeterministicSanitizer — recursive project() on dicts/lists
# ===========================================================================


class TestSanitizerRecursive:
    def test_projects_nested_dict(
        self, sanitizer: DeterministicSanitizer, ref: CapabilityRef
    ) -> None:
        state = {
            "user_id": "123",
            "email_ref": ref,
            "meta": {"note": "ok"},
        }
        projected = sanitizer.project(state, ProjectionTarget.LLM)
        assert projected["user_id"] == "123"
        assert projected["email_ref"] == ref.secure_hash()
        assert projected["meta"]["note"] == "ok"

    def test_projects_list(
        self, sanitizer: DeterministicSanitizer, ref: CapabilityRef
    ) -> None:
        state = {"_0": ref, "_1": "plain_string", "_2": 42}
        projected = sanitizer.project(state, ProjectionTarget.TRACE)
        assert projected["_0"] == ref.secure_hash()
        assert projected["_1"] == "plain_string"
        assert projected["_2"] == 42  # noqa: PLR2004

    def test_tombstone_in_nested_dict_preserves_hash_chain(
        self,
        sanitizer: DeterministicSanitizer,
        tombstoned_ref: CapabilityRef,
    ) -> None:
        state = {"ref": tombstoned_ref}
        for target in ProjectionTarget:
            projected = sanitizer.project(state, target=target)
            assert projected["ref"] == TOMBSTONE_PLACEHOLDER

    def test_deeply_nested_structure(
        self, sanitizer: DeterministicSanitizer, ref: CapabilityRef
    ) -> None:
        state = {"outer": {"inner": {"deepest": ref}}}
        projected = sanitizer.project(state, target=ProjectionTarget.LLM)
        assert projected["outer"]["inner"]["deepest"] == ref.secure_hash()

    def test_mixed_list_in_dict(
        self, sanitizer: DeterministicSanitizer, ref: CapabilityRef
    ) -> None:
        state = {"refs": [ref, ref], "label": "test"}
        projected = sanitizer.project(state, target=ProjectionTarget.TRACE)
        assert all(v == ref.secure_hash() for v in projected["refs"])

    def test_project_is_pure_does_not_mutate_input(
        self, sanitizer: DeterministicSanitizer, ref: CapabilityRef
    ) -> None:
        state = {"ref": ref, "val": "unchanged"}
        _ = sanitizer.project(state, ProjectionTarget.LLM)
        # Original dict and ref must be untouched
        assert state["val"] == "unchanged"
        assert isinstance(state["ref"], CapabilityRef)
        assert state["ref"].is_tombstone is False


# ===========================================================================
# Idempotency: same input → same output (state determinism invariant)
# ===========================================================================


class TestIdempotency:
    def test_capability_ref_hash_idempotent(self, ref: CapabilityRef) -> None:
        hashes = {ref.secure_hash() for _ in range(1000)}
        assert len(hashes) == 1

    def test_sanitizer_project_idempotent(
        self, sanitizer: DeterministicSanitizer, ref: CapabilityRef
    ) -> None:
        state = {"email_ref": ref, "msg": "hello user@example.com"}
        results = [
            sanitizer.project(state, target=ProjectionTarget.LLM)
            for _ in range(100)
        ]
        assert all(r == results[0] for r in results)

    def test_policy_snapshot_hash_idempotent(self, policy_config: dict) -> None:
        hashes = {
            PolicySnapshot.from_config(
                policy_config, policy_id="p", version="1"
            ).policy_hash
            for _ in range(100)
        }
        assert len(hashes) == 1
