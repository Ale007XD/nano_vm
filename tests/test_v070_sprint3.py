"""
tests/test_v070_sprint3.py  [nano_vm репо]
==========================================
Sprint 3: Assembly & Tombstoning — Core

Покрывает:
  - GdprEraseEvent: валидация, frozen=True, пустые target_ref_ids
  - VM.erase(): tombstone CapabilityRef по ref_id, счётчик, детерминизм
  - VM.erase(): не затрагивает не-CapabilityRef значения и step_outputs
  - VM.erase(): не затрагивает CapabilityRef с другим ref_id
  - Trace.canonical_snapshot_hash(): пустой список, 1/2/3/4 снимка
  - Trace.canonical_snapshot_hash(): детерминизм, append-only тампер
  - Forensic replay: erase → hash chain
"""

from __future__ import annotations

import hashlib

import pytest

from nano_vm.models import (
    CapabilityRef,
    GdprEraseEvent,
    PolicySnapshot,
    StateContext,
    Trace,
)
from nano_vm.adapters.mock_adapter import MockLLMAdapter
from nano_vm.vm import ExecutionVM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vm() -> ExecutionVM:
    return ExecutionVM(llm=MockLLMAdapter("yes"))


@pytest.fixture
def ref_email() -> CapabilityRef:
    return CapabilityRef(ref_id="vault://users/42/email", salt="salt1")


@pytest.fixture
def ref_phone() -> CapabilityRef:
    return CapabilityRef(ref_id="vault://users/42/phone", salt="salt2")


@pytest.fixture
def state_with_refs(ref_email: CapabilityRef, ref_phone: CapabilityRef) -> StateContext:
    return StateContext(
        data={
            "email_ref": ref_email,
            "phone_ref": ref_phone,
            "username": "alice",
            "score": 42,
        }
    )


# ---------------------------------------------------------------------------
# GdprEraseEvent
# ---------------------------------------------------------------------------


class TestGdprEraseEvent:
    def test_basic_construction(self) -> None:
        event = GdprEraseEvent(target_ref_ids=("vault://users/42/email",))
        assert event.target_ref_ids == ("vault://users/42/email",)
        assert event.reason == "gdpr_erasure"

    def test_custom_reason(self) -> None:
        event = GdprEraseEvent(target_ref_ids=("vault://x",), reason="legal_hold_expiry")
        assert event.reason == "legal_hold_expiry"

    def test_frozen(self) -> None:
        event = GdprEraseEvent(target_ref_ids=("vault://x",))
        with pytest.raises(Exception):
            event.reason = "mutated"  # type: ignore[misc]

    def test_empty_target_ref_ids_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            GdprEraseEvent(target_ref_ids=())

    def test_multiple_targets(self) -> None:
        event = GdprEraseEvent(target_ref_ids=("vault://a", "vault://b", "vault://c"))
        assert len(event.target_ref_ids) == 3

    def test_issued_at_timezone_aware(self) -> None:
        event = GdprEraseEvent(target_ref_ids=("vault://x",))
        assert event.issued_at.tzinfo is not None


# ---------------------------------------------------------------------------
# VM.erase()
# ---------------------------------------------------------------------------


class TestVmErase:
    def test_erase_single_ref(self, vm: ExecutionVM, state_with_refs: StateContext) -> None:
        event = GdprEraseEvent(target_ref_ids=("vault://users/42/email",))
        new_state, count = vm.erase(event, state_with_refs)

        assert count == 1
        erased = new_state.data["email_ref"]
        assert isinstance(erased, CapabilityRef)
        assert erased.is_tombstone is True
        assert erased.secure_hash() == "TOMBSTONE"

    def test_erase_preserves_untargeted_ref(
        self, vm: ExecutionVM, state_with_refs: StateContext
    ) -> None:
        event = GdprEraseEvent(target_ref_ids=("vault://users/42/email",))
        new_state, _ = vm.erase(event, state_with_refs)

        phone = new_state.data["phone_ref"]
        assert isinstance(phone, CapabilityRef)
        assert phone.is_tombstone is False

    def test_erase_preserves_non_ref_values(
        self, vm: ExecutionVM, state_with_refs: StateContext
    ) -> None:
        event = GdprEraseEvent(target_ref_ids=("vault://users/42/email",))
        new_state, _ = vm.erase(event, state_with_refs)

        assert new_state.data["username"] == "alice"
        assert new_state.data["score"] == 42

    def test_erase_multiple_targets(
        self, vm: ExecutionVM, state_with_refs: StateContext
    ) -> None:
        event = GdprEraseEvent(
            target_ref_ids=("vault://users/42/email", "vault://users/42/phone")
        )
        new_state, count = vm.erase(event, state_with_refs)

        assert count == 2
        assert new_state.data["email_ref"].is_tombstone is True
        assert new_state.data["phone_ref"].is_tombstone is True

    def test_erase_no_match_zero_count(
        self, vm: ExecutionVM, state_with_refs: StateContext
    ) -> None:
        event = GdprEraseEvent(target_ref_ids=("vault://nonexistent",))
        new_state, count = vm.erase(event, state_with_refs)

        assert count == 0
        assert new_state.data["email_ref"].is_tombstone is False

    def test_erase_does_not_mutate_original(
        self, vm: ExecutionVM, state_with_refs: StateContext
    ) -> None:
        event = GdprEraseEvent(target_ref_ids=("vault://users/42/email",))
        vm.erase(event, state_with_refs)

        original_ref = state_with_refs.data["email_ref"]
        assert isinstance(original_ref, CapabilityRef)
        assert original_ref.is_tombstone is False

    def test_erase_step_outputs_untouched(
        self, vm: ExecutionVM, ref_email: CapabilityRef
    ) -> None:
        state = StateContext(
            data={"email_ref": ref_email},
            step_outputs={"step1": "some output"},
        )
        event = GdprEraseEvent(target_ref_ids=("vault://users/42/email",))
        new_state, count = vm.erase(event, state)

        assert count == 1
        assert new_state.step_outputs == {"step1": "some output"}

    def test_erase_deterministic(
        self, vm: ExecutionVM, state_with_refs: StateContext
    ) -> None:
        event = GdprEraseEvent(target_ref_ids=("vault://users/42/email",))
        state_a, count_a = vm.erase(event, state_with_refs)
        state_b, count_b = vm.erase(event, state_with_refs)

        assert count_a == count_b
        assert state_a.data["email_ref"].secure_hash() == state_b.data["email_ref"].secure_hash()

    def test_erase_already_tombstoned_idempotent(self, vm: ExecutionVM) -> None:
        ref = CapabilityRef(ref_id="vault://x", salt="s").tombstone()
        state = StateContext(data={"ref": ref})
        event = GdprEraseEvent(target_ref_ids=("vault://x",))
        new_state, count = vm.erase(event, state)

        assert count == 1
        assert new_state.data["ref"].is_tombstone is True

    def test_erase_empty_state(self, vm: ExecutionVM) -> None:
        state = StateContext()
        event = GdprEraseEvent(target_ref_ids=("vault://anything",))
        new_state, count = vm.erase(event, state)

        assert count == 0
        assert new_state.data == {}


# ---------------------------------------------------------------------------
# Trace.canonical_snapshot_hash()
# ---------------------------------------------------------------------------


class TestCanonicalSnapshotHash:
    def test_empty_snapshots(self) -> None:
        trace = Trace(program_name="test")
        assert trace.canonical_snapshot_hash() == hashlib.sha256(b"empty").hexdigest()

    def test_single_snapshot(self) -> None:
        trace = Trace(program_name="test")
        trace = trace.add_snapshot(0, "abc123")
        assert trace.canonical_snapshot_hash() == hashlib.sha256(b"0:abc123").hexdigest()

    def test_two_snapshots(self) -> None:
        trace = Trace(program_name="test")
        trace = trace.add_snapshot(0, "aaa")
        trace = trace.add_snapshot(1, "bbb")

        l0 = hashlib.sha256(b"0:aaa").digest()
        l1 = hashlib.sha256(b"1:bbb").digest()
        assert trace.canonical_snapshot_hash() == hashlib.sha256(l0 + l1).hexdigest()

    def test_three_snapshots_odd_leaf_duplicated(self) -> None:
        trace = Trace(program_name="test")
        for i, fp in enumerate(["aaa", "bbb", "ccc"]):
            trace = trace.add_snapshot(i, fp)

        l0 = hashlib.sha256(b"0:aaa").digest()
        l1 = hashlib.sha256(b"1:bbb").digest()
        l2 = hashlib.sha256(b"2:ccc").digest()
        n0 = hashlib.sha256(l0 + l1).digest()
        n1 = hashlib.sha256(l2 + l2).digest()
        assert trace.canonical_snapshot_hash() == hashlib.sha256(n0 + n1).hexdigest()

    def test_four_snapshots_balanced(self) -> None:
        trace = Trace(program_name="test")
        for i, fp in enumerate(["aa", "bb", "cc", "dd"]):
            trace = trace.add_snapshot(i, fp)

        leaves = [hashlib.sha256(f"{i}:{fp}".encode()).digest() for i, fp in enumerate(["aa","bb","cc","dd"])]
        n01 = hashlib.sha256(leaves[0] + leaves[1]).digest()
        n23 = hashlib.sha256(leaves[2] + leaves[3]).digest()
        assert trace.canonical_snapshot_hash() == hashlib.sha256(n01 + n23).hexdigest()

    def test_deterministic(self) -> None:
        trace = Trace(program_name="test")
        for i in range(5):
            trace = trace.add_snapshot(i, f"fp{i}")
        assert trace.canonical_snapshot_hash() == trace.canonical_snapshot_hash()

    def test_append_changes_hash(self) -> None:
        trace = Trace(program_name="test")
        trace = trace.add_snapshot(0, "fp0")
        h_before = trace.canonical_snapshot_hash()
        trace = trace.add_snapshot(1, "fp1")
        assert trace.canonical_snapshot_hash() != h_before

    def test_tamper_detection(self) -> None:
        t_a = Trace(program_name="test").add_snapshot(0, "honest")
        t_b = Trace(program_name="test").add_snapshot(0, "tampered")
        assert t_a.canonical_snapshot_hash() != t_b.canonical_snapshot_hash()


# ---------------------------------------------------------------------------
# Forensic replay
# ---------------------------------------------------------------------------


class TestForensicReplay:
    def test_erase_extends_hash_chain(self) -> None:
        ref = CapabilityRef(ref_id="vault://users/99/ssn", salt="fixed")
        state = StateContext(data={"ssn_ref": ref})
        trace = Trace(program_name="forensic_test")
        trace = trace.add_snapshot(0, "before_erasure")
        h_before = trace.canonical_snapshot_hash()

        vm = ExecutionVM(llm=MockLLMAdapter("ok"))
        event = GdprEraseEvent(target_ref_ids=("vault://users/99/ssn",), reason="user_request")
        _, count = vm.erase(event, state)
        assert count == 1

        trace = trace.add_snapshot(1, "after_erasure")
        assert trace.canonical_snapshot_hash() != h_before
        assert len(trace.state_snapshots) == 2

    def test_tombstone_hash_constant(self) -> None:
        r1 = CapabilityRef(ref_id="vault://a", salt="s1").tombstone()
        r2 = CapabilityRef(ref_id="vault://b", salt="s2").tombstone()
        assert r1.secure_hash() == "TOMBSTONE"
        assert r2.secure_hash() == "TOMBSTONE"

    def test_policy_and_gdpr_event_compatible(self) -> None:
        policy = PolicySnapshot(
            policy_id="p-gdpr",
            version="1.0.0",
            tool_capabilities={"redact_tool": ["pii.erase"]},
        )
        event = GdprEraseEvent(
            target_ref_ids=("vault://users/1/email",),
            reason="gdpr_article_17",
        )
        assert policy.has_capability("redact_tool", "pii.erase")
        assert event.reason == "gdpr_article_17"
