"""
tests/test_sprint4_core.py
==========================
Sprint4 / P9: рекурсивный обход nested dict/list в ExecutionVM.erase().
Репо: nano_vm
"""

from __future__ import annotations

import pytest

from nano_vm.adapters import MockLLMAdapter
from nano_vm.models import CapabilityRef, GdprEraseEvent, StateContext
from nano_vm.vm import ExecutionVM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref(ref_id: str, salt: str = "testsalt") -> CapabilityRef:
    return CapabilityRef(ref_id=ref_id, salt=salt)


def _erase(*ref_ids: str) -> GdprEraseEvent:
    return GdprEraseEvent(target_ref_ids=tuple(ref_ids), issued_by="test")


def _vm() -> ExecutionVM:
    return ExecutionVM(llm=MockLLMAdapter("ok"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEraseNested:
    def test_flat_dict(self) -> None:
        """Базовый случай: плоский dict."""
        state = StateContext(data={"email": _ref("vault://u/1/email"), "name": "Alice"})
        new_state, count = _vm().erase(_erase("vault://u/1/email"), state)

        assert count == 1
        assert new_state.data["email"].is_tombstone is True
        assert new_state.data["name"] == "Alice"

    def test_nested_dict(self) -> None:
        """CapabilityRef внутри вложенного dict."""
        state = StateContext(data={"contact": {"phone": _ref("vault://u/2/phone"), "zip": "12345"}})
        new_state, count = _vm().erase(_erase("vault://u/2/phone"), state)

        assert count == 1
        assert new_state.data["contact"]["phone"].is_tombstone is True
        assert new_state.data["contact"]["zip"] == "12345"

    def test_nested_list(self) -> None:
        """CapabilityRef внутри list."""
        state = StateContext(data={"docs": [_ref("vault://u/3/doc1"), _ref("vault://u/3/doc2")]})
        new_state, count = _vm().erase(_erase("vault://u/3/doc1"), state)

        assert count == 1
        assert new_state.data["docs"][0].is_tombstone is True
        assert new_state.data["docs"][1].is_tombstone is False

    def test_deep_list_of_dicts(self) -> None:
        """list[dict] с CapabilityRef на глубине 2+."""
        state = StateContext(data={
            "users": [
                {"id": 1, "ssn": _ref("vault://u/4/ssn")},
                {"id": 2, "ssn": None},
            ]
        })
        new_state, count = _vm().erase(_erase("vault://u/4/ssn"), state)

        assert count == 1
        assert new_state.data["users"][0]["ssn"].is_tombstone is True
        assert new_state.data["users"][1]["ssn"] is None

    def test_multiple_refs_same_id(self) -> None:
        """Один ref_id встречается несколько раз — все tombstoned."""
        state = StateContext(data={
            "primary": _ref("vault://u/5/email", salt="s1"),
            "backup": {"email": _ref("vault://u/5/email", salt="s2")},
        })
        new_state, count = _vm().erase(_erase("vault://u/5/email"), state)

        assert count == 2
        assert new_state.data["primary"].is_tombstone is True
        assert new_state.data["backup"]["email"].is_tombstone is True

    def test_multiple_targets(self) -> None:
        """Несколько ref_id в одном событии."""
        state = StateContext(data={
            "email": _ref("vault://u/6/email"),
            "phone": _ref("vault://u/6/phone"),
            "other": _ref("vault://u/6/other"),
        })
        new_state, count = _vm().erase(_erase("vault://u/6/email", "vault://u/6/phone"), state)

        assert count == 2
        assert new_state.data["email"].is_tombstone is True
        assert new_state.data["phone"].is_tombstone is True
        assert new_state.data["other"].is_tombstone is False

    def test_immutable_original(self) -> None:
        """Исходный StateContext не мутируется."""
        ref = _ref("vault://u/7/email")
        state = StateContext(data={"email": ref})
        new_state, _ = _vm().erase(_erase("vault://u/7/email"), state)

        assert state.data["email"].is_tombstone is False
        assert new_state.data["email"].is_tombstone is True

    def test_no_match_returns_zero(self) -> None:
        """ref_id не найден — count=0, data без изменений."""
        ref = _ref("vault://u/8/email")
        state = StateContext(data={"email": ref})
        new_state, count = _vm().erase(_erase("vault://u/99/other"), state)

        assert count == 0
        assert new_state.data["email"].is_tombstone is False

    def test_already_tombstoned_is_idempotent(self) -> None:
        """Повторный erase уже tombstoned ref — idempotent, count=1."""
        ref = CapabilityRef(ref_id="vault://u/9/email", salt="s", is_tombstone=True)
        state = StateContext(data={"email": ref})
        new_state, count = _vm().erase(_erase("vault://u/9/email"), state)

        assert count == 1
        assert new_state.data["email"].is_tombstone is True

    def test_scalars_unchanged(self) -> None:
        """Скалярные значения (str, int, None, bool) не затрагиваются."""
        state = StateContext(data={
            "name": "Alice",
            "age": 30,
            "active": True,
            "score": None,
            "tags": ["a", "b"],
        })
        new_state, count = _vm().erase(_erase("vault://u/10/email"), state)

        assert count == 0
        assert new_state.data == state.data
