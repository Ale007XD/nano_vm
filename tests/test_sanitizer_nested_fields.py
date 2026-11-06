"""SN-01..: sensitive field name survives list recursion (gap G-03)."""

from __future__ import annotations

import pytest

from nano_vm import ProjectionTarget
from nano_vm.projection import DeterministicSanitizer

REDACTED = "[REDACTED]"


def _llm(data: dict) -> dict:  # type: ignore[type-arg]
    return DeterministicSanitizer().project(data, ProjectionTarget.LLM)


@pytest.mark.parametrize(
    "data,secret",
    [
        ({"password": ["P2", "P3"]}, "P2"),
        ({"api_key": ["K1"]}, "K1"),
        ({"cfg": {"secret": ["S1"]}}, "S1"),
        ({"password": [["P5"]]}, "P5"),
        ({"token": [["a", ["T1"]]]}, "T1"),
    ],
)
def test_sn01_scalars_in_list_under_sensitive_key_are_masked(data: dict, secret: str) -> None:  # type: ignore[type-arg]
    assert secret not in str(_llm(data))


def test_sn02_every_list_item_is_masked() -> None:
    assert _llm({"password": ["a", "b", "c"]}) == {"password": [REDACTED] * 3}


def test_sn03_dict_in_list_still_masked() -> None:
    assert _llm({"items": [{"password": "P1"}]}) == {"items": [{"password": REDACTED}]}


def test_sn04_non_sensitive_list_unchanged() -> None:
    assert _llm({"names": ["ann", "bob"], "n": [1, 2]}) == {"names": ["ann", "bob"], "n": [1, 2]}


def test_sn05_step_outputs_lists_follow_the_same_rule() -> None:
    class _S:
        data: dict = {}  # type: ignore[type-arg]
        step_outputs = {"password": ["P9"]}

    out = DeterministicSanitizer().project(_S(), ProjectionTarget.LLM)
    assert "P9" not in str(out)
