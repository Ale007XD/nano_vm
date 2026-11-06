"""CH-01..: canonical, process-independent state hash (gap G-01)."""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import subprocess
import sys
import textwrap
from decimal import Decimal
from enum import Enum
from uuid import UUID

import pytest

from nano_vm import ExecutionVM, Program, TraceAnalyzer, TraceStatus
from nano_vm.canonical import CanonicalizationError, canonical_json

_CHILD = textwrap.dedent(
    """
    import asyncio, sys
    from nano_vm import ExecutionVM, Program, TraceAnalyzer

    OUT = {
        "set": lambda: {"alpha", "beta", "gamma", "delta", "omega"},
        "frozenset": lambda: frozenset({"x", "y", "z", "w", "v"}),
        "nested": lambda: {"k": [{"s", "t", "u"}, (1, 2)], "m": {3, 1, 2}},
        "dict_a": lambda: {"a": 1, "b": 2},
        "dict_b": lambda: {"b": 2, "a": 1},
    }[sys.argv[1]]
    p = Program.from_dict({"name": "ch", "steps": [
        {"id": "t", "type": "tool", "tool": "f", "output_key": "o", "is_terminal": True}]})
    vm = ExecutionVM(tools={"f": lambda **k: OUT()})
    print(TraceAnalyzer(asyncio.run(vm.run(p))).receipt().trace_hash)
    """
)


def _hash(kind: str, seed: int) -> str:
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    out = subprocess.run(
        [sys.executable, "-I", "-c", _CHILD, kind],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


# CH-01: unordered containers hash identically across processes (hash seeds)
@pytest.mark.parametrize("kind", ["set", "frozenset", "nested"])
def test_ch01_trace_hash_stable_across_processes(kind: str) -> None:
    assert len({_hash(kind, seed) for seed in (1, 2, 3, 4)}) == 1


# CH-02: dict insertion order does not change the hash
def test_ch02_dict_insertion_order_irrelevant() -> None:
    assert _hash("dict_a", 1) == _hash("dict_b", 1)


# CH-03: different values still hash differently
def test_ch03_different_values_different_hash() -> None:
    assert _hash("set", 1) != _hash("dict_a", 1)


class _Color(Enum):
    RED = 1


def test_ch04_types_stay_distinct() -> None:
    forms = [1, 1.0, True, "1", [1], (1,), {1}, frozenset({1}), b"1", Decimal("1")]
    encoded = [canonical_json(v) for v in forms]
    assert len(set(encoded)) == len(forms)


def test_ch05_supported_types_encode() -> None:
    for value in (
        None,
        "x",
        3,
        2.5,
        float("nan"),
        b"\x00\xff",
        Decimal("1.10"),
        dt.datetime(2026, 1, 2, 3, 4, 5),
        dt.date(2026, 1, 2),
        dt.time(3, 4),
        UUID(int=7),
        _Color.RED,
        {1: "a", 2: "b"},
        {"k": {"z": 1, "a": [1, 2]}},
    ):
        assert canonical_json(value) == canonical_json(value)


def test_ch06_non_string_dict_keys_order_independent() -> None:
    assert canonical_json({1: "a", 2: "b"}) == canonical_json({2: "b", 1: "a"})


def test_ch07_set_order_independent_and_sorted_by_canonical_form() -> None:
    assert canonical_json({"b", "a", 3, 1}) == canonical_json({1, 3, "a", "b"})


class _Opaque:
    pass


@pytest.mark.parametrize("value", [_Opaque(), lambda: 1, object(), {"k": _Opaque()}, [_Opaque()]])
def test_ch08_unrepresentable_values_are_refused(value: object) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json(value)


def test_ch09_self_referential_is_refused() -> None:
    loop: list[object] = []
    loop.append(loop)
    with pytest.raises(CanonicalizationError):
        canonical_json(loop)


# CH-10: the VM fails the run (no silent repr() hash) when a tool output cannot be encoded
def test_ch10_vm_fails_run_on_unrepresentable_tool_output() -> None:
    p = Program.from_dict(
        {
            "name": "ch10",
            "steps": [{"id": "t", "type": "tool", "tool": "f", "output_key": "o"}],
        }
    )
    vm = ExecutionVM(tools={"f": lambda **k: _Opaque()})
    trace = asyncio.run(vm.run(p))
    assert trace.status == TraceStatus.FAILED
    assert "canonically" in (trace.error or "")
    assert "_Opaque" in (trace.error or "")


def test_ch11_snapshot_hash_depends_only_on_content() -> None:
    p = Program.from_dict(
        {"name": "ch11", "steps": [{"id": "t", "type": "tool", "tool": "f", "is_terminal": True}]}
    )

    def run(value: object) -> str:
        vm = ExecutionVM(tools={"f": lambda **k: value})
        return TraceAnalyzer(asyncio.run(vm.run(p))).receipt().trace_hash

    assert run({"x": 1, "y": 2}) == run({"y": 2, "x": 1})
    assert run({"x": 1}) != run({"x": 2})
