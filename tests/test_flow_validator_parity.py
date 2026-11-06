"""FL-01..: validator reachability == real control flow (gap G-02)."""

from __future__ import annotations

import asyncio
import itertools
import random

import pytest

from nano_vm import ExecutionVM, Program, ProgramValidator
from nano_vm.flow import Entry, next_after
from nano_vm.validator import IssueKind


def _reachable(p: Program) -> set[str]:
    adj = ProgramValidator(p)._adjacency
    seen: set[str] = set()
    stack = [p.steps[0].id]
    while stack:
        s = stack.pop()
        if s not in seen:
            seen.add(s)
            stack.extend(adj.get(s, []))
    return seen


def _executed(p: Program, cond_ids: list[str]) -> set[str]:
    """Union of executed step ids over every combination of condition outcomes."""
    ex: set[str] = set()
    for outcome in itertools.product((0, 1), repeat=len(cond_ids)):
        ctx = {f"c_{cid}": v for cid, v in zip(cond_ids, outcome)}
        tools = {s.tool: (lambda **k: 1) for s in p.steps if s.tool}
        trace = asyncio.run(ExecutionVM(tools=tools).run(p, ctx))
        ex |= {r.step_id for r in trace.steps}
    return ex


AUDIT = {
    "name": "audit",
    "max_steps": 50,
    "steps": [
        {"id": "chk", "type": "tool", "tool": "t0"},
        {
            "id": "gate",
            "type": "condition",
            "condition": "$c_gate > 0",
            "then": "approve",
            "otherwise": "deny",
        },
        {"id": "approve", "type": "tool", "tool": "t1"},
        {"id": "deny", "type": "tool", "tool": "t2"},
        {"id": "audit", "type": "tool", "tool": "t3", "is_terminal": True},
    ],
}


# FL-01: the review's case — 'audit' is never executed, so it must not be reachable
def test_fl01_audit_after_branch_without_next_step_is_unreachable() -> None:
    p = Program.from_dict(AUDIT)
    report = ProgramValidator(p).validate()
    assert [i.step_id for i in report.by_kind(IssueKind.UNREACHABLE_STEP)] == ["audit"]
    assert not report.is_valid()
    assert _reachable(p) == _executed(p, ["gate"]) == {"chk", "gate", "approve", "deny"}


# FL-02: explicit next_step on a branch target keeps the continuation reachable
def test_fl02_branch_target_with_next_step_reaches_audit() -> None:
    data = {**AUDIT, "steps": [dict(s) for s in AUDIT["steps"]]}  # type: ignore[arg-type]
    data["steps"][2]["next_step"] = "audit"  # approve -> audit
    data["steps"][3]["next_step"] = "audit"  # deny -> audit
    p = Program.from_dict(data)
    assert ProgramValidator(p).validate().is_valid()
    assert _reachable(p) == _executed(p, ["gate"]) == {"chk", "gate", "approve", "deny", "audit"}


# FL-03: plain sequential flow after a next_step jump still falls through
def test_fl03_plain_entry_keeps_sequential_fallthrough() -> None:
    p = Program.from_dict(
        {
            "name": "seq",
            "steps": [
                {"id": "a", "type": "tool", "tool": "t", "next_step": "c"},
                {"id": "b", "type": "tool", "tool": "t"},
                {"id": "c", "type": "tool", "tool": "t"},
                {"id": "d", "type": "tool", "tool": "t", "is_terminal": True},
            ],
        }
    )
    assert _reachable(p) == _executed(p, []) == {"a", "c", "d"}


# FL-04: CONDITION -> CONDITION: the nested targets are entered plain (sequential fallthrough)
def test_fl04_nested_condition_targets_enter_plain() -> None:
    p = Program.from_dict(
        {
            "name": "nested",
            "max_steps": 50,
            "steps": [
                {
                    "id": "c1",
                    "type": "condition",
                    "condition": "$c_c1 > 0",
                    "then": "c2",
                    "otherwise": "x",
                },
                {
                    "id": "c2",
                    "type": "condition",
                    "condition": "$c_c2 > 0",
                    "then": "y",
                    "otherwise": "z",
                },
                {"id": "x", "type": "tool", "tool": "t"},
                {"id": "y", "type": "tool", "tool": "t"},
                {"id": "z", "type": "tool", "tool": "t"},
                {"id": "tail", "type": "tool", "tool": "t", "is_terminal": True},
            ],
        }
    )
    assert _reachable(p) == _executed(p, ["c1", "c2"])


def _random_program(rng: random.Random) -> tuple[Program, list[str]] | None:
    k = rng.randint(4, 8)
    ids = [f"s{i}" for i in range(k)]
    n_cond = rng.randint(1, 2)
    cond_idx = set(rng.sample(range(0, k - 1), n_cond))
    steps: list[dict[str, object]] = []
    cond_ids: list[str] = []
    for i, sid in enumerate(ids):
        if i in cond_idx:
            others = [x for x in ids if x != sid]
            a, b = rng.sample(others, 2)
            steps.append(
                {
                    "id": sid,
                    "type": "condition",
                    "condition": f"$c_{sid} > 0",
                    "then": a,
                    "otherwise": b,
                }
            )
            cond_ids.append(sid)
            continue
        d: dict[str, object] = {"id": sid, "type": "tool", "tool": "t"}
        r = rng.random()
        if i == k - 1 or r < 0.25:
            d["is_terminal"] = True
        elif r < 0.5:
            d["next_step"] = rng.choice(ids)
        steps.append(d)
    try:
        return Program.from_dict({"name": "rnd", "max_steps": 40, "steps": steps}), cond_ids
    except Exception:
        return None


# FL-05: property — over all branch outcomes, executed steps == validator-reachable steps
def test_fl05_property_validator_matches_vm() -> None:
    rng = random.Random(20261106)
    checked = 0
    for _ in range(600):
        made = _random_program(rng)
        if made is None:
            continue
        program, cond_ids = made
        report = ProgramValidator(program).validate()
        if report.by_kind(IssueKind.CYCLE_DETECTED) or report.by_kind(IssueKind.MISSING_TARGET):
            continue  # cyclic programs revisit conditions; out of scope for set equality
        reachable = _reachable(program)
        executed = _executed(program, cond_ids)
        assert reachable == executed, (program.model_dump(exclude_none=True), reachable, executed)
        checked += 1
    assert checked >= 150


# FL-06: the VM uses flow.next_after — the validator and runtime share one definition
def test_fl06_vm_follows_flow_next_after(monkeypatch: pytest.MonkeyPatch) -> None:
    import nano_vm.vm as vm_mod

    p = Program.from_dict(
        {
            "name": "share",
            "max_steps": 10,
            "steps": [
                {"id": "a", "type": "tool", "tool": "t"},
                {"id": "b", "type": "tool", "tool": "t"},
                {"id": "c", "type": "tool", "tool": "t", "is_terminal": True},
            ],
        }
    )

    def jump_to_c(step, index, steps, entry):  # type: ignore[no-untyped-def]
        adv = next_after(step, index, steps, entry)
        if step.id == "a" and entry is Entry.PLAIN:
            return type(adv)("c")
        return adv

    monkeypatch.setattr(vm_mod, "next_after", jump_to_c)
    trace = asyncio.run(ExecutionVM(tools={"t": lambda **k: 1}).run(p))
    assert [r.step_id for r in trace.steps] == ["a", "c"]
