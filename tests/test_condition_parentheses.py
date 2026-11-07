"""PR-01..: parentheses grouping in condition expressions (gap G-08)."""

from __future__ import annotations

import itertools
import random

import pytest

from nano_vm.ast_engine import (
    MAX_PAREN_DEPTH,
    ASTEvalError,
    LogicalNode,
    NotNode,
    eval_condition,
    parse_condition,
)

VARS = ("a", "b", "c")


def _gen(rng: random.Random, depth: int) -> tuple[str, str]:
    """Random boolean expression -> (DSL text, equivalent Python text)."""
    if depth == 0 or rng.random() < 0.25:
        v = rng.choice(VARS)
        return f"${v}", f"{v}"
    kind = rng.choice(("and", "or", "not", "group"))
    if kind == "not":
        d, p = _gen(rng, depth - 1)
        return f"not {d}", f"not {p}"
    if kind == "group":
        d, p = _gen(rng, depth - 1)
        return f"({d})", f"({p})"
    ld, lp = _gen(rng, depth - 1)
    rd, rp = _gen(rng, depth - 1)
    return f"{ld} {kind} {rd}", f"{lp} {kind} {rp}"


# PR-01: full truth table vs Python's own evaluation, parentheses everywhere
def test_pr01_truth_table_matches_python() -> None:
    rng = random.Random(20261107)
    checked = 0
    for _ in range(400):
        dsl, py = _gen(rng, 4)
        for vals in itertools.product((False, True), repeat=len(VARS)):
            ctx = dict(zip(VARS, vals))
            assert eval_condition(dsl, ctx) is bool(eval(py, {}, dict(ctx))), (dsl, ctx)  # noqa: S307
            checked += 1
    assert checked >= 400 * 8


# PR-02: grouping changes the result (the case the old tokeniser silently broke)
def test_pr02_grouping_overrides_precedence() -> None:
    ctx = {"a": True, "b": False, "c": False}
    assert eval_condition("$a or $b and $c", ctx) is True
    assert eval_condition("($a or $b) and $c", ctx) is False
    assert eval_condition("not ($a or $b)", ctx) is False
    assert eval_condition("not $a or $b", ctx) is False


# PR-03: tree shape — a group is transparent, the inner node is returned as-is
def test_pr03_group_returns_inner_node() -> None:
    node = parse_condition("($a or $b) and $c")
    assert isinstance(node, LogicalNode) and node.op == "and"
    assert isinstance(node.left, LogicalNode) and node.left.op == "or"
    neg = parse_condition("not ($a and $b)")
    assert isinstance(neg, NotNode) and isinstance(neg.operand, LogicalNode)
    assert parse_condition("(($a))") == parse_condition("$a")


# PR-04: comparisons inside groups, string/number literals, dotted paths
def test_pr04_comparisons_inside_groups() -> None:
    ctx = {"n": 3, "d": "yes", "s": {"x": 7}}
    assert eval_condition('($n > 2 and $d == "yes") or $s.x == 0', ctx) is True
    assert eval_condition('$n > 5 or ($d == "yes" and $s.x == 7)', ctx) is True
    assert eval_condition('($n > 5 or $d == "no") and $s.x == 7', ctx) is False


@pytest.mark.parametrize(
    "bad",
    [
        "($a or $b",  # missing ')'
        "$a or $b)",  # stray ')'
        "()",  # empty group
        "($a) == 1",  # a group is not an operand
        "$a == ($b)",  # ... on the right either
        "$a in ($b)",
        "(",
        ")",
    ],
)
def test_pr05_malformed_parentheses_raise(bad: str) -> None:
    with pytest.raises(ASTEvalError):
        parse_condition(bad)


def test_pr06_nesting_limit_is_deterministic() -> None:
    ok = "(" * MAX_PAREN_DEPTH + "$a" + ")" * MAX_PAREN_DEPTH
    assert eval_condition(ok, {"a": 1}) is True
    too_deep = "(" * (MAX_PAREN_DEPTH + 1) + "$a" + ")" * (MAX_PAREN_DEPTH + 1)
    with pytest.raises(ASTEvalError, match="nested deeper"):
        parse_condition(too_deep)


# PR-07: method-call guard and unsupported characters are unchanged
def test_pr07_other_rejections_unchanged() -> None:
    with pytest.raises(ASTEvalError, match="Method calls"):
        parse_condition("$a.lower() == 'x'")
    with pytest.raises(ASTEvalError, match="Unsupported character"):
        parse_condition("$a == 1 ;")
    with pytest.raises(ASTEvalError, match="Unsupported character"):
        parse_condition("$a + 1")


# PR-08: string literal on the right-hand side compares correctly (CONSTRAINTS claim was wrong);
# only a QUOTED $var is a literal text, by design (documented in the planner prompt)
def test_pr08_string_literal_rhs() -> None:
    ctx = {"d": "yes"}
    assert eval_condition('$d == "yes"', ctx) is True
    assert eval_condition("$d == 'yes'", ctx) is True
    assert eval_condition('$d != "no"', ctx) is True
    assert eval_condition('$d == "no"', ctx) is False
    assert eval_condition("$d == '$d'", ctx) is False
