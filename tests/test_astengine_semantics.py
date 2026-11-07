"""ASTEngine operator semantics: and/or/not precedence and `contains` direction.

AS-01  `and` binds tighter than `or` (matches Python)
AS-02  full truth table vs Python's own evaluation (3 vars, mixed and/or/not)
AS-03  `not` binds tighter than `and`/`or` and is accepted after and/or
AS-04  same-level chains stay left-associative
AS-05  `contains`: left contains right (inverse of `in`)
AS-06  `contains` / `in` are exact mirrors; non-container operand -> ASTEvalError
AS-07  parentheses group boolean sub-expressions
"""

from __future__ import annotations

import itertools

import pytest

from nano_vm.ast_engine import ASTEvalError, LogicalNode, eval_condition, parse_condition


def test_as01_and_binds_tighter_than_or() -> None:
    ctx = {"a": 1, "b": 2, "c": 3}
    assert eval_condition("$a == 1 or $b == 9 and $c == 9", ctx) is True
    assert eval_condition("$a == 9 and $b == 2 or $c == 3", ctx) is True
    assert eval_condition("$a == 9 and $b == 2 or $c == 9", ctx) is False
    node = parse_condition("$a == 1 or $b == 2 and $c == 3")
    assert isinstance(node, LogicalNode) and node.op == "or"
    assert isinstance(node.right, LogicalNode) and node.right.op == "and"


_EXPRS = [
    ("$x or $y and $z", "x or y and z"),
    ("$x and $y or $z", "x and y or z"),
    ("$x or $y or $z and $x", "x or y or z and x"),
    ("not $x or $y", "(not x) or y"),
    ("not $x and $y or $z", "((not x) and y) or z"),
    ("$x and not $y or $z", "(x and (not y)) or z"),
    ("$x or not $y and $z", "x or ((not y) and z)"),
    ("not not $x and $y", "(not (not x)) and y"),
]


@pytest.mark.parametrize(("dsl", "py"), _EXPRS)
def test_as02_truth_table_matches_python(dsl: str, py: str) -> None:
    for x, y, z in itertools.product([True, False], repeat=3):
        got = eval_condition(dsl, {"x": x, "y": y, "z": z})
        want = bool(eval(py, {"__builtins__": {}}, {"x": x, "y": y, "z": z}))  # noqa: S307
        assert got is want, (dsl, x, y, z)


def test_as03_not_after_logic_operator_is_accepted() -> None:
    assert eval_condition("$a == 1 and not $b == 9", {"a": 1, "b": 2}) is True
    assert eval_condition("$a == 9 or not $b == 2", {"a": 1, "b": 2}) is False


def test_as04_same_level_left_associative() -> None:
    node = parse_condition("$x and $y and $z")
    assert isinstance(node, LogicalNode) and isinstance(node.left, LogicalNode)


def test_as05_contains_left_contains_right() -> None:
    ctx = {"decision": "yes approved", "roles": ["admin", "user"]}
    assert eval_condition("$decision contains 'yes'", ctx) is True
    assert eval_condition("'yes' contains $decision", ctx) is False
    assert eval_condition("$roles contains 'admin'", ctx) is True
    assert eval_condition("$roles contains 'root'", ctx) is False


def test_as06_contains_mirrors_in() -> None:
    ctx = {"s": "hello world", "w": "world"}
    assert eval_condition("$s contains $w", ctx) is eval_condition("$w in $s", ctx) is True
    with pytest.raises(ASTEvalError):
        eval_condition("$n contains 1", {"n": 5})


def test_as07_parentheses_group_boolean_subexpressions() -> None:
    assert eval_condition("($a == 1 or $b == 2) and $c == 3", {"a": 0, "b": 2, "c": 3}) is True
    assert eval_condition("$a == 1 or ($b == 2 and $c == 3)", {"a": 0, "b": 2, "c": 4}) is False
