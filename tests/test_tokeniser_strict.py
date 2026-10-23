"""
tests/test_tokeniser_strict.py
==============================
ASTEngine tokeniser must not silently drop characters. Before: text that no
token pattern matched was skipped by ``re.finditer``, so ``(a or b) and c``
lost its parentheses (wrong precedence, no error) and ``'x' in '$v'.lower()``
lost its ``.lower()`` tail (silent False).

  TK-01  parentheses raise (they used to be dropped -> wrong precedence)
  TK-02  trailing unmatched text raises
  TK-03  leading unmatched text raises
  TK-04  unterminated string literal raises with a hint
  TK-05  bare '$' raises with a hint
  TK-06  stray punctuation (';', '#', '-') raises, with position in message
  TK-07  quoted-var-plus-method tail raises (the README foot-gun)
  TK-08  the METHOD_CALL guard still takes precedence for $var.method()
  TK-09  valid expressions are unaffected (operators, dotted paths, whitespace)
  TK-10  characterisation: and/or are left-associative with NO precedence (pre-existing)
  TK-11  error message carries the offending character and its position
"""

from __future__ import annotations

import pytest

from nano_vm.ast_engine import ASTEvalError, eval_condition, parse_condition

CTX = {"a": 1, "b": 2, "c": 3, "decision": "yes", "s": {"x": {"y": 7}}}


def test_tk01_parentheses_raise() -> None:
    with pytest.raises(ASTEvalError, match="Parentheses"):
        parse_condition("($a == 1 or $b == 2) and $c == 3")


def test_tk01_parentheses_were_silently_dropped_semantics() -> None:
    # Without the fix this evaluated as `$a == 1 or ($b == 2 and $c == 4)` -> True,
    # while the author meant `($a == 1 or $b == 2) and $c == 4` -> False.
    with pytest.raises(ASTEvalError):
        eval_condition("($a == 1 or $b == 2) and $c == 4", CTX)


def test_tk02_trailing_garbage_raises() -> None:
    with pytest.raises(ASTEvalError, match="position"):
        parse_condition("$a == 1 ;")


def test_tk03_leading_garbage_raises() -> None:
    with pytest.raises(ASTEvalError):
        parse_condition("@ $a == 1")


def test_tk04_unterminated_string_raises() -> None:
    with pytest.raises(ASTEvalError, match="Unterminated"):
        parse_condition("$a == 'x")


def test_tk05_bare_dollar_raises() -> None:
    with pytest.raises(ASTEvalError, match="variable name"):
        parse_condition("$ a == 1")


@pytest.mark.parametrize("expr", ["$a ; 1", "$a == 1 # note", "$a-b == 1"])
def test_tk06_stray_punctuation_raises(expr: str) -> None:
    with pytest.raises(ASTEvalError):
        parse_condition(expr)


def test_tk07_quoted_var_with_method_tail_raises() -> None:
    with pytest.raises(ASTEvalError):
        eval_condition("'yes' in '$decision'.lower()", CTX)


def test_tk08_method_call_guard_still_wins() -> None:
    with pytest.raises(ASTEvalError, match="Method calls"):
        parse_condition("$decision.lower() == 'yes'")


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ('$decision == "yes"', True),
        ("$a == 1 and $b != 3", True),
        ("$a > 5 or $c >= 3", True),
        ("not $a == 2", True),
        ("'es' in $decision", True),
        ("'yes' contains $decision", True),  # `contains`: left is contained in right
        ("  $a   ==   1  ", True),
        ("$s.x.y == 7", True),
        ("'yes' in '$decision'", False),  # valid literal, documented always-False
    ],
)
def test_tk09_valid_expressions_unaffected(expr: str, expected: bool) -> None:
    assert eval_condition(expr, CTX) is expected


def test_tk10_and_or_are_left_associative_without_precedence() -> None:
    # Pre-existing behaviour, pinned so a future precedence fix is a deliberate
    # change. Python would give True here (and binds tighter than or); ASTEngine
    # evaluates left to right: ($a == 1 or $b == 9) and $c == 9 -> False.
    assert eval_condition("$a == 1 or $b == 9 and $c == 9", CTX) is False


def test_tk11_message_has_char_and_position() -> None:
    with pytest.raises(ASTEvalError) as ei:
        parse_condition("$a == 1 ; $b == 2")
    msg = str(ei.value)
    assert "';'" in msg and "position 8" in msg
