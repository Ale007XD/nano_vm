"""
nano_vm.ast_engine — Sprint 1: Deterministic AST Evaluator
===========================================================
Replaces eval() for condition step evaluation.

Public API:
  ASTEngine       — evaluates ConditionExpr node trees
  eval_condition  — parses a DSL string and evaluates it
  ASTEvalError    — raised on type errors or unknown operators
  BinaryNode      — op, left, right
  LogicalNode     — op (and/or), left, right
  NotNode         — op (not), operand
  LitNode         — literal value
  VarNode         — variable name resolved from context

Design invariants:
  - Pure function evaluation: no I/O, no global state, no eval().
  - Same (node, ctx) → same result always.
  - Unknown operator → ASTEvalError, not silent fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Union


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ASTEvalError(Exception):
    """Raised when the AST evaluator encounters a type error or bad operator."""


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------

# Forward reference for recursive type alias
ConditionExpr = Union[
    "BinaryNode", "LogicalNode", "NotNode", "LitNode", "VarNode"
]


@dataclass(frozen=True)
class LitNode:
    """Literal value node."""
    value: Any


@dataclass(frozen=True)
class VarNode:
    """Variable reference resolved from the evaluation context.

    Name formats:
      "key"              -> ctx["key"]
      "step_id.output"   -> ctx["__step_outputs__"]["step_id"]
    """
    name: str


@dataclass(frozen=True)
class BinaryNode:
    """Binary comparison node.

    Supported operators: ==, !=, >, <, in, not in, contains
    """
    op: str
    left: ConditionExpr
    right: ConditionExpr


@dataclass(frozen=True)
class LogicalNode:
    """Logical combinator node.

    Supported operators: and, or
    """
    op: str
    left: ConditionExpr
    right: ConditionExpr


@dataclass(frozen=True)
class NotNode:
    """Logical negation node."""
    op: str  # always "not"
    operand: ConditionExpr


# ---------------------------------------------------------------------------
# ASTEngine
# ---------------------------------------------------------------------------


class ASTEngine:
    """Evaluates ConditionExpr node trees against a context dict.

    All methods are pure functions — no state is mutated.
    """

    def evaluate(self, node: ConditionExpr, ctx: dict[str, Any]) -> bool:
        """Evaluate *node* against *ctx*.

        Returns
        -------
        bool
            Result of the expression.

        Raises
        ------
        ASTEvalError
            On type mismatch or unsupported operator.
        """
        if isinstance(node, LitNode):
            # Literals evaluate to themselves — used as sub-expressions.
            return bool(node.value)

        if isinstance(node, VarNode):
            return bool(self._resolve_var(node, ctx))

        if isinstance(node, NotNode):
            return not self.evaluate(node.operand, ctx)

        if isinstance(node, LogicalNode):
            return self._eval_logical(node, ctx)

        if isinstance(node, BinaryNode):
            return self._eval_binary(node, ctx)

        raise ASTEvalError(f"Unknown node type: {type(node)}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve(self, node: ConditionExpr, ctx: dict[str, Any]) -> Any:
        """Resolve a node to its raw value (not coerced to bool)."""
        if isinstance(node, LitNode):
            return node.value
        if isinstance(node, VarNode):
            return self._resolve_var(node, ctx)
        # For compound nodes, evaluate to bool
        return self.evaluate(node, ctx)

    @staticmethod
    def _resolve_var(node: VarNode, ctx: dict[str, Any]) -> Any:
        """Resolve VarNode name from context.

        "classify.output" -> ctx["__step_outputs__"]["classify"]
        "key"             -> ctx.get("key")
        """
        if node.name.endswith(".output"):
            step_id = node.name[: -len(".output")]
            return (ctx.get("__step_outputs__") or {}).get(step_id)
        return ctx.get(node.name)

    def _eval_logical(self, node: LogicalNode, ctx: dict[str, Any]) -> bool:
        if node.op == "and":
            return self.evaluate(node.left, ctx) and self.evaluate(node.right, ctx)
        if node.op == "or":
            return self.evaluate(node.left, ctx) or self.evaluate(node.right, ctx)
        raise ASTEvalError(f"Unknown logical operator: {node.op!r}")

    def _eval_binary(self, node: BinaryNode, ctx: dict[str, Any]) -> bool:
        left = self._resolve(node.left, ctx)
        right = self._resolve(node.right, ctx)
        op = node.op

        try:
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right  # type: ignore[operator]
            if op == ">=":
                return left >= right  # type: ignore[operator]
            if op == "<":
                return left < right  # type: ignore[operator]
            if op == "<=":
                return left <= right  # type: ignore[operator]
            if op == "in":
                return left in right  # type: ignore[operator]
            if op == "not in":
                return left not in right  # type: ignore[operator]
            if op == "contains":
                # "contains": checks if left is contained in right
                return left in right  # type: ignore[operator]
        except TypeError as exc:
            raise ASTEvalError(f"Type error evaluating '{op}': {exc}") from exc

        raise ASTEvalError(f"Unknown binary operator: {op!r}")


# ---------------------------------------------------------------------------
# DSL parser  — string condition → ConditionExpr
# ---------------------------------------------------------------------------

# Tokeniser patterns (order matters)
_TOKEN_PATTERNS = [
    ("LOGIC", r"\band\b|\bor\b"),
    ("NOT", r"\bnot\b(?!\s+in\b)"),
    ("NOT_IN", r"\bnot\s+in\b"),
    ("IN", r"\bin\b"),
    ("CONTAINS", r"\bcontains\b"),
    ("OP", r"==|!=|>=|<=|>|<"),
    ("VAR", r"\$[\w.]+"),
    ("STR", r"'[^']*'|\"[^\"]*\""),
    ("NUM", r"-?\d+(?:\.\d+)?"),
    ("BOOL", r"\b(?:True|False|None)\b"),
    ("WS", r"\s+"),
]

_TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_PATTERNS)
)


def _tokenise(expr: str) -> list[tuple[str, str]]:
    tokens = []
    for m in _TOKEN_RE.finditer(expr):
        kind = m.lastgroup
        if kind == "WS":
            continue
        tokens.append((kind, m.group()))
    return tokens


def _parse_literal(kind: str, val: str) -> LitNode:
    if kind == "STR":
        return LitNode(value=val[1:-1])
    if kind == "NUM":
        return LitNode(value=float(val) if "." in val else int(val))
    if kind == "BOOL":
        return LitNode(value={"True": True, "False": False, "None": None}[val])
    raise ASTEvalError(f"Cannot parse literal: {kind}={val!r}")


def _parse_operand(tokens: list[tuple[str, str]], pos: int) -> tuple[ConditionExpr, int]:
    if pos >= len(tokens):
        raise ASTEvalError("Unexpected end of expression")
    kind, val = tokens[pos]
    if kind == "VAR":
        return VarNode(name=val[1:]), pos + 1
    if kind in ("STR", "NUM", "BOOL"):
        return _parse_literal(kind, val), pos + 1
    raise ASTEvalError(f"Expected operand, got {kind}={val!r} at position {pos}")


def _parse_binary(tokens: list[tuple[str, str]], pos: int) -> tuple[ConditionExpr, int]:
    left, pos = _parse_operand(tokens, pos)
    if pos >= len(tokens):
        return left, pos

    kind, val = tokens[pos]
    if kind in ("OP", "IN", "NOT_IN", "CONTAINS"):
        # Map token to operator string
        op_map = {
            "IN": "in", "NOT_IN": "not in", "CONTAINS": "contains",
        }
        op = op_map.get(kind, val)
        # Filter unsupported ops
        if op not in ("==", "!=", ">", "<", ">=", "<=", "in", "not in", "contains"):
            raise ASTEvalError(f"Unsupported operator: {op!r}")
        right, pos = _parse_operand(tokens, pos + 1)
        return BinaryNode(op=op, left=left, right=right), pos

    return left, pos


def _parse_expr(tokens: list[tuple[str, str]], pos: int) -> tuple[ConditionExpr, int]:
    """Parse with left-to-right and/or logical combinators."""
    # NOT prefix
    if pos < len(tokens) and tokens[pos][0] == "NOT":
        operand, pos = _parse_binary(tokens, pos + 1)
        node: ConditionExpr = NotNode(op="not", operand=operand)
    else:
        node, pos = _parse_binary(tokens, pos)

    while pos < len(tokens) and tokens[pos][0] == "LOGIC":
        op = tokens[pos][1]
        right, pos = _parse_binary(tokens, pos + 1)
        node = LogicalNode(op=op, left=node, right=right)

    return node, pos


def parse_condition(expr: str) -> ConditionExpr:
    """Parse a DSL condition string into a ConditionExpr node tree.

    Raises ASTEvalError on parse failure.
    """
    tokens = _tokenise(expr.strip())
    if not tokens:
        raise ASTEvalError("Empty condition expression")
    node, pos = _parse_expr(tokens, 0)
    if pos < len(tokens):
        raise ASTEvalError(
            f"Unexpected token at position {pos}: {tokens[pos]!r}"
        )
    return node


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

_engine = ASTEngine()


def eval_condition(expr: str, ctx: dict[str, Any]) -> bool:
    """Parse *expr* and evaluate it against *ctx*.

    Equivalent to::

        ASTEngine().evaluate(parse_condition(expr), ctx)
    """
    node = parse_condition(expr)
    return _engine.evaluate(node, ctx)
