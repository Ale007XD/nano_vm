"""
nano_vm.ast_engine
==================
Deterministic AST Evaluator — замена Python eval() для condition steps.

RFC v0.7.0: «Pure function evaluation, no I/O or global state access.»

Архитектура:
  - ConditionExpr — Pydantic-модель JSON-дерева выражения.
  - ASTEngine — evaluator, принимает ConditionExpr + контекст, возвращает bool.
  - parse_condition() — компилирует строку DSL («'yes' in $decision») в ConditionExpr.

Поддерживаемые операторы (RFC):
  ==, !=, >, <, in, not in, and, or, contains

Инварианты:
  - Нет вызовов eval() / exec().
  - Нет обращений к глобальному состоянию или I/O.
  - Одинаковый (expr, context) → всегда одинаковый результат.
  - Неизвестный оператор → ASTEvalError (не молчаливое игнорирование).

Формат JSON-дерева:
  {
    "op": "and",
    "left":  { "op": "in", "left": {"op": "lit", "value": "yes"}, "right": {"op": "var", "name": "decision"} },
    "right": { "op": ">",  "left": {"op": "var", "name": "score"}, "right": {"op": "lit", "value": 0} }
  }
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ASTEvalError(Exception):
    """Ошибка вычисления AST-выражения."""


class ASTParseError(Exception):
    """Ошибка компиляции строки DSL в ConditionExpr."""


# ---------------------------------------------------------------------------
# AST Node models
# ---------------------------------------------------------------------------

# Терминальные узлы


class LitNode(BaseModel):
    """Литеральное значение: строка, число, bool."""

    op: Literal["lit"] = "lit"
    value: Any


class VarNode(BaseModel):
    """Переменная из контекста: $name или $step_id.output."""

    op: Literal["var"] = "var"
    name: str  # без '$'


# Составные узлы (рекурсивные)


class BinaryNode(BaseModel):
    """Бинарный оператор: ==, !=, >, <, in, not in, contains."""

    op: Literal["==", "!=", ">", "<", "in", "not in", "contains"]
    left: ConditionExpr
    right: ConditionExpr


class LogicalNode(BaseModel):
    """Логический оператор: and, or."""

    op: Literal["and", "or"]
    left: ConditionExpr
    right: ConditionExpr


class NotNode(BaseModel):
    """Унарное отрицание."""

    op: Literal["not"]
    operand: ConditionExpr


# Дискриминированный union

ConditionExpr = Union[LitNode, VarNode, BinaryNode, LogicalNode, NotNode]

# Pydantic v2: rebuild для рекурсивных моделей
BinaryNode.model_rebuild()
LogicalNode.model_rebuild()
NotNode.model_rebuild()


# ---------------------------------------------------------------------------
# ASTEngine
# ---------------------------------------------------------------------------


class ASTEngine:
    """
    Детерминированный evaluator для ConditionExpr.

    Использование:
        engine = ASTEngine()
        expr = parse_condition("'yes' in $decision")
        result: bool = engine.evaluate(expr, context={"decision": "yes, approved"})
    """

    def evaluate(self, node: ConditionExpr, context: dict[str, Any]) -> bool:
        """
        Вычисляет выражение в заданном контексте.
        Возвращает bool.
        Бросает ASTEvalError при ошибке типов или неизвестном операторе.
        """
        val = self._eval(node, context)
        return bool(val)

    # ------------------------------------------------------------------
    # Internal recursive evaluator
    # ------------------------------------------------------------------

    def _eval(self, node: ConditionExpr, context: dict[str, Any]) -> Any:
        if isinstance(node, LitNode):
            return node.value

        if isinstance(node, VarNode):
            return self._resolve_var(node.name, context)

        if isinstance(node, NotNode):
            return not bool(self._eval(node.operand, context))

        if isinstance(node, LogicalNode):
            left = bool(self._eval(node.left, context))
            if node.op == "and":
                return left and bool(self._eval(node.right, context))
            if node.op == "or":
                return left or bool(self._eval(node.right, context))
            raise ASTEvalError(f"Unknown logical op: {node.op}")  # защита от расширения enum

        if isinstance(node, BinaryNode):
            return self._eval_binary(node, context)

        raise ASTEvalError(f"Unknown AST node type: {type(node).__name__}")

    def _eval_binary(self, node: BinaryNode, context: dict[str, Any]) -> bool:
        left = self._eval(node.left, context)
        right = self._eval(node.right, context)
        op = node.op

        try:
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right  # type: ignore[operator]
            if op == "<":
                return left < right  # type: ignore[operator]
            if op == "in":
                return left in right  # type: ignore[operator]
            if op == "not in":
                return left not in right  # type: ignore[operator]
            if op == "contains":
                # contains: right contains left  (e.g. "$text contains 'yes'")
                return left in right  # type: ignore[operator]
        except TypeError as exc:
            raise ASTEvalError(
                f"Type error in '{op}': left={left!r} ({type(left).__name__}), "
                f"right={right!r} ({type(right).__name__}): {exc}"
            ) from exc

        raise ASTEvalError(f"Unknown binary op: {op}")

    def _resolve_var(self, name: str, context: dict[str, Any]) -> Any:
        """
        Разрешает переменную из контекста.

        Поддерживает:
          - "key"          → context["key"]
          - "step_id.output" → context["__step_outputs__"]["step_id"]

        Если ключ не найден — возвращает None (не бросает исключение,
        чтобы условие могло быть вычислено как False).
        """
        if "." in name:
            step_id, field = name.split(".", 1)
            step_outputs = context.get("__step_outputs__", {})
            step_out = step_outputs.get(step_id)
            if step_out is None:
                return None
            if field == "output":
                return step_out
            if isinstance(step_out, dict):
                return step_out.get(field)
            return None

        return context.get(name)


# ---------------------------------------------------------------------------
# DSL parser: строка → ConditionExpr
# ---------------------------------------------------------------------------

# Поддерживаемые операторы в строковом DSL
# Порядок важен: «not in» должен проверяться до «in»
_BINARY_OPS = ["not in", "==", "!=", ">=", "<=", ">", "<", " in ", "contains"]

# Допустимые операторы RFC (без алиасов >=, <=)
_RFC_OPS = {"==", "!=", ">", "<", "in", "not in", "contains"}


def parse_condition(expr_str: str) -> ConditionExpr:
    """
    Компилирует строку DSL в ConditionExpr.

    Поддерживаемые форматы:
      'yes' in $decision
      $score > 0
      len($summary) > 100          ← NOT поддерживается (нет вызовов функций)
      $a == $b
      $decision != 'no'
      $flag == True

    Для сложных выражений с and/or:
      $a > 0 and $b == 'ok'

    Ограничения:
      - Нет арифметики (+-*/).
      - Нет вызовов функций (len(), str()…).
      - Нет вложенных скобок.
      Для таких случаев используйте ConditionExpr-дерево напрямую.

    Бросает ASTParseError если выражение не распознано.
    """
    expr_str = expr_str.strip()

    # Попытка разбить по 'and' / 'or' (низкий приоритет)
    for logical_op in (" and ", " or "):
        idx = _find_logical_split(expr_str, logical_op)
        if idx != -1:
            left_str = expr_str[:idx].strip()
            right_str = expr_str[idx + len(logical_op) :].strip()
            left = parse_condition(left_str)
            right = parse_condition(right_str)
            op_key = logical_op.strip()
            return LogicalNode(op=op_key, left=left, right=right)  # type: ignore[arg-type]

    # Бинарные операторы
    for op in _BINARY_OPS:
        idx = _find_op(expr_str, op)
        if idx != -1:
            left_str = expr_str[:idx].strip()
            right_str = expr_str[idx + len(op) :].strip()
            left = _parse_atom(left_str)
            right = _parse_atom(right_str)
            # Нормализуем ' in ' → 'in'
            normalized_op = op.strip()
            if normalized_op not in _RFC_OPS:
                raise ASTParseError(
                    f"Operator '{normalized_op}' not in RFC-allowed set: {_RFC_OPS}"
                )
            return BinaryNode(op=normalized_op, left=left, right=right)  # type: ignore[arg-type]

    # Одиночный атом (bool/var/lit)
    return _parse_atom(expr_str)


def _find_logical_split(expr: str, logical_op: str) -> int:
    """Находит индекс logical_op вне кавычек."""
    in_single = False
    in_double = False
    for i, ch in enumerate(expr):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if not in_single and not in_double:
            if expr[i : i + len(logical_op)] == logical_op:
                return i
    return -1


def _find_op(expr: str, op: str) -> int:
    """Находит индекс оператора вне кавычек."""
    in_single = False
    in_double = False
    for i, ch in enumerate(expr):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if not in_single and not in_double:
            if expr[i : i + len(op)] == op:
                return i
    return -1


def _parse_atom(s: str) -> ConditionExpr:
    """Разбирает одиночный атом: переменная, строковый литерал, число, bool."""
    s = s.strip()

    # Переменная: $name или $step_id.output
    if s.startswith("$"):
        return VarNode(name=s[1:])

    # Строковый литерал в одинарных или двойных кавычках
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return LitNode(value=s[1:-1])

    # Bool
    if s == "True":
        return LitNode(value=True)
    if s == "False":
        return LitNode(value=False)
    if s == "None":
        return LitNode(value=None)

    # Число
    try:
        if "." in s:
            return LitNode(value=float(s))
        return LitNode(value=int(s))
    except ValueError:
        pass

    # Неизвестный атом — возвращаем как строковый литерал с предупреждением
    # (обратная совместимость: 'yes' без кавычек в old DSL)
    return LitNode(value=s)


# ---------------------------------------------------------------------------
# Convenience: compile + evaluate in one call
# ---------------------------------------------------------------------------


def eval_condition(expr_str: str, context: dict[str, Any]) -> bool:
    """
    Компилирует строку и вычисляет результат.
    Эквивалент: ASTEngine().evaluate(parse_condition(expr_str), context)

    Используется в vm._execute_condition() вместо eval().
    """
    engine = ASTEngine()
    try:
        node = parse_condition(expr_str)
        return engine.evaluate(node, context)
    except ASTParseError:
        raise
    except ASTEvalError:
        raise
    except Exception as exc:
        raise ASTEvalError(f"Unexpected error evaluating '{expr_str}': {exc}") from exc
