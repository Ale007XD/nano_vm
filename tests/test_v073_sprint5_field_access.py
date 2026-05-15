"""
test_v073_sprint5_field_access.py
Sprint 5: structured-field-access — dotted path in ASTEngine VarNode.

Coverage:
  FA-01  simple key (no dots) — backward compat
  FA-02  step_id.output scalar — backward compat
  FA-03  step_id.output.field — dict output, one level
  FA-04  step_id.output.nested.field — dict output, two levels
  FA-05  missing top-level key → None → condition False
  FA-06  missing nested key → None → condition False
  FA-07  non-dict intermediate → None → condition False
  FA-08  step_id.output where output is a dict and 'output' is a key inside it
  FA-09  eval_condition with dotted path end-to-end
  FA-10  vm._execute_condition with dotted path via StateContext
"""

from __future__ import annotations

from ast_engine import (
    ASTEngine,
    BinaryNode,
    LitNode,
    VarNode,
    eval_condition,
    parse_condition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

engine = ASTEngine()


def ctx_with_step(step_id: str, output: object) -> dict:
    """Build a ctx dict that mirrors what vm._execute_condition produces:
    {**state.step_outputs, **state.data}
    step_outputs = {step_id: output}
    """
    return {step_id: output}


# ---------------------------------------------------------------------------
# FA-01: simple key — no dots
# ---------------------------------------------------------------------------


def test_fa01_simple_key():
    ctx = {"decision": "yes"}
    node = VarNode(name="decision")
    assert engine._resolve_var(node, ctx) == "yes"


# ---------------------------------------------------------------------------
# FA-02: step_id.output scalar — backward compat
# ---------------------------------------------------------------------------


def test_fa02_step_output_scalar():
    ctx = ctx_with_step("classify", "refund")
    node = VarNode(name="classify.output")
    assert engine._resolve_var(node, ctx) == "refund"


def test_fa02_step_output_scalar_in_condition():
    ctx = ctx_with_step("classify", "refund")
    result = eval_condition("$classify.output == 'refund'", ctx)
    assert result is True


# ---------------------------------------------------------------------------
# FA-03: step_id.output.field — one level into dict
# ---------------------------------------------------------------------------


def test_fa03_one_level_field():
    ctx = ctx_with_step("poll_payment", {"payment_status": "SUCCESS", "amount": 100})
    node = VarNode(name="poll_payment.output.payment_status")
    assert engine._resolve_var(node, ctx) == "SUCCESS"


def test_fa03_one_level_field_in_condition():
    ctx = ctx_with_step("poll_payment", {"payment_status": "SUCCESS"})
    result = eval_condition("$poll_payment.output.payment_status == 'SUCCESS'", ctx)
    assert result is True


def test_fa03_one_level_field_false():
    ctx = ctx_with_step("poll_payment", {"payment_status": "PENDING"})
    result = eval_condition("$poll_payment.output.payment_status == 'SUCCESS'", ctx)
    assert result is False


# ---------------------------------------------------------------------------
# FA-04: step_id.output.nested.field — two levels deep
# ---------------------------------------------------------------------------


def test_fa04_two_level_nested():
    ctx = ctx_with_step("create_payment", {"data": {"order_id": "ord_123"}, "status": "OK"})
    node = VarNode(name="create_payment.output.data.order_id")
    assert engine._resolve_var(node, ctx) == "ord_123"


def test_fa04_two_level_condition():
    ctx = ctx_with_step("create_payment", {"data": {"order_id": "ord_123"}})
    result = eval_condition("$create_payment.output.data.order_id == 'ord_123'", ctx)
    assert result is True


# ---------------------------------------------------------------------------
# FA-05: missing top-level key → None
# ---------------------------------------------------------------------------


def test_fa05_missing_top_key():
    ctx: dict = {}
    node = VarNode(name="nonexistent.output.field")
    assert engine._resolve_var(node, ctx) is None


def test_fa05_missing_top_key_in_condition():
    ctx: dict = {}
    result = eval_condition("$nonexistent.output.field == 'SUCCESS'", ctx)
    assert result is False


# ---------------------------------------------------------------------------
# FA-06: missing nested key → None
# ---------------------------------------------------------------------------


def test_fa06_missing_nested_key():
    ctx = ctx_with_step("poll", {"status": "OK"})
    node = VarNode(name="poll.output.no_such_field")
    assert engine._resolve_var(node, ctx) is None


def test_fa06_missing_nested_key_condition():
    ctx = ctx_with_step("poll", {"status": "OK"})
    result = eval_condition("$poll.output.no_such_field == 'SUCCESS'", ctx)
    assert result is False


# ---------------------------------------------------------------------------
# FA-07: non-dict intermediate → None
# ---------------------------------------------------------------------------


def test_fa07_non_dict_intermediate():
    # output is a scalar string, not a dict — cannot traverse further
    ctx = ctx_with_step("step", "just_a_string")
    node = VarNode(name="step.output.field")
    assert engine._resolve_var(node, ctx) is None


# ---------------------------------------------------------------------------
# FA-08: output is dict — transparent skip applies only to scalars
# ---------------------------------------------------------------------------


def test_fa08_dict_output_transparent_skip_does_not_apply():
    # Tool returns {"output": "real_value", "other": 1}.
    # $step.output → dict has "output" key → skip does NOT apply → traverse → "real_value"
    ctx = ctx_with_step("step", {"output": "real_value", "other": 1})
    node = VarNode(name="step.output")
    assert engine._resolve_var(node, ctx) == "real_value"


def test_fa08_scalar_output_transparent_skip_applies():
    # Tool returns a plain scalar → $step.output → skip "output" → return scalar directly
    ctx = ctx_with_step("step", "plain_scalar")
    node = VarNode(name="step.output")
    assert engine._resolve_var(node, ctx) == "plain_scalar"


# ---------------------------------------------------------------------------
# FA-09: parse_condition tokeniser handles multi-level dotted $var
# ---------------------------------------------------------------------------


def test_fa09_tokeniser_multi_level():
    """Parser must produce VarNode with full dotted name."""
    node = parse_condition("$poll_payment.output.payment_status == 'SUCCESS'")
    assert isinstance(node, BinaryNode)
    assert isinstance(node.left, VarNode)
    assert node.left.name == "poll_payment.output.payment_status"
    assert isinstance(node.right, LitNode)
    assert node.right.value == "SUCCESS"


def test_fa09_tokeniser_three_levels():
    node = parse_condition("$a.output.b.c == 'x'")
    assert isinstance(node, BinaryNode)
    assert isinstance(node.left, VarNode)
    assert node.left.name == "a.output.b.c"


# ---------------------------------------------------------------------------
# FA-10: миmic vm._execute_condition behaviour without vm import
#        ctx = {**state.step_outputs, **state.data}
# ---------------------------------------------------------------------------


def test_fa10_vm_ctx_dotted_true():
    """Simulate _execute_condition: step_outputs merged into ctx."""
    step_outputs = {"poll_payment": {"payment_status": "SUCCESS", "amount": 50000}}
    data = {"user_id": "u1"}
    ctx = {**step_outputs, **data}

    result = eval_condition("$poll_payment.output.payment_status == 'SUCCESS'", ctx)
    assert result is True


def test_fa10_vm_ctx_dotted_false():
    step_outputs = {"poll_payment": {"payment_status": "PENDING"}}
    ctx = {**step_outputs}
    result = eval_condition("$poll_payment.output.payment_status == 'SUCCESS'", ctx)
    assert result is False


def test_fa10_vm_ctx_then_otherwise():
    """Verify the full branch selection mirrors _execute_condition logic."""
    step_outputs = {"poll_payment": {"payment_status": "SUCCESS"}}
    ctx = {**step_outputs}

    condition = "$poll_payment.output.payment_status == 'SUCCESS'"
    then = "finalize"
    otherwise = "retry"

    branch = then if eval_condition(condition, ctx) else otherwise
    assert branch == "finalize"


# ---------------------------------------------------------------------------
# FA-11: 'in' operator with dotted path
# ---------------------------------------------------------------------------


def test_fa11_in_operator_with_dotted():
    ctx = ctx_with_step("classify", {"category": "refund_request"})
    result = eval_condition("'refund' in $classify.output.category", ctx)
    assert result is True


def test_fa11_not_in_operator_with_dotted():
    ctx = ctx_with_step("classify", {"category": "info_request"})
    result = eval_condition("'refund' in $classify.output.category", ctx)
    assert result is False
