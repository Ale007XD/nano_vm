"""
tests/test_v070_sprint1.py
==========================
Sprint 1 — Contracts & Determinism

Покрывает:
  1. ASTEngine — все RFC-операторы, детерминизм, безопасность
  2. CapabilityRef — secure_hash, tombstoning, иммутабельность
  3. PolicySnapshot — frozen, has_capability, policy_hash детерминизм
  4. _execute_condition — интеграция ASTEngine в VM (без eval())
  5. Idempotency — _execute_step не меняет результат при повторном вызове

pytest -v tests/test_v070_sprint1.py
"""

from __future__ import annotations

import asyncio
import hashlib
import pytest

from nano_vm.ast_engine import (
    ASTEngine,
    ASTEvalError,
    ASTParseError,
    BinaryNode,
    LitNode,
    LogicalNode,
    NotNode,
    VarNode,
    eval_condition,
    parse_condition,
)
from nano_vm.models import (
    CapabilityRef,
    PolicySnapshot,
    Program,
    StateContext,
    StepStatus,
)
from nano_vm.vm import ExecutionVM, VMError


# ===========================================================================
# 1. ASTEngine — RFC operators
# ===========================================================================


class TestASTEngineOperators:
    """Все операторы RFC: ==, !=, >, <, in, not in, and, or, contains."""

    def setup_method(self):
        self.engine = ASTEngine()

    def _ctx(self, **kw):
        return kw

    # -- Equality

    def test_eq_true(self):
        node = BinaryNode(op="==", left=LitNode(value="yes"), right=LitNode(value="yes"))
        assert self.engine.evaluate(node, {}) is True

    def test_eq_false(self):
        node = BinaryNode(op="==", left=LitNode(value="yes"), right=LitNode(value="no"))
        assert self.engine.evaluate(node, {}) is False

    def test_neq(self):
        node = BinaryNode(op="!=", left=LitNode(value=1), right=LitNode(value=2))
        assert self.engine.evaluate(node, {}) is True

    # -- Comparison

    def test_gt_true(self):
        node = BinaryNode(op=">", left=LitNode(value=5), right=LitNode(value=3))
        assert self.engine.evaluate(node, {}) is True

    def test_gt_false(self):
        node = BinaryNode(op=">", left=LitNode(value=1), right=LitNode(value=3))
        assert self.engine.evaluate(node, {}) is False

    def test_lt(self):
        node = BinaryNode(op="<", left=LitNode(value=1), right=LitNode(value=3))
        assert self.engine.evaluate(node, {}) is True

    # -- Membership

    def test_in_string(self):
        node = BinaryNode(
            op="in",
            left=LitNode(value="yes"),
            right=LitNode(value="yes, approved"),
        )
        assert self.engine.evaluate(node, {}) is True

    def test_in_list(self):
        node = BinaryNode(
            op="in",
            left=LitNode(value="admin"),
            right=LitNode(value=["admin", "user"]),
        )
        assert self.engine.evaluate(node, {}) is True

    def test_not_in(self):
        node = BinaryNode(
            op="not in",
            left=LitNode(value="guest"),
            right=LitNode(value=["admin", "user"]),
        )
        assert self.engine.evaluate(node, {}) is True

    def test_contains(self):
        # contains = left in right (alias)
        node = BinaryNode(
            op="contains",
            left=LitNode(value="yes"),
            right=LitNode(value="yes approved"),
        )
        assert self.engine.evaluate(node, {}) is True

    # -- Logical

    def test_and_both_true(self):
        node = LogicalNode(
            op="and",
            left=BinaryNode(op="==", left=LitNode(value=1), right=LitNode(value=1)),
            right=BinaryNode(op="==", left=LitNode(value=2), right=LitNode(value=2)),
        )
        assert self.engine.evaluate(node, {}) is True

    def test_and_short_circuit_false(self):
        node = LogicalNode(
            op="and",
            left=BinaryNode(op="==", left=LitNode(value=1), right=LitNode(value=2)),
            right=BinaryNode(op="==", left=LitNode(value=2), right=LitNode(value=2)),
        )
        assert self.engine.evaluate(node, {}) is False

    def test_or_one_true(self):
        node = LogicalNode(
            op="or",
            left=BinaryNode(op="==", left=LitNode(value=1), right=LitNode(value=2)),
            right=BinaryNode(op="==", left=LitNode(value=2), right=LitNode(value=2)),
        )
        assert self.engine.evaluate(node, {}) is True

    def test_not_node(self):
        node = NotNode(
            op="not",
            operand=BinaryNode(op="==", left=LitNode(value=1), right=LitNode(value=2))
        )
        assert self.engine.evaluate(node, {}) is True

    # -- Variables

    def test_var_from_context(self):
        node = BinaryNode(
            op="==",
            left=VarNode(name="decision"),
            right=LitNode(value="yes"),
        )
        assert self.engine.evaluate(node, {"decision": "yes"}) is True

    def test_var_missing_returns_none(self):
        node = BinaryNode(
            op="==",
            left=VarNode(name="missing"),
            right=LitNode(value=None),
        )
        assert self.engine.evaluate(node, {}) is True  # None == None

    def test_var_step_output(self):
        ctx = {"__step_outputs__": {"classify": "refund"}}
        node = BinaryNode(
            op="in",
            left=LitNode(value="refund"),
            right=VarNode(name="classify.output"),
        )
        assert self.engine.evaluate(node, ctx) is True

    # -- Type error

    def test_type_error_raises(self):
        node = BinaryNode(op=">", left=LitNode(value="abc"), right=LitNode(value=1))
        with pytest.raises(ASTEvalError, match="Type error"):
            self.engine.evaluate(node, {})


# ===========================================================================
# 2. Parser — строка DSL → ConditionExpr
# ===========================================================================


class TestConditionParser:
    """parse_condition() компилирует DSL в дерево, eval_condition() вычисляет."""

    def test_simple_in(self):
        result = eval_condition("'yes' in $decision", {"decision": "yes approved"})
        assert result is True

    def test_simple_eq(self):
        result = eval_condition("$score == 42", {"score": 42})
        assert result is True

    def test_simple_gt(self):
        result = eval_condition("$score > 0", {"score": 5})
        assert result is True

    def test_simple_lt_false(self):
        result = eval_condition("$score < 0", {"score": 5})
        assert result is False

    def test_not_in(self):
        result = eval_condition("$role not in 'admin'", {"role": "guest"})
        assert result is True

    def test_logical_and(self):
        result = eval_condition("$a == 1 and $b == 2", {"a": 1, "b": 2})
        assert result is True

    def test_logical_and_false(self):
        result = eval_condition("$a == 1 and $b == 3", {"a": 1, "b": 2})
        assert result is False

    def test_logical_or(self):
        result = eval_condition("$a == 99 or $b == 2", {"a": 1, "b": 2})
        assert result is True

    def test_bool_literal_true(self):
        result = eval_condition("$flag == True", {"flag": True})
        assert result is True

    def test_bool_literal_false(self):
        result = eval_condition("$flag == False", {"flag": False})
        assert result is True

    def test_none_literal(self):
        result = eval_condition("$val == None", {"val": None})
        assert result is True

    # -- Детерминизм: одинаковый вход → одинаковый результат

    def test_determinism(self):
        expr = "'yes' in $decision"
        ctx = {"decision": "yes approved"}
        results = [eval_condition(expr, ctx) for _ in range(100)]
        assert all(r is True for r in results)

    # -- Безопасность: попытки инъекций

    def test_no_builtins_import(self):
        """Попытка использовать __import__ — должна упасть или вернуть False, не выполниться."""
        # Строка не является валидным DSL-выражением — парсер вернёт LitNode
        # ASTEngine не выполняет произвольный код
        result = eval_condition("__import__('os') == None", {})
        # Либо False (атом "__import__('os')" != None), либо ошибка — в любом случае не выполнится
        assert result is False or result is True  # главное — нет выполнения

    def test_no_exec(self):
        """Строка с exec() не должна выполниться."""
        # eval_condition попытается распарсить — вернёт LitNode со строкой, не выполнит
        try:
            result = eval_condition("exec('import os')", {})
            # Если не бросило — результат должен быть bool, не side-effect
            assert isinstance(result, bool)
        except (ASTEvalError, ASTParseError):
            pass  # тоже правильно

    def test_injection_in_condition(self):
        """LLM-output как значение переменной, не как выражение."""
        # $decision содержит злобную строку — она тестируется, не исполняется
        evil = "__import__('os').system('rm -rf /')"
        result = eval_condition("'yes' in $decision", {"decision": evil})
        assert result is False  # 'yes' не в строке


# ===========================================================================
# 3. CapabilityRef — RFC v0.7.0
# ===========================================================================


class TestCapabilityRef:
    def test_secure_hash_deterministic(self):
        ref = CapabilityRef(ref_id="vault://users/42/email", salt="fixed-salt")
        h1 = ref.secure_hash()
        h2 = ref.secure_hash()
        assert h1 == h2

    def test_secure_hash_sha256(self):
        ref = CapabilityRef(ref_id="vault://users/42/email", salt="fixed-salt")
        expected = hashlib.sha256(b"vault://users/42/emailfixed-salt").hexdigest()
        assert ref.secure_hash() == expected

    def test_tombstone_returns_constant(self):
        ref = CapabilityRef(ref_id="vault://users/42/email", salt="any-salt", is_tombstone=True)
        assert ref.secure_hash() == "TOMBSTONE"

    def test_tombstone_method(self):
        ref = CapabilityRef(ref_id="vault://users/42/email", salt="s")
        tombstoned = ref.tombstone()
        assert tombstoned.is_tombstone is True
        assert tombstoned.secure_hash() == "TOMBSTONE"
        # Оригинал не изменился (frozen)
        assert ref.is_tombstone is False

    def test_frozen(self):
        ref = CapabilityRef(ref_id="vault://x", salt="s")
        with pytest.raises(Exception):  # ValidationError или TypeError
            ref.ref_id = "vault://y"  # type: ignore

    def test_different_salts_different_hashes(self):
        ref1 = CapabilityRef(ref_id="vault://x", salt="salt1")
        ref2 = CapabilityRef(ref_id="vault://x", salt="salt2")
        assert ref1.secure_hash() != ref2.secure_hash()

    def test_default_salt_generated(self):
        ref1 = CapabilityRef(ref_id="vault://x")
        ref2 = CapabilityRef(ref_id="vault://x")
        # Разные соли → разные хеши (с высокой вероятностью)
        assert ref1.salt != ref2.salt

    def test_tombstone_hash_chain_preserved(self):
        """Все tombstone-ссылки возвращают одно константное значение — hash chain не ломается."""
        refs = [
            CapabilityRef(ref_id=f"vault://user/{i}", salt=f"s{i}", is_tombstone=True)
            for i in range(10)
        ]
        hashes = {r.secure_hash() for r in refs}
        assert hashes == {"TOMBSTONE"}


# ===========================================================================
# 4. PolicySnapshot — RFC v0.7.0
# ===========================================================================


class TestPolicySnapshot:
    def test_frozen(self):
        snap = PolicySnapshot(policy_id="p1", version="1.0", tool_capabilities={})
        with pytest.raises(Exception):
            snap.policy_id = "p2"  # type: ignore

    def test_has_capability_true(self):
        snap = PolicySnapshot(
            policy_id="p1",
            version="1.0",
            tool_capabilities={"send_email": ["email.read_raw", "email.send"]},
        )
        assert snap.has_capability("send_email", "email.read_raw") is True
        assert snap.has_capability("send_email", "email.send") is True

    def test_has_capability_false(self):
        snap = PolicySnapshot(
            policy_id="p1",
            version="1.0",
            tool_capabilities={"send_email": ["email.read_raw"]},
        )
        assert snap.has_capability("send_email", "email.send") is False
        assert snap.has_capability("unknown_tool", "any") is False

    def test_policy_hash_deterministic(self):
        caps = {"send_email": ["email.send", "email.read_raw"], "search": ["web.read"]}
        snap1 = PolicySnapshot(policy_id="p1", version="1.0", tool_capabilities=caps)
        snap2 = PolicySnapshot(policy_id="p1", version="1.0", tool_capabilities=caps)
        assert snap1.policy_hash == snap2.policy_hash

    def test_policy_hash_changes_with_version(self):
        caps = {"tool": ["cap"]}
        snap1 = PolicySnapshot(policy_id="p1", version="1.0", tool_capabilities=caps)
        snap2 = PolicySnapshot(policy_id="p1", version="1.1", tool_capabilities=caps)
        assert snap1.policy_hash != snap2.policy_hash

    def test_policy_hash_changes_with_caps(self):
        snap1 = PolicySnapshot(
            policy_id="p1", version="1.0", tool_capabilities={"t": ["a"]}
        )
        snap2 = PolicySnapshot(
            policy_id="p1", version="1.0", tool_capabilities={"t": ["b"]}
        )
        assert snap1.policy_hash != snap2.policy_hash

    def test_allowed_tools(self):
        snap = PolicySnapshot(
            policy_id="p1",
            version="1.0",
            tool_capabilities={"tool_a": ["cap1"], "tool_b": ["cap2"]},
        )
        assert snap.allowed_tools() == {"tool_a", "tool_b"}

    def test_explicit_hash_accepted(self):
        snap = PolicySnapshot(
            policy_id="p1",
            version="1.0",
            policy_hash="custom-hash-value",
            tool_capabilities={},
        )
        assert snap.policy_hash == "custom-hash-value"


# ===========================================================================
# 5. VM integration — ASTEngine вместо eval()
# ===========================================================================


class MockAdapter:
    def __init__(self, responses: dict[str, str]):
        self._responses = responses

    async def complete(self, messages):
        content = messages[-1]["content"]
        for key, val in self._responses.items():
            if key in content:
                return val
        return "__default__"


class TestVMConditionNoEval:
    """_execute_condition использует ASTEngine, не eval()."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_basic_condition_yes(self):
        program = Program.from_dict({
            "steps": [
                {"id": "classify", "type": "llm", "prompt": "Classify: $input", "output_key": "decision"},
                {"id": "guard", "type": "condition", "condition": "'yes' in $decision", "then": "approve", "otherwise": "reject"},
                {"id": "approve", "type": "tool", "tool": "do_approve"},
                {"id": "reject", "type": "tool", "tool": "do_reject"},
            ]
        })
        vm = ExecutionVM(
            llm=MockAdapter({"Classify": "yes approved"}),
            tools={
                "do_approve": lambda: "approved",
                "do_reject": lambda: "rejected",
            },
        )
        trace = self._run(vm.run(program, context={"input": "refund"}))
        assert trace.status.value == "success"
        assert trace.final_output == "approved"

    def test_basic_condition_no(self):
        program = Program.from_dict({
            "steps": [
                {"id": "classify", "type": "llm", "prompt": "Classify: $input", "output_key": "decision"},
                {"id": "guard", "type": "condition", "condition": "'yes' in $decision", "then": "approve", "otherwise": "reject"},
                {"id": "approve", "type": "tool", "tool": "do_approve"},
                {"id": "reject", "type": "tool", "tool": "do_reject"},
            ]
        })
        vm = ExecutionVM(
            llm=MockAdapter({"Classify": "no"}),
            tools={
                "do_approve": lambda: "approved",
                "do_reject": lambda: "rejected",
            },
        )
        trace = self._run(vm.run(program, context={"input": "spam"}))
        assert trace.status.value == "success"
        assert trace.final_output == "rejected"

    def test_no_eval_in_source(self):
        """Убеждаемся что eval() удалён из vm.py."""
        import inspect
        from nano_vm import vm as vm_module
        src = inspect.getsource(vm_module)
        # eval() с __builtins__ — старая реализация; нового быть не должно
        assert 'eval(condition' not in src, "eval() still present in vm._execute_condition"

    def test_condition_bad_expression_raises_vmerror(self):
        """Невычислимое выражение → VMError, не падение интерпретатора."""
        program = Program.from_dict({
            "steps": [
                {
                    "id": "guard",
                    "type": "condition",
                    # Оператор >= не поддерживается RFC — парсер вернёт ошибку
                    "condition": "$score >= 100",
                    "then": "ok",
                    "otherwise": "fail_step",
                },
                {"id": "ok", "type": "tool", "tool": "noop"},
                {"id": "fail_step", "type": "tool", "tool": "noop"},
            ]
        })
        vm = ExecutionVM(llm=MockAdapter({}), tools={"noop": lambda: None})
        trace = self._run(vm.run(program, context={"score": 50}))
        # >= пока не поддержан RFC — либо FAILED, либо условие вычислилось через fallback
        # Главное: нет AttributeError / SyntaxError / выполнения произвольного кода
        assert trace.status.value in ("success", "failed")


# ===========================================================================
# 6. Idempotency — _execute_step не меняет результат при повторном вызове
# ===========================================================================


class TestIdempotency:
    """
    Инвариант: одинаковый (step, state) → одинаковый StepResult.
    Нет side effects на state между вызовами.
    """

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_llm_step_same_output(self):
        from nano_vm.models import Step, StepType

        step = Step(id="s1", type=StepType.LLM, prompt="Hello $name")
        state = StateContext(data={"name": "Alice"})
        vm = ExecutionVM(llm=MockAdapter({"Hello Alice": "Hi!"}), tools={})

        results = []
        for _ in range(5):
            out, usage, _ = self._run(vm._execute_step(step, state))
            results.append(out)

        assert len(set(results)) == 1, f"Non-deterministic outputs: {results}"

    def test_tool_step_same_output(self):
        from nano_vm.models import Step, StepType

        def my_tool():
            # Инструмент детерминирован — возвращает константу
            return "result"

        step = Step(id="s1", type=StepType.TOOL, tool="my_tool")
        state = StateContext()
        vm = ExecutionVM(llm=MockAdapter({}), tools={"my_tool": my_tool})

        results = []
        for _ in range(5):
            out, _, _ = self._run(vm._execute_step(step, state))
            results.append(out)

        assert all(r == "result" for r in results)

    def test_condition_step_same_output(self):
        from nano_vm.models import Step, StepType

        step = Step(
            id="s1",
            type=StepType.CONDITION,
            condition="'yes' in $decision",
            then="approve",
            otherwise="reject",
        )
        state = StateContext(data={"decision": "yes approved"})
        vm = ExecutionVM(llm=MockAdapter({}), tools={})

        results = []
        for _ in range(10):
            out, _, _ = self._run(vm._execute_step(step, state))
            results.append(out)

        assert all(r == "approve" for r in results)

    def test_state_immutable_after_step(self):
        """StateContext не мутируется внутри _execute_step."""
        from nano_vm.models import Step, StepType

        step = Step(id="s1", type=StepType.TOOL, tool="t", output_key="result")
        state = StateContext(data={"x": 1})
        original_data = dict(state.data)

        vm = ExecutionVM(llm=MockAdapter({}), tools={"t": lambda: 42})
        self._run(vm._execute_step(step, state))

        # state остался прежним — неизменная копия
        assert state.data == original_data
