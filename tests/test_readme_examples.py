"""
tests/test_readme_examples.py
=============================
Q4: README doc-drift guard. Every claim in README.md that can be checked
mechanically is checked here, so the README cannot silently diverge from the
code it documents.

  RM-01  every ```python block parses (top-level await allowed)
  RM-02  every `from nano_vm... import X` in a block resolves
  RM-03  every Program.from_dict({...}) literal builds and passes
         ProgramValidator.is_valid()
  RM-04  Quick Start (tool pipeline) runs end-to-end -> SUCCESS
  RM-05  Quick Start (LLM pipeline) runs with MockLLMAdapter -> refund branch
  RM-06  Suspend / Resume example: SUSPENDED -> SUCCESS
  RM-07  condition-syntax warnings hold: method call -> ASTEvalError,
         quoted $var -> silent False, unquoted $var == "..." -> True
  RM-08  allowed_outputs on_error table (fail / skip) behaves as documented
  RM-09  TraceAnalyzer / receipt attributes shown in README exist
  RM-10  TraceStatus members listed in the Observability block exist
  RM-11  MockLLMAdapter forms shown under "Testing" behave as documented
  RM-12  README does not claim projection is enforced while vm.py has no
         projection call site (and the claim must be revisited when it does)
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from typing import Any

import pytest

from nano_vm import (
    ExecutionVM,
    Program,
    ProgramValidator,
    TraceAnalyzer,
    TraceStatus,
    WebhookEvent,
)
from nano_vm.adapters import MockLLMAdapter
from nano_vm.ast_engine import ASTEvalError, eval_condition

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

_FENCE = re.compile(r"^```python[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
BLOCKS: list[str] = _FENCE.findall(README)

_TOP_LEVEL_AWAIT = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT | ast.PyCF_ONLY_AST


def _parse(src: str) -> ast.Module:
    return compile(src, "<readme>", "exec", flags=_TOP_LEVEL_AWAIT)  # type: ignore[return-value]


def _block_with(needle: str) -> str:
    for b in BLOCKS:
        if needle in b:
            return b
    raise AssertionError(f"README has no python block containing {needle!r}")


def _from_dict_literals() -> list[tuple[int, dict[str, Any]]]:
    out: list[tuple[int, dict[str, Any]]] = []
    for i, src in enumerate(BLOCKS):
        for node in ast.walk(_parse(src)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_dict"
                and node.args
            ):
                out.append((i, ast.literal_eval(node.args[0])))
    return out


# ---------------------------------------------------------------------------
# RM-01 / RM-02 — blocks parse, imports resolve
# ---------------------------------------------------------------------------


def test_rm01_readme_has_python_blocks() -> None:
    assert len(BLOCKS) >= 10  # fence regex must actually find the blocks


@pytest.mark.parametrize("idx", range(len(BLOCKS)))
def test_rm01_block_parses(idx: int) -> None:
    _parse(BLOCKS[idx])


def _readme_imports() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for src in BLOCKS:
        for node in ast.walk(_parse(src)):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] == "nano_vm"
            ):
                pairs.extend((node.module, a.name) for a in node.names)
    return sorted(set(pairs))


@pytest.mark.parametrize(("module", "name"), _readme_imports())
def test_rm02_readme_import_resolves(module: str, name: str) -> None:
    if (module, name) == ("nano_vm.adapters", "LiteLLMAdapter"):
        pytest.importorskip("litellm")  # lazy by design on bare install
    assert hasattr(importlib.import_module(module), name)


# ---------------------------------------------------------------------------
# RM-03 — program literals are valid programs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("idx", "literal"), _from_dict_literals())
def test_rm03_program_literal_is_valid(idx: int, literal: dict[str, Any]) -> None:
    program = Program.from_dict(literal)
    report = ProgramValidator(program).validate()
    assert report.is_valid(), [str(i) for i in report.issues]


# ---------------------------------------------------------------------------
# RM-04 / RM-05 — Quick Starts actually run
# ---------------------------------------------------------------------------


async def test_rm04_quick_start_tool_pipeline() -> None:
    literal = dict(next(lit for i, lit in _from_dict_literals() if "payment_flow" in str(lit)))
    calls: list[str] = []

    def _tool(name: str) -> Any:
        async def fn(**kwargs: Any) -> str:
            calls.append(name)
            return "ok"

        return fn

    vm = ExecutionVM(
        tools={n: _tool(n) for n in ("reserve_funds", "capture_payment", "send_receipt")}
    )  # llm omitted: README says LLMs are not required
    trace = await vm.run(Program.from_dict(literal))
    assert trace.status == TraceStatus.SUCCESS
    assert calls == ["reserve_funds", "capture_payment", "send_receipt"]


async def test_rm05_quick_start_llm_pipeline() -> None:
    literal = next(lit for _, lit in _from_dict_literals() if lit["name"] == "customer_refund")
    calls: list[str] = []

    def _tool(name: str) -> Any:
        async def fn(**kwargs: Any) -> str:
            calls.append(name)
            return "done"

        return fn

    vm = ExecutionVM(
        llm=MockLLMAdapter("yes"),
        tools={"issue_refund": _tool("issue_refund"), "send_rejection": _tool("send_rejection")},
    )
    trace = await vm.run(Program.from_dict(literal), context={"user_input": "I was charged twice"})
    assert trace.status == TraceStatus.SUCCESS
    assert calls == ["issue_refund"]  # `$decision == "yes"` took the then-branch

    calls.clear()
    vm = ExecutionVM(
        llm=MockLLMAdapter("no"),
        tools={"issue_refund": _tool("issue_refund"), "send_rejection": _tool("send_rejection")},
    )
    trace = await vm.run(Program.from_dict(literal), context={"user_input": "hi"})
    assert trace.status == TraceStatus.SUCCESS
    assert calls == ["send_rejection"]


# ---------------------------------------------------------------------------
# RM-06 — suspend / resume
# ---------------------------------------------------------------------------


async def test_rm06_suspend_resume_example() -> None:
    program = Program.from_dict(
        {
            "name": "pay",
            "steps": [
                {"id": "initiate", "type": "tool", "tool": "initiate_payment"},
                {"id": "finish", "type": "tool", "tool": "finish", "is_terminal": True},
            ],
        }
    )

    async def initiate_payment(**kwargs: Any) -> str:
        return "PENDING"

    async def finish(**kwargs: Any) -> str:
        return "ok"

    vm = ExecutionVM(tools={"initiate_payment": initiate_payment, "finish": finish})
    trace = await vm.run(program, context={"order_id": "123"})
    assert trace.status.name == "SUSPENDED"

    trace = await vm.resume_with_program(
        webhook_event=WebhookEvent(
            trace_id=trace.trace_id,
            payload={"type": "payment.confirmed", "order_id": "123"},
        ),
        program=program,
    )
    assert trace.status.name == "SUCCESS"


# ---------------------------------------------------------------------------
# RM-07 — condition-syntax claims
# ---------------------------------------------------------------------------


def test_rm07_method_call_raises_at_parse_time() -> None:
    with pytest.raises(ASTEvalError):
        eval_condition("$decision.lower() == 'yes'", {"decision": "yes"})


def test_rm07_quoted_var_is_silently_false() -> None:
    # README: "No error is raised; this condition is always False."
    assert eval_condition("'yes' in '$decision'", {"decision": "yes"}) is False


def test_rm07_unquoted_var_equality_works() -> None:
    assert eval_condition('$decision == "yes"', {"decision": "yes"}) is True
    assert eval_condition('$decision == "yes"', {"decision": "no"}) is False


# ---------------------------------------------------------------------------
# RM-08 — allowed_outputs on_error table
# ---------------------------------------------------------------------------


def _classify_program(on_error: str) -> Program:
    return Program.from_dict(
        {
            "name": "classify",
            "steps": [
                {
                    "id": "classify",
                    "type": "llm",
                    "prompt": "Classify",
                    "output_key": "category",
                    "allowed_outputs": ["refund", "query", "other"],
                    "on_error": on_error,
                    "is_terminal": True,
                }
            ],
        }
    )


async def test_rm08_on_error_fail_marks_trace_failed() -> None:
    vm = ExecutionVM(llm=MockLLMAdapter("banana"))
    trace = await vm.run(_classify_program("fail"))
    assert trace.status == TraceStatus.FAILED


async def test_rm08_on_error_skip_uses_first_allowed_output() -> None:
    vm = ExecutionVM(llm=MockLLMAdapter("banana"))
    trace = await vm.run(_classify_program("skip"))
    assert trace.status == TraceStatus.SUCCESS
    assert trace.final_output == "refund"  # allowed_outputs[0]


# ---------------------------------------------------------------------------
# RM-09 / RM-10 — analyzer, receipt, status names shown in README
# ---------------------------------------------------------------------------

_REPORT_ATTRS = (
    "rollback_density",
    "tool_churn_rate",
    "path_variance",
    "transition_entropy",
    "invariant_violation_rate",
)
_RECEIPT_ATTRS = (
    "trace_id",
    "final_status",
    "resumable",
    "replayable",
    "failed_steps",
    "retried_steps",
    "rejected_transitions",
    "health",
)


async def _some_trace() -> Any:
    async def t(**kwargs: Any) -> str:
        return "ok"

    program = Program.from_dict(
        {
            "name": "p",
            "steps": [{"id": "a", "type": "tool", "tool": "t", "is_terminal": True}],
        }
    )
    return await ExecutionVM(tools={"t": t}).run(program)


async def test_rm09_report_and_receipt_attributes_exist() -> None:
    analyzer = TraceAnalyzer(await _some_trace())
    report = analyzer.report()
    receipt = analyzer.receipt()
    for attr in _REPORT_ATTRS:
        assert hasattr(report, attr), attr
        assert attr in README, f"{attr} no longer documented"
    for attr in _RECEIPT_ATTRS:
        assert hasattr(receipt, attr), attr
        assert attr in README, f"{attr} no longer documented"


def test_rm10_observability_status_names_exist() -> None:
    block = _block_with("TraceStatus.SUCCESS |")
    line = next(ln for ln in block.splitlines() if "TraceStatus.SUCCESS" in ln)
    names = re.findall(r"[A-Z_]{4,}", line.split("TraceStatus.", 1)[1])
    assert "CANCELLED" in names
    for n in names:
        assert n in TraceStatus.__members__, n


# ---------------------------------------------------------------------------
# RM-11 — MockLLMAdapter forms under "Testing"
# ---------------------------------------------------------------------------


async def test_rm11_mock_adapter_forms() -> None:
    msg = [{"role": "user", "content": "x"}]

    assert await MockLLMAdapter("yes").complete(msg) == "yes"

    seq = MockLLMAdapter(["SAFE", "yes"])
    assert [await seq.complete(msg), await seq.complete(msg)] == ["SAFE", "yes"]

    mapped = MockLLMAdapter({"Classify": "SAFE", "eligible": "yes", "__default__": "ok"})
    assert await mapped.complete([{"role": "user", "content": "Classify this"}]) == "SAFE"
    assert await mapped.complete([{"role": "user", "content": "unmatched"}]) == "ok"


# ---------------------------------------------------------------------------
# RM-12 — projection claim must match the code
# ---------------------------------------------------------------------------


def test_rm12_projection_claim_matches_code() -> None:
    vm_src = (ROOT / "nano_vm" / "vm.py").read_text(encoding="utf-8")
    wired = bool(re.search(r"Projection|DeterministicSanitizer|\.project\(", vm_src))
    claims_not_enforced = "does not route model-facing state through" in README
    assert "structural, not configured" not in README
    # While vm.py has no projection call site the README must say so; once it
    # does, this test fails and the README section must be rewritten.
    assert wired != claims_not_enforced, (
        "README 'What the LLM sees' and nano_vm/vm.py disagree about whether "
        "ProjectionLayer is wired into the LLM path"
    )
