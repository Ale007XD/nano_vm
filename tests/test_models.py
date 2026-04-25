"""Tests for models — no IO, no LLM."""
import pytest
from nano_vm.models import (
    LLMUsage, OnError, Program, StateContext, Step, StepResult,
    StepStatus, StepType, Trace, TraceStatus,
)


def test_step_llm_requires_prompt():
    with pytest.raises(Exception):
        Step(id="s1", type=StepType.LLM)


def test_step_tool_requires_tool():
    with pytest.raises(Exception):
        Step(id="s1", type=StepType.TOOL)


def test_step_condition_requires_condition():
    with pytest.raises(Exception):
        Step(id="s1", type=StepType.CONDITION)


def test_valid_llm_step():
    s = Step(id="s1", type=StepType.LLM, prompt="Hello")
    assert s.on_error == OnError.FAIL
    assert s.max_retries == 1


def test_state_context_immutable():
    ctx = StateContext(data={"x": 1})
    with pytest.raises(Exception):
        ctx.data = {"y": 2}
    ctx2 = ctx.with_data("y", 2)
    assert ctx2.data["y"] == 2
    assert "y" not in ctx.data


def test_state_context_with_output():
    ctx = StateContext()
    ctx2 = ctx.with_output("step_1", "result")
    assert ctx2.step_outputs["step_1"] == "result"
    assert ctx.step_outputs == {}


def test_program_from_dict():
    data = {
        "name": "test",
        "steps": [{"id": "s1", "type": "llm", "prompt": "Hi"}]
    }
    p = Program.from_dict(data)
    assert p.name == "test"
    assert len(p.steps) == 1


def test_program_get_step():
    data = {
        "steps": [
            {"id": "a", "type": "llm", "prompt": "Hi"},
            {"id": "b", "type": "llm", "prompt": "Bye"},
        ]
    }
    p = Program.from_dict(data)
    assert p.get_step("a").id == "a"
    assert p.get_step("z") is None


def test_step_result_finish_success():
    r = StepResult(step_id="s1", status=StepStatus.RUNNING)
    r2 = r.finish(output="hello")
    assert r2.status == StepStatus.SUCCESS
    assert r2.output == "hello"
    assert r2.duration_ms is not None
    assert r2.usage is None


def test_step_result_finish_with_usage():
    r = StepResult(step_id="s1", status=StepStatus.RUNNING)
    usage = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost_usd=0.000015)
    r2 = r.finish(output="hello", usage=usage)
    assert r2.usage.total_tokens == 150
    assert r2.usage.cost_usd == 0.000015


def test_step_result_finish_error():
    r = StepResult(step_id="s1", status=StepStatus.RUNNING)
    r2 = r.finish(error="oops")
    assert r2.status == StepStatus.FAILED
    assert r2.error == "oops"


def test_trace_add_step_and_last_output():
    t = Trace(program_name="test")
    r = StepResult(step_id="s1", status=StepStatus.RUNNING).finish(output="42")
    t2 = t.add_step(r)
    assert len(t2.steps) == 1
    assert t2.last_output() == "42"


def test_trace_finish():
    t = Trace(program_name="test")
    t2 = t.finish(TraceStatus.SUCCESS, final_output="done")
    assert t2.status == TraceStatus.SUCCESS
    assert t2.final_output == "done"
    assert t2.duration_ms is not None


def test_trace_total_tokens():
    t = Trace(program_name="test")
    r1 = StepResult(step_id="s1", status=StepStatus.RUNNING).finish(
        output="a", usage=LLMUsage(total_tokens=100)
    )
    r2 = StepResult(step_id="s2", status=StepStatus.RUNNING).finish(
        output="b", usage=LLMUsage(total_tokens=200)
    )
    t = t.add_step(r1).add_step(r2)
    assert t.total_tokens() == 300


def test_trace_total_cost():
    t = Trace(program_name="test")
    r1 = StepResult(step_id="s1", status=StepStatus.RUNNING).finish(
        output="a", usage=LLMUsage(total_tokens=100, cost_usd=0.0001)
    )
    r2 = StepResult(step_id="s2", status=StepStatus.RUNNING).finish(
        output="b", usage=LLMUsage(total_tokens=200, cost_usd=0.0002)
    )
    t = t.add_step(r1).add_step(r2)
    assert abs(t.total_cost_usd() - 0.0003) < 1e-9


def test_llm_usage_str():
    u = LLMUsage(total_tokens=150, cost_usd=0.000015)
    assert "150" in str(u)
    assert "0.000015" in str(u)
