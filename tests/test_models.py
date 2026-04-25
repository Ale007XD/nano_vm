"""Тесты моделей — без IO, без LLM."""
import pytest
from nano_vm.models import (
    OnError, Program, StateContext, Step, StepResult,
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
    # frozen=True запрещает переприсвоение атрибута модели
    with pytest.raises(Exception):
        ctx.data = {"y": 2}
    # with_data возвращает новый объект, оригинал не тронут
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
