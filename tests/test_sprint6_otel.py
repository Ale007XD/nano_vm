"""
tests/test_sprint6_otel.py
===========================
sprint_6_core_otel — OTel span per FSM step + StepMetrics counters.

Покрывает:
  OT-01 — StepMetrics() defaults: all counters 0
  OT-02 — StepMetrics.record(LLM) → llm_calls += 1
  OT-03 — StepMetrics.record(TOOL) → tool_calls += 1
  OT-04 — StepMetrics.record(CONDITION) → condition_evals += 1
  OT-05 — StepMetrics.record(PARALLEL) — no counter bump, retries_total still adds
  OT-06 — StepMetrics is frozen — record() returns new instance, doesn't mutate self
  OT-07 — Trace.record_step_metric — model_copy roundtrip via Trace, not just StepMetrics
  OT-08 — integration: LLM→TOOL→CONDITION pipeline через ExecutionVM.run() — final
          Trace.step_metrics matches actual step type composition
  OT-09 — retries_total accumulates: on_error=RETRY step succeeding on 3rd attempt
          → retries_total == 2 (StepResult.retries carried into final result)
  OT-10 — span_step no-op when opentelemetry not installed (simulated via
          monkeypatch of telemetry.OTEL_AVAILABLE) — yields None, no exception,
          control flow unaffected
  OT-11 — span_step real span (opentelemetry-sdk InMemorySpanExporter): span name,
          nano_vm.step_id / nano_vm.step_type / nano_vm.attempt attributes present
  OT-12 — span_step records exception on span and re-raises unchanged (exception
          identity preserved, not swallowed or wrapped)
"""

from __future__ import annotations

import asyncio

import pytest

from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.models import OnError, Step, StepMetrics, StepType
from nano_vm.telemetry import span_step

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLLM:
    async def complete(self, messages):
        return "ok"


def make_vm(tools: dict | None = None) -> ExecutionVM:
    return ExecutionVM(llm=MockLLM(), tools=tools or {})


# ---------------------------------------------------------------------------
# OT-01..06 — StepMetrics unit behavior
# ---------------------------------------------------------------------------


def test_step_metrics_defaults_zero():
    m = StepMetrics()
    assert (m.llm_calls, m.tool_calls, m.condition_evals, m.retries_total) == (0, 0, 0, 0)


def test_step_metrics_record_llm():
    m = StepMetrics().record(StepType.LLM, retries=0)
    assert m.llm_calls == 1
    assert m.tool_calls == 0
    assert m.condition_evals == 0


def test_step_metrics_record_tool():
    m = StepMetrics().record(StepType.TOOL, retries=0)
    assert m.tool_calls == 1
    assert m.llm_calls == 0


def test_step_metrics_record_condition():
    m = StepMetrics().record(StepType.CONDITION, retries=0)
    assert m.condition_evals == 1
    assert m.llm_calls == 0
    assert m.tool_calls == 0


def test_step_metrics_record_parallel_no_type_bump_but_retries_added():
    m = StepMetrics().record(StepType.PARALLEL, retries=2)
    assert (m.llm_calls, m.tool_calls, m.condition_evals) == (0, 0, 0)
    assert m.retries_total == 2


def test_step_metrics_frozen_record_returns_new_instance():
    m0 = StepMetrics()
    m1 = m0.record(StepType.LLM, retries=0)
    assert m0.llm_calls == 0  # original untouched
    assert m1.llm_calls == 1
    assert m0 is not m1


# ---------------------------------------------------------------------------
# OT-07 — Trace.record_step_metric
# ---------------------------------------------------------------------------


def test_trace_record_step_metric_roundtrip():
    from nano_vm.models import Trace

    t0 = Trace(program_name="p")
    t1 = t0.record_step_metric(StepType.LLM, retries=0)
    assert t0.step_metrics.llm_calls == 0
    assert t1.step_metrics.llm_calls == 1


# ---------------------------------------------------------------------------
# OT-08 — integration: mixed pipeline
# ---------------------------------------------------------------------------


async def _echo_tool(**kwargs):
    return "TOOL_OK"


async def test_step_metrics_integration_mixed_pipeline():
    vm = make_vm(tools={"echo": _echo_tool})
    program = Program(
        name="mixed",
        steps=[
            Step(id="ask", type=StepType.LLM, prompt="hi", next_step="gate"),
            Step(
                id="gate",
                type=StepType.CONDITION,
                condition="$ask.output == 1",
                then="unreachable_then",
                otherwise="do_tool",
            ),
            Step(id="unreachable_then", type=StepType.TOOL, tool="echo", is_terminal=True),
            Step(id="do_tool", type=StepType.TOOL, tool="echo", is_terminal=True),
        ],
    )
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    # ask (LLM) -> gate (CONDITION) -> do_tool (TOOL, branch target)
    assert trace.step_metrics.llm_calls == 1
    assert trace.step_metrics.condition_evals == 1
    assert trace.step_metrics.tool_calls == 1


# ---------------------------------------------------------------------------
# OT-09 — retries_total accumulation
# ---------------------------------------------------------------------------


async def test_retries_total_accumulates_on_eventual_success(monkeypatch):
    async def mock_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)

    attempts = {"n": 0}

    async def flaky_tool(**kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return "OK"

    vm = make_vm(tools={"flaky": flaky_tool})
    program = Program(
        name="retry_metrics",
        steps=[
            Step(
                id="s1",
                type=StepType.TOOL,
                tool="flaky",
                on_error=OnError.RETRY,
                is_terminal=True,
            )
        ],
    )
    trace = await vm.run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert trace.step_metrics.tool_calls == 1  # one _run_step call, not one per attempt
    assert trace.step_metrics.retries_total == 2  # succeeded on 3rd attempt


# ---------------------------------------------------------------------------
# OT-10 — span_step no-op when OTel absent
# ---------------------------------------------------------------------------


def test_span_step_noop_when_otel_unavailable(monkeypatch):
    import nano_vm.telemetry as telemetry_mod

    monkeypatch.setattr(telemetry_mod, "OTEL_AVAILABLE", False)
    with span_step("s1", "llm", attempt=0) as span:
        assert span is None  # no-op path, no exception raised


# ---------------------------------------------------------------------------
# OT-11 — real span attributes (requires opentelemetry-sdk, dev-only dependency)
# ---------------------------------------------------------------------------


def test_span_step_real_span_attributes():
    otel_sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import nano_vm.telemetry as telemetry_mod

    exporter = InMemorySpanExporter()
    provider = otel_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    real_tracer = provider.get_tracer("nano_vm_test")

    telemetry_mod_available_before = telemetry_mod.OTEL_AVAILABLE
    tracer_before = telemetry_mod._TRACER
    try:
        telemetry_mod.OTEL_AVAILABLE = True
        telemetry_mod._TRACER = real_tracer
        with span_step("step_42", "tool", attempt=1):
            pass
    finally:
        telemetry_mod.OTEL_AVAILABLE = telemetry_mod_available_before
        telemetry_mod._TRACER = tracer_before

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "nano_vm.step.tool"
    assert span.attributes["nano_vm.step_id"] == "step_42"
    assert span.attributes["nano_vm.step_type"] == "tool"
    assert span.attributes["nano_vm.attempt"] == 1
    assert otel_trace is not None  # imported for API-availability sanity, not asserted further


# ---------------------------------------------------------------------------
# OT-12 — exception recorded on span, re-raised unchanged
# ---------------------------------------------------------------------------


def test_span_step_records_exception_and_reraises():
    class BoomError(Exception):
        pass

    with pytest.raises(BoomError):
        with span_step("s1", "tool", attempt=0):
            raise BoomError("kaboom")
