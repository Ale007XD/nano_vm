"""
nano_vm.telemetry
==================
sprint_6_core_otel: one OTel span per FSM step execution attempt.

Soft dependency, same pattern as LiteLLMAdapter (nano_vm/adapters/litellm_adapter.py):
opentelemetry-api is NOT a hard dependency of nano_vm. If it isn't installed,
span_step() degrades to a no-op context manager — StepMetrics counting
(nano_vm/models.py::StepMetrics) is unaffected either way, since metric
accounting has no OTel dependency at all.

Span attributes: nano_vm.step_id, nano_vm.step_type, nano_vm.attempt.
This module only observes. It never changes control flow: on exception it
records the exception on the span and re-raises unchanged — retry/on_error
semantics remain owned exclusively by vm.py::_run_step.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import trace as _otel_trace

    _TRACER: Any | None = _otel_trace.get_tracer("nano_vm")
    OTEL_AVAILABLE = True
except ImportError:  # opentelemetry-api not installed — degrade to no-op
    _TRACER = None
    OTEL_AVAILABLE = False


@contextmanager
def span_step(step_id: str, step_type: str, attempt: int = 0) -> Iterator[Any]:
    """Open a span named 'nano_vm.step.<step_type>' for one _run_step attempt.

    No-op (yields None, no span created) when opentelemetry-api is absent.
    """
    if not OTEL_AVAILABLE or _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span(f"nano_vm.step.{step_type}") as span:
        span.set_attribute("nano_vm.step_id", step_id)
        span.set_attribute("nano_vm.step_type", step_type)
        span.set_attribute("nano_vm.attempt", attempt)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise
