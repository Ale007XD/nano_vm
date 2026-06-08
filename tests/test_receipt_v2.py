cat > tests / test_receipt_v2.py << "EOF"
"""ER-11..15 — RejectedTransition + новые поля ExecutionReceipt."""

from __future__ import annotations

from nano_vm.analyzer import RejectedTransition, TraceAnalyzer
from nano_vm.models import StepResult, StepStatus, Trace, TraceStatus


def _step(
    step_id: str, status: StepStatus = StepStatus.SUCCESS, error: str | None = None
) -> StepResult:
    return StepResult(step_id=step_id, status=status, error=error)


def _trace(
    steps: list[StepResult] | None = None, status: TraceStatus = TraceStatus.SUCCESS
) -> Trace:
    t = Trace(program_name="p", status=status)
    for s in steps or []:
        t = t.add_step(s)
    return t


def test_er11_blocked_actions_default() -> None:
    """ER-11: blocked_actions == 0 (deferred field)"""
    r = TraceAnalyzer(_trace([_step("a")])).receipt()
    assert r.blocked_actions == 0


def test_er12_escalations_default() -> None:
    """ER-12: escalations == 0 (deferred field)"""
    r = TraceAnalyzer(_trace([_step("a")])).receipt()
    assert r.escalations == 0


def test_er13_rejected_transitions_empty_on_clean() -> None:
    """ER-13: no FAILED steps → rejected_transitions == ()"""
    r = TraceAnalyzer(_trace([_step("a"), _step("b")])).receipt()
    assert r.rejected_transitions == ()


def test_er14_rejected_transitions_from_failed() -> None:
    """ER-14: FAILED step → RejectedTransition with correct fields"""
    trace = _trace(
        [_step("validate"), _step("enrich", StepStatus.FAILED, error="timeout"), _step("review")],
        status=TraceStatus.FAILED,
    )
    r = TraceAnalyzer(trace).receipt()
    assert len(r.rejected_transitions) == 1
    rt = r.rejected_transitions[0]
    assert isinstance(rt, RejectedTransition)
    assert rt.step_id == "enrich"
    assert rt.reason == "timeout"
    assert rt.rule_id is None
    assert isinstance(rt.timestamp, str)


def test_er15_rejected_transitions_order() -> None:
    """ER-15: order matches trace.steps iteration order"""
    trace = _trace(
        [
            _step("a", StepStatus.FAILED, error="err_a"),
            _step("b"),
            _step("c", StepStatus.FAILED, error="err_c"),
        ],
        status=TraceStatus.FAILED,
    )
    r = TraceAnalyzer(trace).receipt()
    assert [rt.step_id for rt in r.rejected_transitions] == ["a", "c"]


def test_er16_rejected_transition_unknown_reason() -> None:
    """ER-16: FAILED step with error=None → reason == 'unknown'"""
    step = StepResult(step_id="x", status=StepStatus.FAILED, error=None)
    t = Trace(program_name="p", status=TraceStatus.FAILED)
    t = t.add_step(step)
    r = TraceAnalyzer(t).receipt()
    assert r.rejected_transitions[0].reason == "unknown"


def test_er17_rejected_transition_timestamp_nonempty() -> None:
    """ER-17: timestamp is non-empty string (started_at always set by default_factory)"""
    step = StepResult(step_id="y", status=StepStatus.FAILED, error="boom")
    t = Trace(program_name="p", status=TraceStatus.FAILED)
    t = t.add_step(step)
    r = TraceAnalyzer(t).receipt()
    assert r.rejected_transitions[0].timestamp != ""


EOF
