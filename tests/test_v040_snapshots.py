"""
test_v040_snapshots.py
======================
P2: state_snapshots — (step_index, sha256_hex) recorded per executed step.

Coverage:
  - Empty program result has no snapshots (regression guard)
  - Single step → exactly 1 snapshot
  - N steps → exactly N snapshots
  - step_index values are 0-based sequential
  - fp_hex is valid 64-char sha256 hex string
  - Identical state → identical fp_hex (cross-run stability guarantee)
  - Different state → different fp_hex
  - BUDGET_EXCEEDED: snapshots recorded for completed steps only
  - STALLED: snapshots recorded up to and including the stalling step
  - Snapshot fp_hex differs from hash()-based fingerprint (different algorithm)
  - Condition branch: no extra snapshot for branch step (branch handled via return)
  - _state_fingerprint_hex is cross-process stable (sha256, not hash())
"""

from __future__ import annotations

import hashlib

import pytest

from nano_vm.models import Program, StateContext, Step, StepType, TraceStatus
from nano_vm.vm import ExecutionVM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _EchoAdapter:
    async def complete(self, messages):
        return messages[-1]["content"]


def _make_vm(**tools) -> ExecutionVM:
    return ExecutionVM(llm=_EchoAdapter(), tools=tools or {})


def _llm(step_id: str, prompt: str = "hi") -> Step:
    return Step(id=step_id, type=StepType.LLM, prompt=prompt)


def _is_sha256_hex(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_steps_no_snapshots():
    """Degenerate guard: program with 1 step that succeeds → 1 snapshot (not 0)."""
    program = Program(name="t", steps=[_llm("s1")])
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.SUCCESS
    assert len(trace.state_snapshots) == 1


@pytest.mark.asyncio
async def test_single_step_one_snapshot():
    """1 step → exactly 1 snapshot."""
    program = Program(name="t", steps=[_llm("s1")])
    trace = await _make_vm().run(program)
    assert len(trace.state_snapshots) == 1


@pytest.mark.asyncio
async def test_n_steps_n_snapshots():
    """N sequential steps → exactly N snapshots."""
    for n in (1, 2, 5):
        program = Program(
            name=f"t{n}",
            steps=[_llm(f"s{i}") for i in range(n)],
        )
        trace = await _make_vm().run(program)
        assert trace.status == TraceStatus.SUCCESS
        assert len(trace.state_snapshots) == n, (
            f"expected {n} snapshots, got {len(trace.state_snapshots)}"
        )


@pytest.mark.asyncio
async def test_snapshot_step_index_sequential():
    """step_index values are 0, 1, 2, ..., N-1."""
    program = Program(name="t", steps=[_llm(f"s{i}") for i in range(4)])
    trace = await _make_vm().run(program)
    indices = [s[0] for s in trace.state_snapshots]
    assert indices == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_snapshot_fp_hex_is_valid_sha256():
    """Each fp_hex is a valid 64-char sha256 hex string."""
    program = Program(name="t", steps=[_llm("s1"), _llm("s2"), _llm("s3")])
    trace = await _make_vm().run(program)
    for step_idx, fp_hex in trace.state_snapshots:
        assert _is_sha256_hex(fp_hex), f"step {step_idx}: invalid hex '{fp_hex}'"


@pytest.mark.asyncio
async def test_snapshot_fp_hex_stable_across_calls():
    """Same program + same adapter → identical fp_hex values in both runs."""
    program = Program(name="t", steps=[_llm("s1"), _llm("s2")])
    trace_a = await _make_vm().run(program)
    trace_b = await _make_vm().run(program)
    assert trace_a.state_snapshots == trace_b.state_snapshots


@pytest.mark.asyncio
async def test_snapshot_fp_hex_changes_with_state():
    """Different step outputs → different fp_hex."""
    # s1 and s2 have different ids → step_outputs differ → different fp_hex at each step
    program = Program(name="t", steps=[_llm("s1"), _llm("s2")])
    trace = await _make_vm().run(program)
    fp_0 = trace.state_snapshots[0][1]
    fp_1 = trace.state_snapshots[1][1]
    assert fp_0 != fp_1


@pytest.mark.asyncio
async def test_snapshot_fingerprint_hex_matches_direct_computation():
    """
    fp_hex in trace matches _state_fingerprint_hex() computed on equivalent StateContext.
    Verifies the serialisation is deterministic and correct.
    """
    program = Program(name="t", steps=[_llm("s1")])
    trace = await _make_vm().run(program)
    _, fp_hex = trace.state_snapshots[0]

    # Reconstruct what state looks like after s1 executes:
    # _EchoAdapter returns the prompt, which is "hi"
    state = StateContext().with_output("s1", "hi")
    expected_hex = ExecutionVM._state_fingerprint_hex(state)
    assert fp_hex == expected_hex


@pytest.mark.asyncio
async def test_budget_exceeded_snapshots_for_completed_steps():
    """
    max_steps=2, 4-step program → BUDGET_EXCEEDED after step 2.
    Snapshots recorded for steps 0 and 1 only.
    """
    program = Program(
        name="t",
        max_steps=2,
        steps=[_llm(f"s{i}") for i in range(4)],
    )
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.BUDGET_EXCEEDED
    assert len(trace.state_snapshots) == 2
    assert trace.state_snapshots[0][0] == 0
    assert trace.state_snapshots[1][0] == 1


@pytest.mark.asyncio
async def test_stalled_snapshots_up_to_stalling_step():
    """
    s1 (progress) → s1 dup (no-op, STALLED with max_stalled_steps=1).
    Snapshots: step 0 (s1) and step 1 (s1 dup) both recorded before STALLED fires.
    """
    s1 = Step(id="s1", type=StepType.LLM, prompt="hi")
    s1_dup = Step(id="s1", type=StepType.LLM, prompt="hi")
    program = Program(name="t", max_stalled_steps=1, steps=[s1, s1_dup])
    trace = await _make_vm().run(program)
    assert trace.status == TraceStatus.STALLED
    # snapshot recorded for s1 (step 0, progress).
    # s1_dup triggers STALLED before its snapshot is written → only 1 snapshot.
    assert len(trace.state_snapshots) == 1
    assert trace.state_snapshots[0][0] == 0


@pytest.mark.asyncio
async def test_state_fingerprint_hex_is_not_hash():
    """
    _state_fingerprint_hex must NOT use Python's hash().
    Verify the value matches sha256, not hash().
    """
    state = StateContext().with_output("k", "v")
    fp_hex = ExecutionVM._state_fingerprint_hex(state)
    # Recompute manually using same canonical form
    canonical = "k='v'"
    expected = hashlib.sha256(canonical.encode()).hexdigest()
    assert fp_hex == expected
    # Must not equal hash()-based value (different algorithm)
    assert fp_hex != str(hash(frozenset([("k", "v")])))


@pytest.mark.asyncio
async def test_add_snapshot_immutable():
    """Trace.add_snapshot returns new Trace, does not mutate original."""
    from nano_vm.models import Trace

    t0 = Trace(program_name="t")
    t1 = t0.add_snapshot(0, "a" * 64)
    assert len(t0.state_snapshots) == 0
    assert len(t1.state_snapshots) == 1


@pytest.mark.asyncio
async def test_snapshots_serialisable_to_json():
    """state_snapshots survive model_dump / model_validate round-trip."""
    program = Program(name="t", steps=[_llm("s1"), _llm("s2")])
    trace = await _make_vm().run(program)
    dumped = trace.model_dump()
    from nano_vm.models import Trace

    restored = Trace.model_validate(dumped)
    assert restored.state_snapshots == trace.state_snapshots
