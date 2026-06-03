"""TraceAnalyzer — post-processing observability over Trace objects.

Pure analysis layer: no kernel changes, no I/O, no FSM interaction.
Input: Trace (nano_vm.models). Output: TraceHealthReport.

Metrics
-------
rollback_density      : retried steps / total steps            → float [0,1]
tool_churn_rate       : duplicate step_id executions / total   → float [0,1]
path_variance         : fraction of snapshots diverging from   → float [0,1]
                         baseline (requires baseline Trace)
invariant_violation_rate    : failed steps / total steps              → float [0,1]
transition_sequence_variance: fraction of (prev→curr) step_id pairs    → float [0,1]
                               in trace not present in baseline
transition_entropy    : Shannon entropy of transition probability matrix → float [0,∞)
                               (bits; higher = more unpredictable)

Alert thresholds (non-interrupting — report only)
-------------------------------------------------
rollback_density             > 0.3
tool_churn_rate              > 0.4
path_variance                > 0.5
invariant_violation_rate     > 0.2
transition_sequence_variance > 0.4
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from nano_vm.models import StepStatus, Trace

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_THRESHOLD_ROLLBACK_DENSITY: float = 0.3
_THRESHOLD_TOOL_CHURN_RATE: float = 0.4
_THRESHOLD_PATH_VARIANCE: float = 0.5
_THRESHOLD_INVARIANT_VIOLATION_RATE: float = 0.2
_THRESHOLD_TRANSITION_SEQUENCE_VARIANCE: float = 0.4
_THRESHOLD_TRANSITION_ENTROPY: float = 1.5  # bits; alert when H > 1.5 bits


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceHealthReport:
    """Immutable analysis report for a single Trace execution."""

    trace_id: str
    program_name: str
    total_steps: int

    rollback_density: float
    tool_churn_rate: float
    path_variance: float | None  # None when no baseline provided
    invariant_violation_rate: float
    transition_sequence_variance: float | None  # None when no baseline provided
    transition_entropy: float  # Shannon entropy of step transitions in bits

    alerts: list[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        return len(self.alerts) == 0

    def summary(self) -> str:
        lines = [
            f"TraceHealthReport({self.trace_id})",
            f"  program        : {self.program_name}",
            f"  total_steps    : {self.total_steps}",
            f"  rollback_density          : {self.rollback_density:.3f}",
            f"  tool_churn_rate           : {self.tool_churn_rate:.3f}",
            "  path_variance             : "
            + (f"{self.path_variance:.3f}" if self.path_variance is not None else "n/a"),
            f"  invariant_violation_rate  : {self.invariant_violation_rate:.3f}",
            "  transition_seq_variance   : "
            + (
                f"{self.transition_sequence_variance:.3f}"
                if self.transition_sequence_variance is not None
                else "n/a"
            ),
            f"  transition_entropy        : {self.transition_entropy:.3f} bits",
        ]
        if self.alerts:
            lines.append("  alerts:")
            for a in self.alerts:
                lines.append(f"    ⚠  {a}")
        else:
            lines.append("  status: HEALTHY")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------


class TraceAnalyzer:
    """Compute health metrics for a Trace.

    Parameters
    ----------
    trace:
        The execution trace to analyse.
    baseline:
        Optional reference Trace whose ``state_snapshots`` serve as the
        baseline for ``path_variance`` computation.  When omitted,
        ``path_variance`` is ``None`` in the report.
    """

    def __init__(self, trace: Trace, baseline: Trace | None = None) -> None:
        self._trace = trace
        self._baseline = baseline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(self) -> TraceHealthReport:
        total = len(self._trace.steps)
        rd = self.rollback_density()
        tcr = self.tool_churn_rate()
        pv = self.path_variance()
        ivr = self.invariant_violation_rate()
        tsv = self.transition_sequence_variance()
        te = self.transition_entropy()

        alerts: list[str] = []
        if rd > _THRESHOLD_ROLLBACK_DENSITY:
            alerts.append(
                f"rollback_density {rd:.3f} > {_THRESHOLD_ROLLBACK_DENSITY} (execution instability)"
            )
        if tcr > _THRESHOLD_TOOL_CHURN_RATE:
            alerts.append(
                f"tool_churn_rate {tcr:.3f} > {_THRESHOLD_TOOL_CHURN_RATE} "
                "(tool oscillation detected)"
            )
        if pv is not None and pv > _THRESHOLD_PATH_VARIANCE:
            alerts.append(
                f"path_variance {pv:.3f} > {_THRESHOLD_PATH_VARIANCE} "
                "(state diverges from baseline)"
            )
        if ivr > _THRESHOLD_INVARIANT_VIOLATION_RATE:
            alerts.append(
                f"invariant_violation_rate {ivr:.3f} > {_THRESHOLD_INVARIANT_VIOLATION_RATE} "
                "(high error rate)"
            )
        if tsv is not None and tsv > _THRESHOLD_TRANSITION_SEQUENCE_VARIANCE:
            thr = _THRESHOLD_TRANSITION_SEQUENCE_VARIANCE
            alerts.append(
                f"transition_sequence_variance {tsv:.3f} > {thr} "
                "(structural path divergence from baseline)"
            )
        if te > _THRESHOLD_TRANSITION_ENTROPY:
            alerts.append(
                f"transition_entropy {te:.3f} > {_THRESHOLD_TRANSITION_ENTROPY} bits "
                "(high path non-determinism)"
            )

        return TraceHealthReport(
            trace_id=self._trace.trace_id,
            program_name=self._trace.program_name,
            total_steps=total,
            rollback_density=rd,
            tool_churn_rate=tcr,
            path_variance=pv,
            invariant_violation_rate=ivr,
            transition_sequence_variance=tsv,
            transition_entropy=te,
            alerts=alerts,
        )

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def rollback_density(self) -> float:
        """Fraction of steps that were retried (retries > 0).

        A step with ``retries=N`` counts as one retried step regardless of N.
        Range: [0.0, 1.0].
        """
        steps = self._trace.steps
        if not steps:
            return 0.0
        retried = sum(1 for s in steps if s.retries > 0)
        return retried / len(steps)

    def tool_churn_rate(self) -> float:
        """Fraction of total steps that are duplicate step_id executions.

        Duplicate = same ``step_id`` appears more than once in the trace
        (caused by retry storms or looping back to the same step).
        Range: [0.0, 1.0].
        """
        steps = self._trace.steps
        if not steps:
            return 0.0
        counts: dict[str, int] = {}
        for s in steps:
            counts[s.step_id] = counts.get(s.step_id, 0) + 1
        duplicates = sum(c - 1 for c in counts.values() if c > 1)
        return duplicates / len(steps)

    def path_variance(self) -> float | None:
        """Fraction of snapshot positions that differ from the baseline trace.

        Compares ``state_snapshots`` element-wise by (step_index, sha256_hex).
        If the baseline has a different length, the shorter one determines the
        comparison window; extra steps in the longer trace are counted as
        divergent.

        Returns ``None`` when no baseline was provided.
        Range: [0.0, 1.0].
        """
        if self._baseline is None:
            return None

        snaps = self._trace.state_snapshots
        base_snaps = self._baseline.state_snapshots

        total = max(len(snaps), len(base_snaps))
        if total == 0:
            return 0.0

        base_map: dict[int, str] = {idx: fp for idx, fp in base_snaps}
        divergent = 0
        for idx, fp in snaps:
            if base_map.get(idx) != fp:
                divergent += 1
        # Positions in baseline not present in trace also diverge
        trace_indices = {idx for idx, _ in snaps}
        for idx, _ in base_snaps:
            if idx not in trace_indices:
                divergent += 1

        return divergent / total

    def invariant_violation_rate(self) -> float:
        """Fraction of steps that ended in FAILED status.

        Corresponds to GovernanceEnvelope invariant: envelopes written only
        on error=None.  Steps with error != None are violations.
        Range: [0.0, 1.0].
        """
        steps = self._trace.steps
        if not steps:
            return 0.0
        failed = sum(1 for s in steps if s.status == StepStatus.FAILED)
        return failed / len(steps)

    def transition_sequence_variance(self) -> float | None:
        """Fraction of (prev_step_id → curr_step_id) pairs in trace not in baseline.

        Captures structural path divergence that ``path_variance`` misses when
        the final state hash matches but the route through the FSM differed.

        Example::

            baseline: A→B→C   transitions: {(A,B), (B,C)}
            trace:    A→D→C   transitions: {(A,D), (D,C)}
            variance = 2/2 = 1.0  (both pairs diverge)

        Returns ``None`` when no baseline was provided.
        Range: [0.0, 1.0].
        """
        if self._baseline is None:
            return None

        def _pairs(trace: Trace) -> set[tuple[str, str]]:
            ids = [s.step_id for s in trace.steps]
            return set(zip(ids, ids[1:]))

        trace_pairs = _pairs(self._trace)
        base_pairs = _pairs(self._baseline)

        all_pairs = trace_pairs | base_pairs
        if not all_pairs:
            return 0.0

        divergent = len(trace_pairs.symmetric_difference(base_pairs))
        return divergent / len(all_pairs)

    def transition_entropy(self) -> float:
        """Shannon entropy of (prev_step_id → curr_step_id) transition distribution.

        H = -Σ p_i * log2(p_i)  where p_i = count(pair_i) / total_pairs

        Single or zero steps → 0.0 (deterministic, no transitions).
        All transitions unique → log2(N) (maximum entropy).
        Range: [0.0, log2(N)] bits.

        Note: absolute (non-normalized) entropy. A deterministic pipeline
        typically scores 0.0–0.5; moderate branching 1.0–1.8; values above
        1.5 trigger an alert. Normalized form (H / log2(N) → [0,1]) is
        deferred until cross-trace comparison is needed.
        """
        import math

        ids = [s.step_id for s in self._trace.steps]
        if len(ids) < 2:
            return 0.0

        counts: dict[tuple[str, str], int] = {}
        for a, b in zip(ids, ids[1:]):
            key = (a, b)
            counts[key] = counts.get(key, 0) + 1

        total = sum(counts.values())
        return -sum((c / total) * math.log2(c / total) for c in counts.values())


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def analyze_batch(
    traces: Sequence[Trace],
    baseline: Trace | None = None,
) -> list[TraceHealthReport]:
    """Analyse a collection of traces, optionally against a shared baseline."""
    return [TraceAnalyzer(t, baseline=baseline).report() for t in traces]
