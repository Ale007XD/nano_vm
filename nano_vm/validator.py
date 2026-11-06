"""nano_vm.validator — static analysis for Program graphs.

Pure in-memory analysis: no I/O, no LLM calls, no kernel changes.
Input: Program (nano_vm.models). Output: ValidationReport.

Checks
------
missing_targets   : O(N) — referenced step ids that don't exist in steps[]
unreachable_steps : BFS from steps[0] — steps never reachable from entry
cycle_detection   : DFS WHITE/GRAY/BLACK — back-edge = cycle

Invariants
----------
- Adjacency is derived from ``nano_vm.flow.successors`` — the same function the
  VM uses — so validator and runtime cannot disagree about control flow
- Edges depend on how a step is entered: a plain step falls through to the next
  array element; an inline CONDITION branch target continues only via an
  explicit next_step (otherwise the run halts)
- Terminal steps (is_terminal=True) have no outgoing edges when entered plain
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from nano_vm.flow import Entry, successors
from nano_vm.models import Program, Step

# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------


class IssueKind(str, Enum):
    MISSING_TARGET = "missing_target"
    UNREACHABLE_STEP = "unreachable_step"
    CYCLE_DETECTED = "cycle_detected"
    NO_FAILURE_TERMINAL = "no_failure_terminal"


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    kind: IssueKind
    step_id: str
    detail: str
    severity: IssueSeverity = IssueSeverity.ERROR


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReport:
    """Immutable result of static program analysis."""

    program_name: str
    issues: list[ValidationIssue] = field(default_factory=list)

    def is_valid(self) -> bool:
        return all(i.severity != IssueSeverity.ERROR for i in self.issues)

    def by_kind(self, kind: IssueKind) -> list[ValidationIssue]:
        return [i for i in self.issues if i.kind == kind]

    def summary(self) -> str:
        if self.is_valid():
            return f"ValidationReport({self.program_name}): VALID"
        lines = [f"ValidationReport({self.program_name}): {len(self.issues)} issue(s)"]
        for issue in self.issues:
            lines.append(f"  [{issue.kind.value}] {issue.step_id}: {issue.detail}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ProgramValidator
# ---------------------------------------------------------------------------


class ProgramValidator:
    """Static validator for Program step graphs.

    Usage::

        report = ProgramValidator(program).validate()
        if not report.is_valid():
            raise ValueError(report.summary())
    """

    def __init__(self, program: Program) -> None:
        self._program = program
        self._id_set: frozenset[str] = frozenset(s.id for s in program.steps)
        self._adjacency: dict[str, list[str]] = self._build_adjacency()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> ValidationReport:
        issues: list[ValidationIssue] = []
        issues.extend(self._check_missing_targets())
        issues.extend(self._check_unreachable_steps())
        issues.extend(self._check_cycles())
        issues.extend(self._check_failure_terminal())
        return ValidationReport(program_name=self._program.name, issues=issues)

    # ------------------------------------------------------------------
    # Adjacency construction
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> dict[str, list[str]]:
        """Build outgoing edge list per step_id from ``nano_vm.flow.successors``.

        The VM and this validator share one successor function, so the graph
        matches real control flow. A step's successors depend on how it is
        entered (plain vs. inline branch target), so the walk runs over
        ``(step_id, entry)`` states from ``steps[0]``; a step's edges are the
        union over the states in which it is reachable. Steps never reached
        fall back to their plain-entry edges (they are reported as unreachable).
        """
        steps = self._program.steps
        index: dict[str, int] = {}
        for i, s in enumerate(steps):
            index.setdefault(s.id, i)
        adj: dict[str, list[str]] = {s.id: [] for s in steps}
        if not steps:
            return adj

        seen: set[tuple[str, Entry]] = set()
        stack: list[tuple[str, Entry]] = [(steps[0].id, Entry.PLAIN)]
        while stack:
            sid, entry = stack.pop()
            if (sid, entry) in seen or sid not in index:
                continue
            seen.add((sid, entry))
            i = index[sid]
            for nxt, mode in successors(steps[i], i, steps, entry):
                if nxt not in adj[sid]:
                    adj[sid].append(nxt)
                stack.append((nxt, mode))

        reached = {sid for sid, _ in seen}
        for i, step in enumerate(steps):
            if step.id in reached:
                continue
            for nxt, _mode in successors(step, i, steps, Entry.PLAIN):
                if nxt not in adj[step.id]:
                    adj[step.id].append(nxt)
        return adj

    # ------------------------------------------------------------------
    # Check: missing targets O(N)
    # ------------------------------------------------------------------

    def _check_missing_targets(self) -> list[ValidationIssue]:
        """Collect all referenced step ids that are not in steps[]."""
        issues: list[ValidationIssue] = []
        for step in self._program.steps:
            for target_field, target_id in self._step_refs(step):
                if target_id not in self._id_set:
                    issues.append(
                        ValidationIssue(
                            kind=IssueKind.MISSING_TARGET,
                            step_id=step.id,
                            detail=f"{target_field}='{target_id}' not found in steps",
                        )
                    )
        return issues

    @staticmethod
    def _step_refs(step: Step) -> list[tuple[str, str]]:
        """Return (field_name, target_id) pairs for all id references in step."""
        refs: list[tuple[str, str]] = []
        if step.next_step:
            refs.append(("next_step", step.next_step))
        if step.then:
            refs.append(("then", step.then))
        if step.otherwise:
            refs.append(("otherwise", step.otherwise))
        return refs

    # ------------------------------------------------------------------
    # Check: unreachable steps — BFS from entry (steps[0])
    # ------------------------------------------------------------------

    def _check_unreachable_steps(self) -> list[ValidationIssue]:
        if not self._program.steps:
            return []

        entry = self._program.steps[0].id
        visited: set[str] = set()
        queue: list[str] = [entry]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self._adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        issues: list[ValidationIssue] = []
        for step in self._program.steps:
            if step.id not in visited:
                issues.append(
                    ValidationIssue(
                        kind=IssueKind.UNREACHABLE_STEP,
                        step_id=step.id,
                        detail=f"step '{step.id}' is not reachable from entry '{entry}'",
                    )
                )
        return issues

    # ------------------------------------------------------------------
    # Check: PV-13 — no reachable failure terminal (warning)
    # ------------------------------------------------------------------

    # Canonical failure-signalling output prefixes / exact values.
    # A terminal step whose allowed_outputs contains at least one of these
    # (or whose id contains a failure keyword) counts as a failure terminal.
    _FAILURE_KEYWORDS: frozenset[str] = frozenset(
        {"FAIL", "FAILED", "ERROR", "REJECTED", "INSUFFICIENT", "BLOCKED", "ESCALATED", "HELP"}
    )

    def _check_failure_terminal(self) -> list[ValidationIssue]:
        """PV-13: warn when no reachable terminal step signals failure.

        A program with only SUCCESS-type terminals cannot exit cleanly on
        unresolvable inputs — the FSM will retry-loop or hallucinate.
        This is a WARNING (not an error): valid for simple linear programs,
        but structurally risky for LLM steps with open-ended output space.
        """
        if not self._program.steps:
            return []

        # Reachable set (reuse BFS logic inline — adjacency already built)
        entry = self._program.steps[0].id
        visited: set[str] = set()
        queue: list[str] = [entry]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self._adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        def _is_failure_terminal(step: Step) -> bool:
            if not step.is_terminal:
                return False
            # Check step id
            sid_upper = step.id.upper()
            if any(kw in sid_upper for kw in self._FAILURE_KEYWORDS):
                return True
            # Check allowed_outputs values
            if step.allowed_outputs:
                for val in step.allowed_outputs:
                    if any(kw in val.upper() for kw in self._FAILURE_KEYWORDS):
                        return True
            return False

        has_failure_terminal = any(
            _is_failure_terminal(s) for s in self._program.steps if s.id in visited
        )

        if has_failure_terminal:
            return []

        return [
            ValidationIssue(
                kind=IssueKind.NO_FAILURE_TERMINAL,
                step_id=self._program.steps[0].id,
                detail=(
                    "no reachable terminal step signals failure "
                    "(id or allowed_outputs containing FAIL/ERROR/REJECTED/BLOCKED/…); "
                    "program may loop on unresolvable inputs"
                ),
                severity=IssueSeverity.WARNING,
            )
        ]

    def _check_cycles(self) -> list[ValidationIssue]:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {s.id: WHITE for s in self._program.steps}
        issues: list[ValidationIssue] = []

        def dfs(node: str, path: list[str]) -> None:
            color[node] = GRAY
            path.append(node)
            for neighbor in self._adjacency.get(node, []):
                if neighbor not in color:
                    continue  # missing target — reported separately
                if color[neighbor] == GRAY:
                    cycle_start = path.index(neighbor)
                    cycle = " → ".join(path[cycle_start:]) + f" → {neighbor}"
                    issues.append(
                        ValidationIssue(
                            kind=IssueKind.CYCLE_DETECTED,
                            step_id=node,
                            detail=f"back-edge to '{neighbor}' creates cycle: {cycle}",
                        )
                    )
                elif color[neighbor] == WHITE:
                    dfs(neighbor, path)
            path.pop()
            color[node] = BLACK

        for step in self._program.steps:
            if color[step.id] == WHITE:
                dfs(step.id, [])

        return issues
