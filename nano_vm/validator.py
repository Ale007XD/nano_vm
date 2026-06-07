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
- Adjacency built once from Step fields: next_step, then, otherwise
- Implicit sequential edge (step[i] → step[i+1]) only when step[i] has
  no explicit next_step and is not is_terminal and is not condition type
- Condition steps never have implicit sequential edge (they branch explicitly)
- Terminal steps (is_terminal=True) have no outgoing edges regardless of fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from nano_vm.models import Program, Step, StepType

# ---------------------------------------------------------------------------
# ValidationIssue
# ---------------------------------------------------------------------------


class IssueKind(str, Enum):
    MISSING_TARGET = "missing_target"
    UNREACHABLE_STEP = "unreachable_step"
    CYCLE_DETECTED = "cycle_detected"


@dataclass(frozen=True)
class ValidationIssue:
    kind: IssueKind
    step_id: str
    detail: str


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationReport:
    """Immutable result of static program analysis."""

    program_name: str
    issues: list[ValidationIssue] = field(default_factory=list)

    def is_valid(self) -> bool:
        return len(self.issues) == 0

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
        return ValidationReport(program_name=self._program.name, issues=issues)

    # ------------------------------------------------------------------
    # Adjacency construction
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> dict[str, list[str]]:
        """Build outgoing edge list per step_id.

        Rules:
        - terminal step → no edges
        - condition step → [then, otherwise] (whichever are set); no implicit edge
        - other step with next_step → [next_step]; no implicit sequential edge
        - other step without next_step → [steps[i+1].id] if i+1 exists
        """
        steps = self._program.steps
        index: dict[str, int] = {s.id: i for i, s in enumerate(steps)}
        adj: dict[str, list[str]] = {s.id: [] for s in steps}

        for step in steps:
            if step.is_terminal:
                continue  # no outgoing edges

            if step.type == StepType.CONDITION:
                if step.then:
                    adj[step.id].append(step.then)
                if step.otherwise:
                    adj[step.id].append(step.otherwise)
                continue  # no implicit sequential edge for conditions

            if step.next_step:
                adj[step.id].append(step.next_step)
            else:
                i = index[step.id]
                if i + 1 < len(steps):
                    adj[step.id].append(steps[i + 1].id)

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
    # Check: cycle detection — DFS WHITE/GRAY/BLACK
    # ------------------------------------------------------------------

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
