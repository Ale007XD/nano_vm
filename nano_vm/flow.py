"""nano_vm.flow — the single definition of "which step runs next".

``ExecutionVM._execute_loop`` and ``ProgramValidator`` both derive successors
from this module, so the static graph and the real control flow cannot drift.

A step's successor depends on **how the step was entered**:

``Entry.PLAIN``
    Reached at the top of the loop: entry step, sequential advance, a
    ``next_step`` jump, or the landing point of a CONDITION -> CONDITION
    recursion. Rules: ``is_terminal`` halts; a CONDITION branches; otherwise
    ``next_step`` if set, else the next step in ``steps[]`` (none after the last).

``Entry.BRANCH``
    Executed inline as the ``then``/``otherwise`` target of a CONDITION that was
    itself entered ``PLAIN``. There is no implicit sequential edge: the branch
    continues only through an explicit ``next_step`` (a CONDITION target
    recurses into its own branches); otherwise the run halts (SUCCESS).

The targets of a CONDITION entered ``PLAIN`` are entered ``BRANCH``; the targets
of a CONDITION that is itself an inline branch target are entered ``PLAIN``
(the recursion restarts the loop at them).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import NamedTuple

from nano_vm.models import Step, StepType


class Entry(str, Enum):
    PLAIN = "plain"
    BRANCH = "branch"


class Advance(NamedTuple):
    """Where a non-CONDITION step hands control.

    ``target`` is ``None`` when the run halts. ``sequential`` marks the implicit
    ``steps[index + 1]`` advance (no declared jump), which the VM follows by
    array position rather than by id.
    """

    target: str | None
    sequential: bool = False


_HALT = Advance(None)


def next_after(step: Step, index: int, steps: Sequence[Step], entry: Entry) -> Advance:
    """Advance of a non-CONDITION *step*.

    A declared id is returned as written; the caller decides what a dangling
    reference means (VM: FAILED, validator: MISSING_TARGET).
    """
    if entry is Entry.PLAIN:
        if step.is_terminal:
            return _HALT
        if step.next_step:
            return Advance(step.next_step)
        if index + 1 < len(steps):
            return Advance(steps[index + 1].id, sequential=True)
        return _HALT
    return Advance(step.next_step) if step.next_step else _HALT


def branch_targets(step: Step) -> list[str]:
    """Declared ``then`` / ``otherwise`` ids of a CONDITION step."""
    return [t for t in (step.then, step.otherwise) if t]


def successors(
    step: Step, index: int, steps: Sequence[Step], entry: Entry
) -> list[tuple[str, Entry]]:
    """All statically possible ``(next_step_id, entry_mode)`` pairs after *step*."""
    if step.type == StepType.CONDITION and not (step.is_terminal and entry is Entry.PLAIN):
        mode = Entry.BRANCH if entry is Entry.PLAIN else Entry.PLAIN
        return [(t, mode) for t in branch_targets(step)]
    nxt = next_after(step, index, steps, entry).target
    return [(nxt, Entry.PLAIN)] if nxt is not None else []
