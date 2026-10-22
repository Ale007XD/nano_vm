"""
tests/test_public_api_contract.py
=================================
Contract tests on nano_vm.__all__ (doc-drift, distinct from the CI
bare-install job): the bare-install job asserts the exports *resolve* in a
no-extras environment; these assert the exported surface still *matches what
the docs claim*.

  PA-01  every name in __all__ resolves
  PA-02  no duplicates in __all__
  PA-03  __all__ equals the pinned public surface (change it on purpose)
  PA-04  every `from nano_vm import X` in README is exported in __all__
  PA-05  public classes/functions of validator/analyzer modules are exported
  PA-06  __version__ is a non-empty str
  PA-07  `from nano_vm import *` exposes exactly __all__
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import nano_vm

ROOT = Path(__file__).resolve().parent.parent

# Adding or removing a public name is a contract change: edit this set in the
# same commit and add a CHANGELOG entry.
EXPECTED_ALL = frozenset(
    {
        # VM
        "ExecutionVM",
        "VMError",
        "ResumeError",
        "WebhookEvent",
        "CursorRepository",
        "InMemoryCursorRepository",
        # Planner
        "Planner",
        "PlannerError",
        # Program / Step
        "Program",
        "Step",
        "StepType",
        "StepStatus",
        "OnError",
        # State
        "StateContext",
        # Results
        "StepResult",
        "LLMUsage",
        # Trace
        "Trace",
        "TraceStatus",
        "StepMetrics",
        # vault primitives
        "InterruptType",
        "VaultStepResult",
        "VaultStepError",
        "VaultStepMetadata",
        # contracts / projection
        "CapabilityRef",
        "PolicySnapshot",
        "GovernanceEnvelope",
        "ProjectionTarget",
        "AbstractProjectionLayer",
        "DeterministicSanitizer",
        # validator
        "ProgramValidator",
        "ValidationReport",
        "ValidationIssue",
        "IssueKind",
        "IssueSeverity",
        # analyzer
        "TraceAnalyzer",
        "TraceHealthReport",
        "ExecutionReceipt",
        "RejectedTransition",
    }
)


def test_pa01_all_names_resolve() -> None:
    missing = [n for n in nano_vm.__all__ if not hasattr(nano_vm, n)]
    assert missing == []


def test_pa02_no_duplicates() -> None:
    assert len(nano_vm.__all__) == len(set(nano_vm.__all__))


def test_pa03_all_matches_pinned_surface() -> None:
    actual = frozenset(nano_vm.__all__)
    assert actual == EXPECTED_ALL, {
        "added": sorted(actual - EXPECTED_ALL),
        "removed": sorted(EXPECTED_ALL - actual),
    }


def test_pa04_readme_top_level_imports_are_exported() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"^```python[^\n]*\n(.*?)^```", readme, re.DOTALL | re.MULTILINE)
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT | ast.PyCF_ONLY_AST
    wanted: set[str] = set()
    for src in blocks:
        for node in ast.walk(compile(src, "<readme>", "exec", flags=flags)):  # type: ignore[arg-type]
            if isinstance(node, ast.ImportFrom) and node.module == "nano_vm":
                wanted.update(a.name for a in node.names)
    assert wanted, "README shows no `from nano_vm import ...` — fence regex broke?"
    assert wanted <= set(nano_vm.__all__), sorted(wanted - set(nano_vm.__all__))


@pytest.mark.parametrize(
    ("module", "skip"),
    [
        ("nano_vm.validator", set()),
        # analyze_batch: public helper, covered by test_analyzer.py (TA-22), not
        # exported from the package root and not documented in README. Left
        # unexported pending an owner decision; drop from skip if exported.
        ("nano_vm.analyzer", {"analyze_batch"}),
    ],
)
def test_pa05_public_symbols_of_analysis_modules_are_exported(module: str, skip: set[str]) -> None:
    import importlib
    import inspect

    mod = importlib.import_module(module)
    public = {
        name
        for name, obj in vars(mod).items()
        if not name.startswith("_")
        and (inspect.isclass(obj) or inspect.isfunction(obj))
        and getattr(obj, "__module__", None) == module
    } - skip
    not_exported = sorted(public - set(nano_vm.__all__))
    assert not_exported == [], (
        f"{module} defines public symbols missing from nano_vm.__all__: {not_exported}"
    )


def test_pa06_version_is_str() -> None:
    assert isinstance(nano_vm.__version__, str) and nano_vm.__version__


def test_pa07_star_import_matches_all() -> None:
    ns: dict[str, object] = {}
    exec("from nano_vm import *", ns)  # noqa: S102 - controlled literal
    ns.pop("__builtins__", None)
    assert set(ns) == set(nano_vm.__all__)
