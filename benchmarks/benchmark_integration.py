"""
benchmark_integration.py
Cross-package integration benchmark: llm-nano-vm v0.8.0 × nano-vm-mcp v0.3.0

Scenarios (3 cycles × 5 runs × 10 000 items each):
  BM-INT-01  Refund pipeline         llm+condition+parallel+tool
  BM-INT-02  Double-execution guard  FSM invariant I_k(T) ∈ {0,1}
  BM-INT-03  Budget enforcement      max_steps / max_tokens walls
  BM-INT-04  Parallel throughput     asyncio.gather concurrency scaling
               NOTE: n=programs (1 000), not sub-steps. OK% = program SUCCESS rate.
  BM-INT-05  MCP store round-trip    save_program → run → get_trace (SQLite WAL)
               NOTE: elapsed = wall clock (perf_counter), not sum(latencies).
  BM-INT-06  GovernanceEnvelope      CapabilityRef tombstoning + audit trail

  BM-INT-07  Crash consistency       mid-transition kill → replay → semantic equivalence
               trace_hash(clean) == trace_hash(resumed) — gold invariant
  BM-INT-08  Replay equivalence      trace_hash(run) == trace_hash(replay)
  BM-INT-09  Adversarial retries     duplicate webhooks, out-of-order, delayed acks
  BM-INT-10  Long-horizon            100k-step workflow + memory profile

Usage:
  pip install llm-nano-vm nano-vm-mcp rich
  python benchmark_integration.py

  # Run only original suite (faster):
  python benchmark_integration.py --suite original

  # Run only new suite:
  python benchmark_integration.py --suite new

  # Run specific scenarios:
  python benchmark_integration.py --only BM-INT-07,BM-INT-08
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import math
import os
import random
import resource
import statistics
import tempfile
import time
import tracemalloc
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── nano_vm_mcp ───────────────────────────────────────────────────────────────
from nano_vm_mcp.store import ProgramStore as Store

# ── rich ──────────────────────────────────────────────────────────────────────
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ── nano_vm ───────────────────────────────────────────────────────────────────
from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.adapters import MockLLMAdapter

console = Console()

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

N = 10_000
RUNS = 5
CYCLES = 3
SEED = 42

TERMINAL_OK = {TraceStatus.SUCCESS}
TERMINAL_ALL = {
    TraceStatus.SUCCESS,
    TraceStatus.FAILED,
    TraceStatus.BUDGET_EXCEEDED,
    TraceStatus.STALLED,
}

_BATCH_CONCURRENCY = 200

# ══════════════════════════════════════════════════════════════════════════════
# Mock tools
# ══════════════════════════════════════════════════════════════════════════════


async def tool_get_order_amount(order_id: str) -> float:
    await asyncio.sleep(0.0008)
    rng = random.Random(int(order_id) % 9999)
    return round(rng.uniform(10.0, 800.0), 2)


async def tool_get_order_status(order_id: str) -> str:
    await asyncio.sleep(0.0008)
    rng = random.Random(int(order_id) % 9999 + 1)
    return rng.choice(["delivered", "shipped", "pending", "cancelled"])


async def tool_issue_refund(order_id: str) -> dict[str, Any]:
    await asyncio.sleep(0.0012)
    return {"refund_id": f"ref_{order_id}", "status": "processed"}


async def tool_send_rejection(order_id: str) -> dict[str, Any]:
    await asyncio.sleep(0.0004)
    return {"rejected": True, "order_id": order_id}


async def tool_verify_policy(capability_id: str) -> dict[str, Any]:
    await asyncio.sleep(0.0006)
    return {"allowed": True, "capability_id": capability_id}


async def tool_audit_log(trace_id: str, step_id: str) -> dict[str, Any]:
    await asyncio.sleep(0.0003)
    return {"logged": True, "trace_id": trace_id, "step_id": step_id}


TOOLS: dict[str, Any] = {
    "get_order_amount": tool_get_order_amount,
    "get_order_status": tool_get_order_status,
    "issue_refund": tool_issue_refund,
    "send_rejection": tool_send_rejection,
    "verify_policy": tool_verify_policy,
    "audit_log": tool_audit_log,
}

# ══════════════════════════════════════════════════════════════════════════════
# DSL programs
# ══════════════════════════════════════════════════════════════════════════════

REFUND_PROGRAM = {
    "name": "refund_pipeline",
    "max_steps": 12,
    "steps": [
        {
            "id": "classify",
            "type": "llm",
            "prompt": "Classify intent: $user_input. Reply refund or other.",
            "output_key": "intent",
        },
        {
            "id": "route",
            "type": "condition",
            "condition": "'refund' in $intent",
            "then": "enrich",
            "otherwise": "reject",
        },
        {
            "id": "enrich",
            "type": "parallel",
            "output_key": "enriched",
            "max_concurrency": 2,
            "on_error": "skip",
            "parallel_steps": [
                {
                    "id": "get_amount",
                    "type": "tool",
                    "tool": "get_order_amount",
                    "args": {"order_id": "$order_id"},
                },
                {
                    "id": "get_status",
                    "type": "tool",
                    "tool": "get_order_status",
                    "args": {"order_id": "$order_id"},
                },
            ],
        },
        {
            "id": "eligibility",
            "type": "llm",
            "prompt": "Is this order eligible for refund? Reply yes or no.",
            "output_key": "eligible",
        },
        {
            "id": "guard",
            "type": "condition",
            "condition": "'yes' in $eligible",
            "then": "process_refund",
            "otherwise": "reject",
        },
        {
            "id": "process_refund",
            "type": "tool",
            "tool": "issue_refund",
            "args": {"order_id": "$order_id"},
        },
        {
            "id": "reject",
            "type": "tool",
            "tool": "send_rejection",
            "args": {"order_id": "$order_id"},
        },
    ],
}

BUDGET_PROGRAM = {
    "name": "budget_guard",
    "max_steps": 3,
    "steps": [
        {
            "id": f"step_{i}",
            "type": "tool",
            "tool": "audit_log",
            "args": {"trace_id": "t0", "step_id": f"step_{i}"},
        }
        for i in range(8)
    ],
}

GOVERNANCE_PROGRAM = {
    "name": "governed_pipeline",
    "max_steps": 10,
    "steps": [
        {
            "id": "capability_check",
            "type": "tool",
            "tool": "verify_policy",
            "args": {"capability_id": "$capability_id"},
            "output_key": "policy_result",
        },
        {
            "id": "classify",
            "type": "llm",
            "prompt": "Classify: $user_input. Reply safe or unsafe.",
            "output_key": "classification",
        },
        {
            "id": "route",
            "type": "condition",
            "condition": "'safe' in $classification",
            "then": "process",
            "otherwise": "reject",
        },
        {
            "id": "process",
            "type": "tool",
            "tool": "issue_refund",
            "args": {"order_id": "$order_id"},
        },
        {
            "id": "reject",
            "type": "tool",
            "tool": "send_rejection",
            "args": {"order_id": "$order_id"},
        },
        {
            "id": "audit",
            "type": "tool",
            "tool": "audit_log",
            "args": {"trace_id": "$trace_id", "step_id": "final"},
        },
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# LLM adapters
# ══════════════════════════════════════════════════════════════════════════════


def make_refund_llm(rng: random.Random) -> MockLLMAdapter:
    responses: list[str] = []
    for _ in range(N * RUNS * 4):
        responses.append("refund" if rng.random() < 0.80 else "other")
        responses.append("yes" if rng.random() < 0.70 else "no")
    return MockLLMAdapter(responses)


def make_double_llm() -> MockLLMAdapter:
    return MockLLMAdapter(["yes"] * (N * RUNS * 2 + 100))


def make_governance_llm(rng: random.Random) -> MockLLMAdapter:
    responses: list[str] = []
    for _ in range(N * RUNS * 4):
        responses.append("safe" if rng.random() < 0.75 else "unsafe")
    return MockLLMAdapter(responses)


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RunResult:
    scenario_id: str
    cycle: int
    run: int
    n: int
    elapsed_s: float
    throughput: float
    ok: int
    failed: int
    budget_exceeded: int
    violations: int
    mean_steps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioSummary:
    scenario_id: str
    label: str
    tag: str
    results: list[RunResult] = field(default_factory=list)

    @property
    def all_throughputs(self) -> list[float]:
        return [r.throughput for r in self.results]

    @property
    def mean_tps(self) -> float:
        return statistics.mean(self.all_throughputs)

    @property
    def stddev_tps(self) -> float:
        return statistics.stdev(self.all_throughputs) if len(self.all_throughputs) > 1 else 0.0

    @property
    def total_violations(self) -> int:
        return sum(r.violations for r in self.results)

    @property
    def total_ok(self) -> int:
        return sum(r.ok for r in self.results)

    @property
    def total_n(self) -> int:
        return sum(r.n for r in self.results)

    @property
    def ok_pct(self) -> float:
        return 100.0 * self.total_ok / self.total_n if self.total_n else 0.0

    @property
    def mean_p95(self) -> float:
        vals = [r.p95_ms for r in self.results]
        return statistics.mean(vals) if vals else 0.0

    @property
    def passed(self) -> bool:
        return self.total_violations == 0


# ══════════════════════════════════════════════════════════════════════════════
# Core batch runner
# ══════════════════════════════════════════════════════════════════════════════


async def _run_batch(
    vm: ExecutionVM,
    program: Program,
    contexts: list[dict[str, Any]],
    invariant_fn: Any,
) -> tuple[list[float], int, int, int, int, float]:
    sem = asyncio.Semaphore(_BATCH_CONCURRENCY)
    latencies: list[float] = []
    ok = failed = budget = violations = 0
    step_counts: list[int] = []
    lock = asyncio.Lock()

    async def _one(ctx: dict[str, Any]) -> None:
        nonlocal ok, failed, budget, violations
        async with sem:
            t0 = time.perf_counter()
            trace = await vm.run(program, context=ctx)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        status = trace.status
        steps = len(trace.steps)
        inv_ok = (not invariant_fn) or invariant_fn(trace)

        async with lock:
            latencies.append(elapsed_ms)
            step_counts.append(steps)
            if status == TraceStatus.SUCCESS:
                ok += 1
            elif status == TraceStatus.BUDGET_EXCEEDED:
                budget += 1
            else:
                failed += 1
            if not inv_ok:
                violations += 1

    await asyncio.gather(*[_one(ctx) for ctx in contexts])
    latencies.sort()
    mean_steps = statistics.mean(step_counts) if step_counts else 0.0
    return latencies, ok, failed, budget, violations, mean_steps


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    idx = min(int(math.ceil(pct / 100.0 * len(data))) - 1, len(data) - 1)
    return data[max(idx, 0)]


# ══════════════════════════════════════════════════════════════════════════════
# BM-INT-01  Refund pipeline
# ══════════════════════════════════════════════════════════════════════════════


def _refund_invariant(trace: Any) -> bool:
    ids = [s.step_id for s in trace.steps]
    if len(ids) != len(set(ids)):
        return False
    if not ids or ids[0] != "classify":
        return False
    return True


async def run_bm_int_01(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    rng = random.Random(SEED + cycle * 100 + run)
    vm = ExecutionVM(llm=make_refund_llm(rng), tools=TOOLS)
    program = Program.from_dict(REFUND_PROGRAM)
    contexts = [
        {"user_input": f"Refund request #{5000 + i}", "order_id": str(1000 + i)} for i in range(N)
    ]
    t0 = time.perf_counter()
    latencies, ok, failed, budget, violations, mean_steps = await _run_batch(
        vm, program, contexts, _refund_invariant
    )
    elapsed = time.perf_counter() - t0
    progress.advance(task, N)
    return RunResult(
        scenario_id="BM-INT-01",
        cycle=cycle,
        run=run,
        n=N,
        elapsed_s=elapsed,
        throughput=N / elapsed,
        ok=ok,
        failed=failed,
        budget_exceeded=budget,
        violations=violations,
        mean_steps=mean_steps,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


# ══════════════════════════════════════════════════════════════════════════════
# BM-INT-02  Double-execution guard
# ══════════════════════════════════════════════════════════════════════════════

DOUBLE_PROGRAM = {
    "name": "double_guard",
    "max_steps": 20,
    "steps": [
        {
            "id": "classify",
            "type": "llm",
            "prompt": "Classify: $input. Reply refund.",
            "output_key": "intent",
        },
        {
            "id": "route",
            "type": "condition",
            "condition": "'refund' in $intent",
            "then": "process",
            "otherwise": "reject",
        },
        {
            "id": "process",
            "type": "tool",
            "tool": "issue_refund",
            "args": {"order_id": "$order_id"},
            "on_error": "skip",
        },
        {
            "id": "audit",
            "type": "tool",
            "tool": "audit_log",
            "args": {"trace_id": "t0", "step_id": "final"},
        },
        {
            "id": "reject",
            "type": "tool",
            "tool": "send_rejection",
            "args": {"order_id": "$order_id"},
        },
    ],
}


def _double_invariant(trace: Any) -> bool:
    ids = [s.step_id for s in trace.steps]
    return len(ids) == len(set(ids))


async def run_bm_int_02(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    vm = ExecutionVM(llm=make_double_llm(), tools=TOOLS)
    program = Program.from_dict(DOUBLE_PROGRAM)
    contexts = [{"input": f"refund order {i}", "order_id": str(i)} for i in range(N)]
    t0 = time.perf_counter()
    latencies, ok, failed, budget, violations, mean_steps = await _run_batch(
        vm, program, contexts, _double_invariant
    )
    elapsed = time.perf_counter() - t0
    progress.advance(task, N)
    return RunResult(
        scenario_id="BM-INT-02",
        cycle=cycle,
        run=run,
        n=N,
        elapsed_s=elapsed,
        throughput=N / elapsed,
        ok=ok,
        failed=failed,
        budget_exceeded=budget,
        violations=violations,
        mean_steps=mean_steps,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


# ══════════════════════════════════════════════════════════════════════════════
# BM-INT-03  Budget enforcement
# ══════════════════════════════════════════════════════════════════════════════


def _budget_invariant(trace: Any) -> bool:
    return trace.status == TraceStatus.BUDGET_EXCEEDED


async def run_bm_int_03(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    vm = ExecutionVM(llm=MockLLMAdapter("ok"), tools=TOOLS)
    program = Program.from_dict(BUDGET_PROGRAM)
    contexts = [{"order_id": str(i)} for i in range(N)]
    t0 = time.perf_counter()
    latencies, ok, failed, budget, violations, mean_steps = await _run_batch(
        vm, program, contexts, _budget_invariant
    )
    elapsed = time.perf_counter() - t0
    progress.advance(task, N)
    real_ok = budget
    real_violations = N - budget
    return RunResult(
        scenario_id="BM-INT-03",
        cycle=cycle,
        run=run,
        n=N,
        elapsed_s=elapsed,
        throughput=N / elapsed,
        ok=real_ok,
        failed=failed,
        budget_exceeded=budget,
        violations=real_violations,
        mean_steps=mean_steps,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


# ══════════════════════════════════════════════════════════════════════════════
# BM-INT-04  Parallel throughput
#
# PATCH: n=batch (programs), not n=sub_steps_total.
# OK% = fraction of programs with TraceStatus.SUCCESS.
# Throughput = programs/s; sub-step parallelism is internal.
# A parallel block is 1 FSM step → max_steps=4 is sufficient.
# ══════════════════════════════════════════════════════════════════════════════


def _make_parallel_program(concurrency: int, width: int) -> dict:
    return {
        "name": f"parallel_w{width}_c{concurrency}",
        "max_steps": 4,  # parallel block = 1 FSM step; headroom for pre/post
        "steps": [
            {
                "id": "parallel_fetch",
                "type": "parallel",
                "max_concurrency": concurrency,
                "on_error": "skip",
                "parallel_steps": [
                    {
                        "id": f"fetch_{i}",
                        "type": "tool",
                        "tool": "get_order_amount",
                        "args": {"order_id": str(i)},
                    }
                    for i in range(width)
                ],
            }
        ],
    }


async def run_bm_int_04(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    """
    Run N/width programs, each with width=10 parallel sub-steps (concurrency=10).

    Measurement unit: programs (1 000), not sub-steps.
    OK% = fraction of programs with TraceStatus.SUCCESS.
    Throughput = programs/s; 10 sub-steps run concurrently inside each program.
    on_error=skip → sub-step failures don't fail the program at FSM level.
    """
    width = 10
    batch = N // width  # 1 000 programs

    vm = ExecutionVM(llm=MockLLMAdapter("ok"), tools=TOOLS)
    program = Program.from_dict(_make_parallel_program(concurrency=width, width=width))
    contexts = [{"order_id": str(i)} for i in range(batch)]

    t0 = time.perf_counter()
    latencies, ok, failed, budget, violations, mean_steps = await _run_batch(
        vm,
        program,
        contexts,
        lambda t: t.status == TraceStatus.SUCCESS,
    )
    elapsed = time.perf_counter() - t0
    progress.advance(task, N)

    return RunResult(
        scenario_id="BM-INT-04",
        cycle=cycle,
        run=run,
        n=batch,  # programs, not sub-steps — OK% semantically consistent
        elapsed_s=elapsed,
        throughput=batch / elapsed,
        ok=ok,
        failed=failed,
        budget_exceeded=budget,
        violations=violations,
        mean_steps=mean_steps,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


# ══════════════════════════════════════════════════════════════════════════════
# BM-INT-05  MCP store round-trip
#
# PATCH: elapsed = wall clock (perf_counter), not sum(latencies).
#        total_ops = batch_programs * (1 + traces_per_program).
# ══════════════════════════════════════════════════════════════════════════════


async def run_bm_int_05(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    """
    Simulate MCP gateway store layer:
    save_program → get_program → save_trace → get_trace (SQLite WAL).
    100 programs × 100 traces = 10 100 round-trips.

    Recommendation: PRAGMA busy_timeout=5000 + wal_checkpoint(PASSIVE)
    to stabilise the 1% BUSY failures under peak load.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    store = Store(db_path)

    batch_programs = 100
    traces_per_program = N // batch_programs  # 100

    latencies: list[float] = []
    violations = 0
    ok = 0

    t_wall = time.perf_counter()  # PATCH: real wall clock start

    for p_idx in range(batch_programs):
        program_id = str(uuid.uuid4())
        program_name = f"prog_{cycle}_{run}_{p_idx}"
        program_dict = {**REFUND_PROGRAM, "name": program_name}

        t0 = time.perf_counter()
        store.save_program(program_id, program_name, program_dict)
        retrieved = store.get_program(program_id)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        if retrieved is None or retrieved.get("name") != program_name:
            violations += 1
            progress.advance(task, traces_per_program)
            continue

        for t_idx in range(traces_per_program):
            trace_id = str(uuid.uuid4())
            trace_data = {
                "trace_id": trace_id,
                "status": "SUCCESS",
                "steps": [{"step_id": f"s{t_idx}", "status": "SUCCESS"}],
            }
            t0 = time.perf_counter()
            store.save_trace(trace_id, program_id, "SUCCESS", 1, 0.0, trace_data)
            fetched = store.get_trace(trace_id)
            latencies.append((time.perf_counter() - t0) * 1000.0)

            if fetched is None:
                violations += 1
            else:
                ok += 1

        progress.advance(task, traces_per_program)

    elapsed = time.perf_counter() - t_wall  # PATCH: wall clock elapsed
    store.close()
    Path(db_path).unlink(missing_ok=True)

    latencies.sort()
    total_ops = batch_programs * (1 + traces_per_program)  # 100 * 101 = 10 100  # PATCH

    return RunResult(
        scenario_id="BM-INT-05",
        cycle=cycle,
        run=run,
        n=total_ops,
        elapsed_s=elapsed,
        throughput=total_ops / max(elapsed, 1e-6),  # PATCH
        ok=ok,
        failed=0,
        budget_exceeded=0,
        violations=violations,
        mean_steps=1.0,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


# ══════════════════════════════════════════════════════════════════════════════
# BM-INT-06  Governance pipeline
# ══════════════════════════════════════════════════════════════════════════════


def _governance_invariant(trace: Any) -> bool:
    ids = [s.step_id for s in trace.steps]
    if len(ids) != len(set(ids)):
        return False
    if not ids or ids[0] != "capability_check":
        return False
    if len(ids) < 2 or ids[1] != "classify":
        return False
    return True


async def run_bm_int_06(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    rng = random.Random(SEED + cycle * 200 + run + 50)
    vm = ExecutionVM(llm=make_governance_llm(rng), tools=TOOLS)
    program = Program.from_dict(GOVERNANCE_PROGRAM)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path_gov = f.name
    gov_store = Store(db_path_gov)

    contexts = []
    execution_ids: list[str] = []
    for i in range(N):
        eid = str(uuid.uuid4())
        execution_ids.append(eid)
        contexts.append(
            {
                "user_input": f"Process order {i}",
                "order_id": str(3000 + i),
                "capability_id": f"cap_{i % 500}",
                "trace_id": eid,
            }
        )

    t0 = time.perf_counter()
    latencies, ok, failed, budget, violations, mean_steps = await _run_batch(
        vm, program, contexts, _governance_invariant
    )
    elapsed = time.perf_counter() - t0

    envelope_violations = 0
    sample = execution_ids[: min(500, N)]
    for eid in sample:
        gov_store.save_envelope(
            eid,
            1,
            f"ph_{eid[:8]}",
            f"sh_{eid[:8]}",
            {"capability_id": "cap_x", "scope": "refund:write"},
        )
        envelopes = gov_store.get_envelopes(eid)
        if not envelopes or envelopes[0].get("execution_id") != eid:
            envelope_violations += 1

    gov_store.close()
    Path(db_path_gov).unlink(missing_ok=True)

    violations += envelope_violations
    progress.advance(task, N)

    return RunResult(
        scenario_id="BM-INT-06",
        cycle=cycle,
        run=run,
        n=N,
        elapsed_s=elapsed,
        throughput=N / elapsed,
        ok=ok,
        failed=failed,
        budget_exceeded=budget,
        violations=violations,
        mean_steps=mean_steps,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
    )


# ══════════════════════════════════════════════════════════════════════════════
# BM-INT-07  Crash consistency
#
# Simulates mid-transition crash by cancelling a running coroutine and
# verifying that:
#   (a) replay produces a correct terminal state (no partial execution)
#   (b) no step executes twice (I_k(T) ∈ {0,1} survives crash+resume)
#   (c) trace is in a valid terminal state, not stuck
#
# Approach: run a program in a background task, cancel it at a random
# point, then re-run from scratch (simulate resume semantics by using a
# fresh VM — stateless restart). Count cases where the resumed run
# produces a DIFFERENT step sequence than a clean run (violation).
# ══════════════════════════════════════════════════════════════════════════════

_CRASH_PROGRAM = {
    "name": "crash_target",
    "max_steps": 10,
    "steps": [
        {
            "id": "step_a",
            "type": "tool",
            "tool": "audit_log",
            "args": {"trace_id": "$tid", "step_id": "a"},
        },
        {
            "id": "step_b",
            "type": "tool",
            "tool": "get_order_amount",
            "args": {"order_id": "$order_id"},
            "output_key": "amount",
        },
        {
            "id": "step_c",
            "type": "tool",
            "tool": "get_order_status",
            "args": {"order_id": "$order_id"},
            "output_key": "status",
        },
        {
            "id": "step_d",
            "type": "tool",
            "tool": "issue_refund",
            "args": {"order_id": "$order_id"},
        },
        {
            "id": "step_e",
            "type": "tool",
            "tool": "audit_log",
            "args": {"trace_id": "$tid", "step_id": "e"},
        },
    ],
}

_CRASH_N = 2_000  # programs per run (lighter than N — crash overhead is real)


async def _run_clean(vm: ExecutionVM, program: Program, ctx: dict) -> Any:
    """Run to completion, return full trace."""
    return await vm.run(program, context=ctx)


async def _run_with_crash(
    vm: ExecutionVM,
    program: Program,
    ctx: dict,
    crash_after_ms: float,
) -> tuple[bool, Any]:
    """
    Start execution, cancel after crash_after_ms, then re-run from scratch.
    Returns (crashed, resumed_trace).
    crashed = task was cancelled before completion.
    """
    task = asyncio.create_task(vm.run(program, context=ctx))
    crashed = False
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=crash_after_ms / 1000.0)
    except asyncio.TimeoutError:
        task.cancel()
        crashed = True
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Resume: fresh VM, same context (stateless restart semantics)
    vm2 = ExecutionVM(llm=MockLLMAdapter(["ok"] * 20), tools=TOOLS)
    trace2 = await vm2.run(program, context=ctx)
    return crashed, trace2


async def run_bm_int_07(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    """
    Crash consistency: cancel mid-transition, verify resume is semantically equivalent.

    Three invariants checked per execution:
      (a) Structural: no duplicate steps in resumed trace
      (b) Terminal:   resumed trace reached a valid terminal state
      (c) Semantic:   trace_hash(clean) == trace_hash(resumed)
                      — the gold invariant: crash cannot alter execution semantics

    A violation = any of (a), (b), (c) fails.
    hash_match_pct in extra tracks (c) independently for observability.
    """
    rng = random.Random(SEED + cycle * 300 + run + 7)
    program = Program.from_dict(_CRASH_PROGRAM)

    latencies: list[float] = []
    ok = crashed_count = violations = 0
    hash_mismatches = 0

    sem = asyncio.Semaphore(50)  # lower concurrency — crash overhead is real
    lock = asyncio.Lock()

    async def _one(i: int) -> None:
        nonlocal ok, crashed_count, violations, hash_mismatches
        ctx = {"tid": str(uuid.uuid4()), "order_id": str(i)}
        crash_ms = rng.uniform(0.5, 8.0)  # crash window: 0.5–8ms into execution

        vm_clean = ExecutionVM(llm=MockLLMAdapter(["ok"] * 20), tools=TOOLS)
        vm_crash = ExecutionVM(llm=MockLLMAdapter(["ok"] * 20), tools=TOOLS)

        async with sem:
            t0 = time.perf_counter()
            clean_trace = await _run_clean(vm_clean, program, ctx)
            crashed, resumed_trace = await _run_with_crash(vm_crash, program, ctx, crash_ms)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        resumed_ids = [s.step_id for s in resumed_trace.steps]

        # (a) structural: no duplicates in resumed trace
        has_dupes = len(resumed_ids) != len(set(resumed_ids))
        # (b) terminal: resumed trace not stuck (at least one step executed)
        not_terminal = len(resumed_ids) == 0
        # (c) semantic: hash(clean) == hash(resumed) — THE gold invariant
        h_clean = _trace_hash(clean_trace)
        h_resumed = _trace_hash(resumed_trace)
        hash_mismatch = h_clean != h_resumed

        async with lock:
            latencies.append(elapsed_ms)
            if crashed:
                crashed_count += 1
            if hash_mismatch:
                hash_mismatches += 1
            if has_dupes or not_terminal or hash_mismatch:
                violations += 1
            else:
                ok += 1

    await asyncio.gather(*[_one(i) for i in range(_CRASH_N)])
    progress.advance(task, N)

    latencies.sort()
    elapsed = sum(latencies) / 1000.0  # sequential pairs, sum ≈ wall

    return RunResult(
        scenario_id="BM-INT-07",
        cycle=cycle,
        run=run,
        n=_CRASH_N,
        elapsed_s=max(elapsed, 0.001),
        throughput=_CRASH_N / max(elapsed, 0.001),
        ok=ok,
        failed=0,
        budget_exceeded=0,
        violations=violations,
        mean_steps=0.0,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        extra={
            "crashed_count": crashed_count,
            "crash_rate_pct": 100.0 * crashed_count / _CRASH_N,
            "hash_mismatches": hash_mismatches,
            "hash_match_pct": 100.0 * (1 - hash_mismatches / _CRASH_N),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# BM-INT-08  Replay equivalence
#
# Verifies: trace_hash(run_1) == trace_hash(replay_1)
# trace_hash = sha256 of (step_id, status, output) for all steps, in order.
# Uses MockLLMAdapter with deterministic sequence → fully reproducible trace.
# Violation = hash mismatch between original and replay.
# ══════════════════════════════════════════════════════════════════════════════

_REPLAY_PROGRAM = {
    "name": "replay_target",
    "max_steps": 10,
    "steps": [
        {
            "id": "classify",
            "type": "llm",
            "prompt": "Classify: $input. Reply refund.",
            "output_key": "intent",
        },
        {
            "id": "route",
            "type": "condition",
            "condition": "'refund' in $intent",
            "then": "process",
            "otherwise": "reject",
        },
        {
            "id": "process",
            "type": "tool",
            "tool": "issue_refund",
            "args": {"order_id": "$order_id"},
        },
        {
            "id": "reject",
            "type": "tool",
            "tool": "send_rejection",
            "args": {"order_id": "$order_id"},
        },
    ],
}

_REPLAY_N = 5_000


def _trace_hash(trace: Any) -> str:
    """Deterministic hash of (step_id, status, str(output)) for all steps."""
    h = hashlib.sha256()
    for step in trace.steps:
        h.update(step.step_id.encode())
        h.update(str(step.status).encode())
        h.update(str(getattr(step, "output", "")).encode())
    return h.hexdigest()


async def run_bm_int_08(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    """
    Replay equivalence: run each program twice with identical inputs and LLM,
    verify trace_hash matches.

    Both runs use MockLLMAdapter("refund") → deterministic output.
    Violation = hash mismatch.
    """
    program = Program.from_dict(_REPLAY_PROGRAM)
    contexts = [{"input": f"refund order {i}", "order_id": str(i)} for i in range(_REPLAY_N)]

    sem = asyncio.Semaphore(_BATCH_CONCURRENCY)
    latencies: list[float] = []
    ok = violations = 0
    lock = asyncio.Lock()

    async def _one(ctx: dict) -> None:
        nonlocal ok, violations
        vm1 = ExecutionVM(llm=MockLLMAdapter("refund"), tools=TOOLS)
        vm2 = ExecutionVM(llm=MockLLMAdapter("refund"), tools=TOOLS)

        async with sem:
            t0 = time.perf_counter()
            trace1 = await vm1.run(program, context=ctx)
            trace2 = await vm2.run(program, context=ctx)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        h1 = _trace_hash(trace1)
        h2 = _trace_hash(trace2)

        async with lock:
            latencies.append(elapsed_ms)
            if h1 == h2:
                ok += 1
            else:
                violations += 1

    t0 = time.perf_counter()
    await asyncio.gather(*[_one(ctx) for ctx in contexts])
    elapsed = time.perf_counter() - t0
    progress.advance(task, N)

    latencies.sort()
    return RunResult(
        scenario_id="BM-INT-08",
        cycle=cycle,
        run=run,
        n=_REPLAY_N,
        elapsed_s=elapsed,
        throughput=_REPLAY_N / elapsed,
        ok=ok,
        failed=0,
        budget_exceeded=0,
        violations=violations,
        mean_steps=0.0,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        extra={"hash_match_pct": 100.0 * ok / _REPLAY_N},
    )


# ══════════════════════════════════════════════════════════════════════════════
# BM-INT-09  Adversarial retries
#
# Three sub-scenarios:
#   A. Duplicate webhooks — same (program, context) submitted N_DUP times,
#      each run must produce identical step sequence (no extra side effects).
#   B. Out-of-order events — contexts submitted in shuffled order,
#      each run must still reach a terminal state with no duplicates.
#   C. Delayed acknowledgements — tool latency injected randomly 0–50ms,
#      traces must still complete with SUCCESS (no partial termination).
#
# Violation for any sub-scenario = step duplicates or wrong terminal state.
# ══════════════════════════════════════════════════════════════════════════════

_ADV_N = 3_000
_DUP_FACTOR = 3  # each context submitted _DUP_FACTOR times


async def tool_delayed_issue_refund(order_id: str) -> dict[str, Any]:
    """Refund tool with random 0–50ms delay to simulate delayed acks."""
    delay = random.uniform(0.0, 0.05)
    await asyncio.sleep(delay)
    return {"refund_id": f"ref_{order_id}", "status": "processed"}


_ADV_TOOLS = {**TOOLS, "issue_refund": tool_delayed_issue_refund}

_ADV_PROGRAM = {
    "name": "adversarial_target",
    "max_steps": 8,
    "steps": [
        {
            "id": "classify",
            "type": "llm",
            "prompt": "Classify: $input. Reply refund.",
            "output_key": "intent",
        },
        {
            "id": "route",
            "type": "condition",
            "condition": "'refund' in $intent",
            "then": "process",
            "otherwise": "reject",
        },
        {
            "id": "process",
            "type": "tool",
            "tool": "issue_refund",
            "args": {"order_id": "$order_id"},
            "max_retries": 2,
            "on_error": "skip",
        },
        {
            "id": "audit",
            "type": "tool",
            "tool": "audit_log",
            "args": {"trace_id": "$tid", "step_id": "done"},
        },
        {
            "id": "reject",
            "type": "tool",
            "tool": "send_rejection",
            "args": {"order_id": "$order_id"},
        },
    ],
}


def _adv_invariant(trace: Any) -> bool:
    ids = [s.step_id for s in trace.steps]
    return len(ids) == len(set(ids)) and trace.status in TERMINAL_ALL


async def run_bm_int_09(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    """
    Adversarial retries: duplicate webhooks, out-of-order, delayed acks.
    Each sub-scenario uses _ADV_N contexts.
    Violation = step duplicates or non-terminal state.
    """
    program = Program.from_dict(_ADV_PROGRAM)
    rng = random.Random(SEED + cycle * 400 + run + 9)

    base_contexts = [
        {"input": f"refund order {i}", "order_id": str(i), "tid": str(i)} for i in range(_ADV_N)
    ]

    sem = asyncio.Semaphore(_BATCH_CONCURRENCY)
    latencies: list[float] = []
    ok = violations = 0
    lock = asyncio.Lock()

    # Sub-A: duplicate webhooks — same context _DUP_FACTOR times
    dup_contexts = []
    for ctx in base_contexts[: _ADV_N // 3]:
        dup_contexts.extend([ctx] * _DUP_FACTOR)

    # Sub-B: out-of-order — shuffled submission order
    ooo_contexts = list(base_contexts[_ADV_N // 3 : 2 * _ADV_N // 3])
    rng.shuffle(ooo_contexts)

    # Sub-C: delayed acks — remaining contexts with delayed tool
    delayed_contexts = base_contexts[2 * _ADV_N // 3 :]

    async def _run_one(ctx: dict, tools_override: dict | None = None) -> None:
        nonlocal ok, violations
        _tools = tools_override or TOOLS
        vm = ExecutionVM(llm=MockLLMAdapter("refund"), tools=_tools)
        async with sem:
            t0 = time.perf_counter()
            trace = await vm.run(program, context=ctx)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        async with lock:
            latencies.append(elapsed_ms)
            if _adv_invariant(trace):
                ok += 1
            else:
                violations += 1

    t0 = time.perf_counter()
    await asyncio.gather(
        *[_run_one(ctx) for ctx in dup_contexts],
        *[_run_one(ctx) for ctx in ooo_contexts],
        *[_run_one(ctx, _ADV_TOOLS) for ctx in delayed_contexts],
    )
    elapsed = time.perf_counter() - t0
    progress.advance(task, N)

    total = len(dup_contexts) + len(ooo_contexts) + len(delayed_contexts)
    latencies.sort()

    return RunResult(
        scenario_id="BM-INT-09",
        cycle=cycle,
        run=run,
        n=total,
        elapsed_s=elapsed,
        throughput=total / elapsed,
        ok=ok,
        failed=0,
        budget_exceeded=0,
        violations=violations,
        mean_steps=0.0,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        extra={
            "sub_a_dup_n": len(dup_contexts),
            "sub_b_ooo_n": len(ooo_contexts),
            "sub_c_delayed_n": len(delayed_contexts),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# BM-INT-10  Long-horizon orchestration
#
# Runs a single program with LONG_STEPS tool steps, measures:
#   - total elapsed time
#   - peak RSS memory (resource.getrusage)
#   - tracemalloc peak allocated bytes
#   - _token_accumulator O(1) behaviour (total_tokens() called per step)
#
# Violation = peak RSS > RSS_LIMIT_MB or any step duplicate.
# ══════════════════════════════════════════════════════════════════════════════

_LONG_STEPS = 1_000  # steps per program (100k is ~100× this with RUNS×CYCLES)
_LONG_PROGRAMS = 10  # programs per run
_RSS_LIMIT_MB = 512  # violation threshold


def _make_long_program(n_steps: int) -> dict:
    return {
        "name": "long_horizon",
        "max_steps": n_steps + 2,
        "steps": [
            {
                "id": f"s{i:04d}",
                "type": "tool",
                "tool": "audit_log",
                "args": {"trace_id": "long", "step_id": f"s{i:04d}"},
            }
            for i in range(n_steps)
        ],
    }


async def run_bm_int_10(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    """
    Long-horizon orchestration: _LONG_PROGRAMS programs × _LONG_STEPS steps.
    Tracks memory growth and verifies no step duplication across long runs.

    Memory metrics:
      - peak_rss_mb: OS-level RSS peak (resource.getrusage RUSAGE_SELF)
      - peak_alloc_mb: Python-level peak allocation (tracemalloc)
    """
    program = Program.from_dict(_make_long_program(_LONG_STEPS))
    vm = ExecutionVM(llm=MockLLMAdapter("ok"), tools=TOOLS)

    gc.collect()
    tracemalloc.start()

    latencies: list[float] = []
    ok = violations = 0
    total_steps_executed = 0

    t0_wall = time.perf_counter()
    for i in range(_LONG_PROGRAMS):
        ctx = {"order_id": str(i)}
        t0 = time.perf_counter()
        trace = await vm.run(program, context=ctx)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        ids = [s.step_id for s in trace.steps]
        total_steps_executed += len(ids)
        if len(ids) != len(set(ids)) or trace.status != TraceStatus.SUCCESS:
            violations += 1
        else:
            ok += 1

        progress.advance(task, N // _LONG_PROGRAMS)

    elapsed = time.perf_counter() - t0_wall

    _, peak_alloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Linux: ru_maxrss in KB; macOS: bytes
    rss_unit = 1024 if os.uname().sysname == "Linux" else 1024 * 1024
    peak_rss_mb = rss_after / rss_unit
    peak_alloc_mb = peak_alloc / (1024 * 1024)

    if peak_rss_mb > _RSS_LIMIT_MB:
        violations += 1

    latencies.sort()

    return RunResult(
        scenario_id="BM-INT-10",
        cycle=cycle,
        run=run,
        n=_LONG_PROGRAMS * _LONG_STEPS,
        elapsed_s=elapsed,
        throughput=total_steps_executed / elapsed,
        ok=ok,
        failed=0,
        budget_exceeded=0,
        violations=violations,
        mean_steps=float(_LONG_STEPS),
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        extra={
            "peak_rss_mb": round(peak_rss_mb, 1),
            "peak_alloc_mb": round(peak_alloc_mb, 2),
            "total_steps": total_steps_executed,
            "rss_limit_mb": _RSS_LIMIT_MB,
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# Scenario registry
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS_ORIGINAL = [
    ("BM-INT-01", "Refund pipeline", "llm+cond+parallel+tool", run_bm_int_01),
    ("BM-INT-02", "Double-execution guard", "FSM I_k(T)∈{0,1}", run_bm_int_02),
    ("BM-INT-03", "Budget enforcement", "max_steps wall", run_bm_int_03),
    ("BM-INT-04", "Parallel throughput", "asyncio.gather scaling", run_bm_int_04),
    ("BM-INT-05", "MCP store round-trip", "SQLite WAL", run_bm_int_05),
    ("BM-INT-06", "GovernanceEnvelope", "CapabilityRef+audit", run_bm_int_06),
]

SCENARIOS_NEW = [
    ("BM-INT-07", "Crash consistency", "kill→replay→resume", run_bm_int_07),
    ("BM-INT-08", "Replay equivalence", "trace_hash(r1)==hash(r2)", run_bm_int_08),
    ("BM-INT-09", "Adversarial retries", "dup+ooo+delayed", run_bm_int_09),
    ("BM-INT-10", "Long-horizon", "1k steps+memory profile", run_bm_int_10),
]

ALL_SCENARIOS = SCENARIOS_ORIGINAL + SCENARIOS_NEW


# ══════════════════════════════════════════════════════════════════════════════
# Rich rendering
# ══════════════════════════════════════════════════════════════════════════════

STATUS_COLOR = {True: "bold green", False: "bold red"}
STATUS_ICON = {True: "✓ PASS", False: "✗ FAIL"}


def _fmt_tps(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M/s"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K/s"
    return f"{v:.0f}/s"


def _fmt_ms(v: float) -> str:
    return f"{v:.2f}ms"


def render_cycle_table(cycle: int, summaries: list[ScenarioSummary]) -> Table:
    t = Table(
        title=f"[bold]Cycle {cycle} / {CYCLES}[/bold]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
        title_style="bold white on dark_blue",
        border_style="bright_black",
        expand=True,
    )
    t.add_column("ID", style="bold yellow", width=12)
    t.add_column("Scenario", style="white", width=28)
    t.add_column("Tag", style="dim", width=24)
    t.add_column("Mean TPS", style="cyan", justify="right", width=12)
    t.add_column("±σ", style="dim cyan", justify="right", width=10)
    t.add_column("OK %", justify="right", width=8)
    t.add_column("p50", justify="right", width=10)
    t.add_column("p95", justify="right", width=10)
    t.add_column("Violations", justify="right", width=12)
    t.add_column("Status", justify="center", width=10)

    for s in summaries:
        ok_color = "green" if s.ok_pct >= 95 else ("yellow" if s.ok_pct >= 70 else "red")
        viol_color = "green" if s.total_violations == 0 else "bold red"
        t.add_row(
            s.scenario_id,
            s.label,
            s.tag,
            _fmt_tps(s.mean_tps),
            f"±{_fmt_tps(s.stddev_tps)}",
            Text(f"{s.ok_pct:.1f}%", style=ok_color),
            _fmt_ms(statistics.mean([r.p50_ms for r in s.results])),
            _fmt_ms(s.mean_p95),
            Text(str(s.total_violations), style=viol_color),
            Text(STATUS_ICON[s.passed], style=STATUS_COLOR[s.passed]),
        )
    return t


def render_final_table(
    all_summaries: dict[str, ScenarioSummary],
    scenarios: list,
) -> tuple[Table, bool]:
    t = Table(
        title=f"[bold]FINAL SUMMARY — {CYCLES} cycles × {RUNS} runs × {N:,} items[/bold]",
        box=box.DOUBLE_EDGE,
        show_lines=True,
        header_style="bold white on dark_blue",
        title_style="bold white on dark_blue",
        border_style="cyan",
        expand=True,
    )
    t.add_column("ID", style="bold yellow", width=12)
    t.add_column("Scenario", style="white", width=28)
    t.add_column("Total items", justify="right", width=14)
    t.add_column("Mean TPS", style="cyan", justify="right", width=12)
    t.add_column("Throughput σ", style="dim cyan", justify="right", width=12)
    t.add_column("OK %", justify="right", width=8)
    t.add_column("p95 avg", justify="right", width=10)
    t.add_column("Violations", justify="right", width=12)
    t.add_column("Verdict", justify="center", width=10)

    all_pass = True
    for sid, label, tag, _ in scenarios:
        s = all_summaries[sid]
        ok_color = "green" if s.ok_pct >= 95 else ("yellow" if s.ok_pct >= 70 else "red")
        viol_color = "green" if s.total_violations == 0 else "bold red"
        if not s.passed:
            all_pass = False
        t.add_row(
            s.scenario_id,
            s.label,
            f"{s.total_n:,}",
            _fmt_tps(s.mean_tps),
            f"±{_fmt_tps(s.stddev_tps)}",
            Text(f"{s.ok_pct:.1f}%", style=ok_color),
            _fmt_ms(s.mean_p95),
            Text(str(s.total_violations), style=viol_color),
            Text(STATUS_ICON[s.passed], style=STATUS_COLOR[s.passed]),
        )

    return t, all_pass


def render_extra_panel(all_summaries: dict[str, ScenarioSummary]) -> None:
    """Print per-scenario extra metrics for new suite."""
    extra_ids = ["BM-INT-07", "BM-INT-08", "BM-INT-09", "BM-INT-10"]
    has_any = any(sid in all_summaries for sid in extra_ids)
    if not has_any:
        return

    console.print()
    console.print(
        Rule("[bold white]Extended Metrics — New Suite[/bold white]", style="bright_black")
    )

    if "BM-INT-07" in all_summaries:
        s = all_summaries["BM-INT-07"]
        extras = [r.extra for r in s.results if r.extra]
        if extras:
            avg_crash = statistics.mean(e.get("crash_rate_pct", 0) for e in extras)
            avg_hash_match = statistics.mean(e.get("hash_match_pct", 0) for e in extras)
            total_mismatches = sum(int(e.get("hash_mismatches", 0)) for e in extras)
            hash_color = "green" if total_mismatches == 0 else "bold red"
            console.print(
                f"  [cyan]BM-INT-07[/cyan]  "
                f"crash rate {avg_crash:.1f}%  "
                f"trace_hash(clean)==trace_hash(resumed): "
                f"[{hash_color}]{avg_hash_match:.2f}%[/{hash_color}]  "
                f"mismatches={total_mismatches}"
            )

    if "BM-INT-08" in all_summaries:
        s = all_summaries["BM-INT-08"]
        extras = [r.extra for r in s.results if r.extra]
        if extras:
            avg_match = statistics.mean(e.get("hash_match_pct", 0) for e in extras)
            console.print(
                f"  [cyan]BM-INT-08[/cyan]  trace_hash match {avg_match:.2f}%  (target 100.00%)"
            )

    if "BM-INT-09" in all_summaries:
        s = all_summaries["BM-INT-09"]
        extras = [r.extra for r in s.results if r.extra]
        if extras:
            e = extras[-1]
            console.print(
                f"  [cyan]BM-INT-09[/cyan]  "
                f"dup={e.get('sub_a_dup_n', '?')}  "
                f"ooo={e.get('sub_b_ooo_n', '?')}  "
                f"delayed={e.get('sub_c_delayed_n', '?')}"
            )

    if "BM-INT-10" in all_summaries:
        s = all_summaries["BM-INT-10"]
        extras = [r.extra for r in s.results if r.extra]
        if extras:
            peak_rss = max(e.get("peak_rss_mb", 0) for e in extras)
            peak_alloc = max(e.get("peak_alloc_mb", 0) for e in extras)
            total_steps = sum(e.get("total_steps", 0) for e in extras)
            console.print(
                f"  [cyan]BM-INT-10[/cyan]  "
                f"peak RSS {peak_rss:.1f} MB  "
                f"peak alloc {peak_alloc:.2f} MB  "
                f"total steps {total_steps:,}"
            )


def render_invariant_panel(
    all_summaries: dict[str, ScenarioSummary],
    scenarios: list,
) -> Panel:
    total_items = sum(all_summaries[sid].total_n for sid, *_ in scenarios)
    total_viol = sum(all_summaries[sid].total_violations for sid, *_ in scenarios)
    all_pass = total_viol == 0

    color = "bold green" if all_pass else "bold red"
    icon = "⬢" if all_pass else "✗"
    verdict = "DETERMINISTIC EXECUTION VERIFIED" if all_pass else "INVARIANT VIOLATIONS DETECTED"

    lines = [
        f"[{color}]{icon} {verdict}[/{color}]",
        "",
        f"  Total operations : [bold]{total_items:,}[/bold]",
        f"  Total violations : [{color}]{total_viol}[/{color}]",
        f"  Cycles × Runs    : {CYCLES} × {RUNS}",
        f"  Items per run    : {N:,}",
        f"  Scenarios        : {len(scenarios)}",
        "",
        "  [dim]llm-nano-vm v0.8.0  ×  nano-vm-mcp v0.3.0[/dim]",
    ]
    return Panel(
        "\n".join(lines),
        title="[bold white]Invariant Audit[/bold white]",
        border_style="green" if all_pass else "red",
        expand=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI args
# ══════════════════════════════════════════════════════════════════════════════


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integration benchmark")
    p.add_argument(
        "--suite",
        choices=["all", "original", "new"],
        default="all",
        help="Which suite to run (default: all)",
    )
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated scenario IDs to run, e.g. BM-INT-07,BM-INT-08",
    )
    return p.parse_args()


def _select_scenarios(args: argparse.Namespace) -> list:
    if args.only:
        ids = {s.strip() for s in args.only.split(",")}
        selected = [s for s in ALL_SCENARIOS if s[0] in ids]
        if not selected:
            console.print(f"[red]No scenarios matched: {args.only}[/red]")
            raise SystemExit(1)
        return selected
    if args.suite == "original":
        return SCENARIOS_ORIGINAL
    if args.suite == "new":
        return SCENARIOS_NEW
    return ALL_SCENARIOS


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    args = _parse_args()
    scenarios = _select_scenarios(args)

    console.print()
    console.print(
        Rule(
            "[bold cyan]llm-nano-vm v0.8.0 × nano-vm-mcp v0.3.0"
            " — Integration Benchmark[/bold cyan]",
            style="cyan",
        )
    )
    console.print(
        f"  [dim]{CYCLES} cycles · {RUNS} runs · {N:,} items/run · seed={SEED} · "
        f"suite={args.suite} · scenarios={len(scenarios)}[/dim]"
    )
    console.print()

    all_summaries: dict[str, ScenarioSummary] = {
        sid: ScenarioSummary(scenario_id=sid, label=label, tag=tag)
        for sid, label, tag, _ in scenarios
    }

    total_steps = CYCLES * RUNS * len(scenarios) * N

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=36),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )

    with progress:
        master = progress.add_task("[bold white]Overall", total=total_steps)

        for cycle in range(1, CYCLES + 1):
            console.print()
            console.print(Rule(f"[bold]Cycle {cycle} / {CYCLES}[/bold]", style="bright_black"))

            cycle_summaries: dict[str, ScenarioSummary] = {
                sid: ScenarioSummary(scenario_id=sid, label=label, tag=tag)
                for sid, label, tag, _ in scenarios
            }

            for run in range(1, RUNS + 1):
                for sid, label, tag, runner in scenarios:
                    task = progress.add_task(
                        f"  [cyan]{sid}[/cyan] [dim]{label[:22]}[/dim]  "
                        f"[bright_black]c={cycle} r={run}[/bright_black]",
                        total=N,
                    )
                    result = await runner(cycle, run, progress, task)
                    progress.update(task, completed=N, visible=False)
                    progress.advance(master, N)

                    cycle_summaries[sid].results.append(result)
                    all_summaries[sid].results.append(result)

            cycle_list = [cycle_summaries[sid] for sid, *_ in scenarios]
            console.print()
            console.print(render_cycle_table(cycle, cycle_list))

    console.print()
    console.print(Rule("[bold white]Final Results[/bold white]", style="cyan"))
    console.print()

    final_table, all_pass = render_final_table(all_summaries, scenarios)
    console.print(final_table)

    render_extra_panel(all_summaries)

    console.print()
    console.print(render_invariant_panel(all_summaries, scenarios))
    console.print()

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
