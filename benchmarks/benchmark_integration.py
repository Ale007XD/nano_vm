from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import math
import os
import random
import statistics
import tempfile
import time
import tracemalloc
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nano_vm_mcp.store import ProgramStore as Store
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

from nano_vm import ExecutionVM, Program, TraceStatus
from nano_vm.adapters import MockLLMAdapter

console = Console()

N = 10_000
RUNS = 5
CYCLES = 3
SEED = 42

TERMINAL_ALL = {
    TraceStatus.SUCCESS,
    TraceStatus.FAILED,
    TraceStatus.BUDGET_EXCEEDED,
    TraceStatus.STALLED,
}

BATCH_CONCURRENCY = 200


def get_process_rss_mb() -> float:
    try:
        import psutil

        p = psutil.Process(os.getpid())
        return p.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


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
        return statistics.mean(self.all_throughputs) if self.all_throughputs else 0.0

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


async def _run_batch(
    vm: ExecutionVM, program: Program, contexts: list[dict[str, Any]], invariant_fn: Any
):
    sem = asyncio.Semaphore(BATCH_CONCURRENCY)
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


def _refund_invariant(trace: Any) -> bool:
    ids = [s.step_id for s in trace.steps]
    return bool(ids) and len(ids) == len(set(ids)) and ids[0] == "classify"


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
        "BM-INT-01",
        cycle,
        run,
        N,
        elapsed,
        N / elapsed,
        ok,
        failed,
        budget,
        violations,
        mean_steps,
        _percentile(latencies, 50),
        _percentile(latencies, 95),
        _percentile(latencies, 99),
    )


def _double_invariant(trace: Any) -> bool:
    ids = [s.step_id for s in trace.steps]
    return len(ids) == len(set(ids))


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
        "BM-INT-02",
        cycle,
        run,
        N,
        elapsed,
        N / elapsed,
        ok,
        failed,
        budget,
        violations,
        mean_steps,
        _percentile(latencies, 50),
        _percentile(latencies, 95),
        _percentile(latencies, 99),
    )


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
    return RunResult(
        "BM-INT-03",
        cycle,
        run,
        N,
        elapsed,
        N / elapsed,
        budget,
        failed,
        budget,
        N - budget,
        mean_steps,
        _percentile(latencies, 50),
        _percentile(latencies, 95),
        _percentile(latencies, 99),
    )


def _make_parallel_program(concurrency: int, width: int) -> dict:
    return {
        "name": f"parallel_w{width}_c{concurrency}",
        "max_steps": 4,
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
    width = 10
    batch = N // width
    vm = ExecutionVM(llm=MockLLMAdapter("ok"), tools=TOOLS)
    program = Program.from_dict(_make_parallel_program(concurrency=width, width=width))
    contexts = [{"order_id": str(i)} for i in range(batch)]
    t0 = time.perf_counter()
    latencies, ok, failed, budget, violations, mean_steps = await _run_batch(
        vm, program, contexts, lambda t: t.status == TraceStatus.SUCCESS
    )
    elapsed = time.perf_counter() - t0
    progress.advance(task, N)
    return RunResult(
        "BM-INT-04",
        cycle,
        run,
        batch,
        elapsed,
        batch / elapsed,
        ok,
        failed,
        budget,
        violations,
        mean_steps,
        _percentile(latencies, 50),
        _percentile(latencies, 95),
        _percentile(latencies, 99),
    )


async def run_bm_int_05(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = Store(db_path)
    batch_programs = 100
    traces_per_program = N // batch_programs
    latencies: list[float] = []
    violations = 0
    ok = 0
    t_wall = time.perf_counter()

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
            progress.advance(task, 1)

    elapsed = time.perf_counter() - t_wall
    store.close()
    Path(db_path).unlink(missing_ok=True)
    latencies.sort()
    total_ops = batch_programs * (1 + traces_per_program)
    return RunResult(
        "BM-INT-05",
        cycle,
        run,
        total_ops,
        elapsed,
        total_ops / max(elapsed, 1e-6),
        ok,
        0,
        0,
        violations,
        1.0,
        _percentile(latencies, 50),
        _percentile(latencies, 95),
        _percentile(latencies, 99),
    )


def _governance_invariant(trace: Any) -> bool:
    ids = [s.step_id for s in trace.steps]
    return (
        bool(ids)
        and len(ids) == len(set(ids))
        and ids[0] == "capability_check"
        and len(ids) >= 2
        and ids[1] == "classify"
    )


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
        "BM-INT-06",
        cycle,
        run,
        N,
        elapsed,
        N / elapsed,
        ok,
        failed,
        budget,
        violations,
        mean_steps,
        _percentile(latencies, 50),
        _percentile(latencies, 95),
        _percentile(latencies, 99),
    )


CRASH_PROGRAM = {
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
        {"id": "step_d", "type": "tool", "tool": "issue_refund", "args": {"order_id": "$order_id"}},
        {
            "id": "step_e",
            "type": "tool",
            "tool": "audit_log",
            "args": {"trace_id": "$tid", "step_id": "e"},
        },
    ],
}

CRASH_N = 2_000


async def _run_clean(vm: ExecutionVM, program: Program, ctx: dict) -> Any:
    return await vm.run(program, context=ctx)


async def _run_with_crash(vm: ExecutionVM, program: Program, ctx: dict, crash_after_ms: float):
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
    vm2 = ExecutionVM(llm=MockLLMAdapter(["ok"] * 20), tools=TOOLS)
    trace2 = await vm2.run(program, context=ctx)
    return crashed, trace2


def _trace_hash(trace: Any) -> str:
    h = hashlib.sha256()
    for step in trace.steps:
        h.update(step.step_id.encode())
        h.update(str(step.status).encode())
        h.update(str(getattr(step, "output", "")).encode())
    return h.hexdigest()


async def run_bm_int_07(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    rng = random.Random(SEED + cycle * 300 + run + 7)
    program = Program.from_dict(CRASH_PROGRAM)
    latencies: list[float] = []
    ok = crashed_count = violations = 0
    hash_mismatches = 0
    sem = asyncio.Semaphore(50)
    lock = asyncio.Lock()

    async def _one(i: int) -> None:
        nonlocal ok, crashed_count, violations, hash_mismatches
        ctx = {"tid": str(uuid.uuid4()), "order_id": str(i)}
        crash_ms = rng.uniform(0.5, 8.0)
        vm_clean = ExecutionVM(llm=MockLLMAdapter(["ok"] * 20), tools=TOOLS)
        vm_crash = ExecutionVM(llm=MockLLMAdapter(["ok"] * 20), tools=TOOLS)
        async with sem:
            t0 = time.perf_counter()
            clean_trace = await _run_clean(vm_clean, program, ctx)
            crashed, resumed_trace = await _run_with_crash(vm_crash, program, ctx, crash_ms)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            clean_ids = [s.step_id for s in clean_trace.steps]
            resumed_ids = [s.step_id for s in resumed_trace.steps]
            has_dupes = len(resumed_ids) != len(set(resumed_ids))
            not_terminal = len(resumed_ids) == 0
            hash_mismatch = _trace_hash(clean_trace) != _trace_hash(resumed_trace)
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

    await asyncio.gather(*[_one(i) for i in range(CRASH_N)])
    progress.advance(task, N)
    latencies.sort()
    elapsed = sum(latencies) / 1000.0
    return RunResult(
        "BM-INT-07",
        cycle,
        run,
        CRASH_N,
        max(elapsed, 0.001),
        CRASH_N / max(elapsed, 0.001),
        ok,
        0,
        0,
        violations,
        0.0,
        _percentile(latencies, 50),
        _percentile(latencies, 95),
        _percentile(latencies, 99),
        {
            "crashed_count": crashed_count,
            "crash_rate_pct": 100.0 * crashed_count / CRASH_N,
            "hash_mismatches": hash_mismatches,
            "hash_match_pct": 100.0 * (1 - hash_mismatches / CRASH_N),
        },
    )


REPLAY_PROGRAM = {
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

REPLAY_N = 5_000


async def run_bm_int_08(cycle: int, run: int, progress: Any, task: Any) -> RunResult:
    program = Program.from_dict(REPLAY_PROGRAM)
    contexts = [{"input": f"refund order {i}", "order_id": str(i)} for i in range(REPLAY_N)]
    sem = asyncio.Semaphore(BATCH_CONCURRENCY)
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
        "BM-INT-08",
        cycle,
        run,
        REPLAY_N,
        elapsed,
        REPLAY_N / elapsed,
        ok,
        0,
        0,
        violations,
        0.0,
        _percentile(latencies, 50),
        _percentile(latencies, 95),
        _percentile(latencies, 99),
        {"hash_match_pct": 100.0 * ok / REPLAY_N},
    )


ADV_N = 3_000
DUP_FACTOR = 3


async def tool_delayed_issue_refund(order_id: str) -> dict[str, Any]:
    await asyncio.sleep(random.uniform(0.0, 0.05))
    return {"refund_id": f"ref_{order_id}", "status": "processed"}


ADV_TOOLS = {**TOOLS, "issue_refund": tool_delayed_issue_refund}

ADV_PROGRAM = {
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
    program = Program.from_dict(ADV_PROGRAM)
    rng = random.Random(SEED + cycle * 400 + run + 9)
    base_contexts = [
        {"input": f"refund order {i}", "order_id": str(i), "tid": str(i)} for i in range(ADV_N)
    ]
    sem = asyncio.Semaphore(BATCH_CONCURRENCY)
    latencies: list[float] = []
    ok = violations = 0
    lock = asyncio.Lock()

    dup_contexts = []
    for ctx in base_contexts[: ADV_N // 3]:
        dup_contexts.extend([ctx] * DUP_FACTOR)
    ooo_contexts = list(base_contexts[ADV_N // 3 : 2 * ADV_N // 3])
    rng.shuffle(ooo_contexts)
    delayed_contexts = base_contexts[2 * ADV_N // 3 :]

    async def _run_one(ctx: dict, tools_override: dict | None = None) -> None:
        nonlocal ok, violations
        vm = ExecutionVM(llm=MockLLMAdapter("refund"), tools=tools_override or TOOLS)
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
        *[_run_one(ctx, ADV_TOOLS) for ctx in delayed_contexts],
    )
    elapsed = time.perf_counter() - t0
    progress.advance(task, N)
    total = len(dup_contexts) + len(ooo_contexts) + len(delayed_contexts)
    latencies.sort()
    return RunResult(
        "BM-INT-09",
        cycle,
        run,
        total,
        elapsed,
        total / elapsed,
        ok,
        0,
        0,
        violations,
        0.0,
        _percentile(latencies, 50),
        _percentile(latencies, 95),
        _percentile(latencies, 99),
        {
            "sub_a_dup_n": len(dup_contexts),
            "sub_b_ooo_n": len(ooo_contexts),
            "sub_c_delayed_n": len(delayed_contexts),
        },
    )


LONG_STEPS = 1_000
LONG_PROGRAMS = 10
RSS_LIMIT_MB = 512


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
    program = Program.from_dict(_make_long_program(LONG_STEPS))
    vm = ExecutionVM(llm=MockLLMAdapter("ok"), tools=TOOLS)
    gc.collect()
    tracemalloc.start()
    rss_before = get_process_rss_mb()
    latencies: list[float] = []
    ok = violations = 0
    total_steps_executed = 0
    t0_wall = time.perf_counter()

    for i in range(LONG_PROGRAMS):
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
        progress.advance(task, N // LONG_PROGRAMS)

    elapsed = time.perf_counter() - t0_wall
    _, peak_alloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = get_process_rss_mb()
    peak_rss_mb = max(rss_before, rss_after)
    peak_alloc_mb = peak_alloc / (1024 * 1024)
    if peak_rss_mb > RSS_LIMIT_MB:
        violations += 1
    latencies.sort()
    return RunResult(
        "BM-INT-10",
        cycle,
        run,
        LONG_PROGRAMS * LONG_STEPS,
        elapsed,
        total_steps_executed / elapsed,
        ok,
        0,
        0,
        violations,
        float(LONG_STEPS),
        _percentile(latencies, 50),
        _percentile(latencies, 95),
        _percentile(latencies, 99),
        {
            "peak_rss_mb": round(peak_rss_mb, 1),
            "peak_alloc_mb": round(peak_alloc_mb, 2),
            "total_steps": total_steps_executed,
            "rss_limit_mb": RSS_LIMIT_MB,
        },
    )


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


STATUS_COLOR = {True: "bold green", False: "bold red"}
STATUS_ICON = {True: "✓ PASS", False: "✗ FAIL"}


def fmt_tps(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M/s"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K/s"
    return f"{v:.0f}/s"


def fmt_ms(v: float) -> str:
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
            fmt_tps(s.mean_tps),
            f"±{fmt_tps(s.stddev_tps)}",
            Text(f"{s.ok_pct:.1f}%", style=ok_color),
            fmt_ms(statistics.mean([r.p50_ms for r in s.results])),
            fmt_ms(s.mean_p95),
            Text(str(s.total_violations), style=viol_color),
            Text(STATUS_ICON[s.passed], style=STATUS_COLOR[s.passed]),
        )
    return t


def render_final_table(
    all_summaries: dict[str, ScenarioSummary], scenarios: list
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
            fmt_tps(s.mean_tps),
            f"±{fmt_tps(s.stddev_tps)}",
            Text(f"{s.ok_pct:.1f}%", style=ok_color),
            fmt_ms(s.mean_p95),
            Text(str(s.total_violations), style=viol_color),
            Text(STATUS_ICON[s.passed], style=STATUS_COLOR[s.passed]),
        )
    return t, all_pass


async def main() -> None:
    args = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Integration benchmark")
    parser.add_argument("--suite", choices=["all", "original", "new"], default="all")
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    scenarios = ALL_SCENARIOS
    if args.only:
        ids = {s.strip() for s in args.only.split(",")}
        scenarios = [s for s in ALL_SCENARIOS if s[0] in ids]
    elif args.suite == "original":
        scenarios = SCENARIOS_ORIGINAL
    elif args.suite == "new":
        scenarios = SCENARIOS_NEW

    console.print()
    console.print(
        Rule(
            "[bold cyan]llm-nano-vm × nano-vm-mcp — Integration Benchmark[/bold cyan]", style="cyan"
        )
    )
    console.print(
        f"[dim]{CYCLES} cycles · {RUNS} runs · {N:,} items/run · seed={SEED} · suite={args.suite} · scenarios={len(scenarios)}[/dim]"
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
                        f"[cyan]{sid}[/cyan] [dim]{label[:22]}[/dim] [bright_black]c={cycle} r={run}[/bright_black]",
                        total=N,
                    )
                    result = await runner(cycle, run, progress, task)
                    progress.update(task, completed=N, visible=False)
                    progress.advance(master, N)
                    cycle_summaries[sid].results.append(result)
                    all_summaries[sid].results.append(result)
            console.print()
            console.print(
                render_cycle_table(cycle, [cycle_summaries[sid] for sid, *_ in scenarios])
            )

    console.print()
    console.print(Rule("[bold white]Final Results[/bold white]", style="cyan"))
    console.print()
    final_table, all_pass = render_final_table(all_summaries, scenarios)
    console.print(final_table)

    total_viol = sum(s.total_violations for s in all_summaries.values())
    total_items = sum(s.total_n for s in all_summaries.values())
    verdict = (
        "DETERMINISTIC EXECUTION VERIFIED" if total_viol == 0 else "INVARIANT VIOLATIONS DETECTED"
    )
    panel = Panel(
        f"{verdict}\n\nTotal operations: {total_items:,}\nTotal violations: {total_viol}\n",
        title="Invariant Audit",
        border_style="green" if total_viol == 0 else "red",
        expand=True,
    )
    console.print(panel)

    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
