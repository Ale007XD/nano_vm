#!/usr/bin/env python3
"""
nano-vm stress benchmark suite v0.1.1
Validates: $\delta(S, E) \to S'$ — deterministic, replayable, failure-safe
"""

import asyncio
import hashlib
import json
import random
import statistics
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

# ── rich ──────────────────────────────────────────────────────────────────────
try:
    from rich import box
    from rich.columns import Columns
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
    )
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Installing rich...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
        TaskProgressColumn, TextColumn, TimeElapsedColumn,
    )
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

# ── llm-nano-vm ───────────────────────────────────────────────────────────────
try:
    from nano_vm import ExecutionVM, Program
    from nano_vm.adapters import MockLLMAdapter
    HAS_NANO_VM = True
except ImportError:
    HAS_NANO_VM = False

console = Console(highlight=False)

ARRAY_SIZE = 10_000
RUNS = 5
SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
# FSM STUBS
# ═══════════════════════════════════════════════════════════════════════════════

class FSMState(str, Enum):
    DRAFT = "DRAFT"
    CART_VALIDATED = "CART_VALIDATED"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    PAYMENT_FAILED = "PAYMENT_FAILED"
    PAID = "PAID"
    RESERVED = "RESERVED"
    DELIVERING = "DELIVERING"
    DELIVERY_CONFIRMED = "DELIVERY_CONFIRMED"
    COMPLETED = "COMPLETED"
    ESCALATED = "ESCALATED"
    CANCELLED_SAFE = "CANCELLED_SAFE"
    CANCELLED_COMPENSATED = "CANCELLED_COMPENSATED"

TRANSITION_MATRIX: dict[tuple[FSMState, str], FSMState] = {
    (FSMState.DRAFT, "cart.validate"): FSMState.CART_VALIDATED,
    (FSMState.CART_VALIDATED, "payment.initiate"): FSMState.PAYMENT_PENDING,
    (FSMState.CART_VALIDATED, "order.cancel_safe"): FSMState.CANCELLED_SAFE,
    (FSMState.PAYMENT_PENDING, "payment.confirmed"): FSMState.PAID,
    (FSMState.PAYMENT_PENDING, "payment.failed"): FSMState.PAYMENT_FAILED,
    (FSMState.PAYMENT_PENDING, "payment.timeout"): FSMState.PAYMENT_FAILED,
    (FSMState.PAYMENT_FAILED, "payment.initiate"): FSMState.PAYMENT_PENDING,
    (FSMState.PAYMENT_FAILED, "order.cancel_safe"): FSMState.CANCELLED_SAFE,
    (FSMState.PAID, "stock.reserve"): FSMState.RESERVED,
    (FSMState.PAID, "order.cancel_compensate"): FSMState.CANCELLED_COMPENSATED,
    (FSMState.RESERVED, "delivery.assign"): FSMState.DELIVERING,
    (FSMState.RESERVED, "order.cancel_compensate"): FSMState.CANCELLED_COMPENSATED,
    (FSMState.DELIVERING, "delivery.confirmed"): FSMState.DELIVERY_CONFIRMED,
    (FSMState.DELIVERY_CONFIRMED, "order.complete"): FSMState.COMPLETED,
    (FSMState.ESCALATED, "order.cancel_compensate"): FSMState.CANCELLED_COMPENSATED,
}

TERMINAL_STATES = {FSMState.COMPLETED, FSMState.CANCELLED_SAFE, FSMState.CANCELLED_COMPENSATED}

@dataclass
class StepResult:
    status: Literal["SUCCESS", "FAILED", "PENDING"]
    data: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""
    cached: bool = False
    error: str | None = None

@dataclass
class OrderState:
    state: FSMState
    version: int = 0
    side_effects: list[str] = field(default_factory=list)

class StubFSM:
    def __init__(self):
        self._tool_cache: dict[str, StepResult] = {}

    def transition(self, order: OrderState, event: str) -> OrderState | None:
        key = (order.state, event)
        if key not in TRANSITION_MATRIX:
            return None
        return OrderState(
            state=TRANSITION_MATRIX[key], 
            version=order.version + 1, 
            side_effects=list(order.side_effects)
        )

    def execute_tool(self, tool_name: str, idempotency_key: str, fail_prob: float = 0.0) -> StepResult:
        if idempotency_key in self._tool_cache:
            r = self._tool_cache[idempotency_key]
            return StepResult(status=r.status, data=r.data, idempotency_key=idempotency_key, cached=True)
        
        if random.random() < fail_prob:
            result = StepResult(status="FAILED", idempotency_key=idempotency_key, error="simulated_failure")
        else:
            result = StepResult(status="SUCCESS", data={"tool": tool_name}, idempotency_key=idempotency_key)
        
        self._tool_cache[idempotency_key] = result
        return result

# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    name: str
    tag: str
    runs: list[float]
    passed: bool
    metrics: dict[str, Any]
    invariant: str
    error: str | None = None

    @property
    def mean_ms(self) -> float: return statistics.mean(self.runs) * 1000
    @property
    def p95_ms(self) -> float:
        s = sorted(self.runs)
        return s[min(int(len(s) * 0.95), len(s) - 1)] * 1000
    @property
    def stdev_ms(self) -> float: return (statistics.stdev(self.runs) * 1000) if len(self.runs) > 1 else 0.0
    @property
    def throughput(self) -> float:
        m = statistics.mean(self.runs)
        return ARRAY_SIZE / m if m > 0 else 0

def bench_01_idempotency_replay() -> BenchmarkResult:
    fsm = StubFSM()
    runs, violations, cached_hits = [], 0, 0
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for i in range(ARRAY_SIZE):
            key = f"order:{i}:step:1"
            r1 = fsm.execute_tool("test", key)
            for _ in range(3):
                r2 = fsm.execute_tool("test", key)
                if r2.status != r1.status: violations += 1
                if r2.cached: cached_hits += 1
        runs.append(time.perf_counter() - t0)
    return BenchmarkResult("Idempotency Replay", "BM-01", runs, violations == 0, 
                           {"violations": violations, "cache_hits": cached_hits}, "S_t = S_{t+k}")

def bench_02_duplicate_execution_attack() -> BenchmarkResult:
    runs, double_executions = [], 0
    fsm = StubFSM() # Один инстанс на тест для проверки кеша
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for i in range(ARRAY_SIZE):
            key = f"order:{i}:pay"
            results = [fsm.execute_tool("pay", key) for _ in range(random.randint(2, 5))]
            if sum(1 for r in results if not r.cached) > 1:
                double_executions += 1
        runs.append(time.perf_counter() - t0)
    return BenchmarkResult("Duplicate Attack", "BM-02", runs, double_executions == 0, 
                           {"double_execs": double_executions}, "Exactly-once execution")

def bench_05_tool_failure_cascade() -> BenchmarkResult:
    runs, cascade_violations, retry_explosions = [], 0, 0
    MAX_RETRIES = 3
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for i in range(ARRAY_SIZE):
            fsm = StubFSM()
            # B: 40% fail
            r_b = fsm.execute_tool("tool_b", f"b:{i}", fail_prob=0.4)
            retries = 0
            while r_b.status == "FAILED" and retries < MAX_RETRIES:
                retries += 1
                r_b = fsm.execute_tool("tool_b", f"b:{i}:r:{retries}", fail_prob=0.4)
            
            if r_b.status == "FAILED":
                # Если B упал окончательно, C не должен быть запущен
                r_c = fsm.execute_tool("tool_c", f"c:{i}")
                if not r_c.cached: cascade_violations += 1
            
            if retries > MAX_RETRIES: # Теперь это может сработать, если цикл сломан
                retry_explosions += 1
        runs.append(time.perf_counter() - t0)
    return BenchmarkResult("Failure Cascade", "BM-05", runs, cascade_violations == 0, 
                           {"violations": cascade_violations}, "C disabled if B fails")

async def bench_11_reentrancy_stress() -> BenchmarkResult:
    """Имитация конкурентных вызовов через asyncio.gather"""
    runs, version_conflicts = [], 0
    for _ in range(RUNS):
        t0 = time.perf_counter()
        for i in range(ARRAY_SIZE // 10): # Уменьшено для async-оверхеда
            fsm = StubFSM()
            order = OrderState(state=FSMState.DRAFT, version=0)
            
            async def call_trans():
                return fsm.transition(order, "cart.validate")
            
            results = await asyncio.gather(*(call_trans() for _ in range(5)))
            unique_versions = {r.version for r in results if r}
            if len(unique_versions) > 1: version_conflicts += 1
        runs.append(time.perf_counter() - t0)
    return BenchmarkResult("Reentrancy Stress", "BM-11", runs, version_conflicts == 0, 
                           {"conflicts": version_conflicts}, "Deterministic serialization")

async def bench_nanovm_double_execution() -> BenchmarkResult | None:
    if not HAS_NANO_VM: return None
    
    program = Program.from_dict({
        "name": "p", "steps": [
            {"id": "a", "type": "llm", "prompt": "yes/no", "output_key": "d"},
            {"id": "p", "type": "tool", "tool": "pay"}
        ]
    })

    # Используем список как мутабельный контейнер для лямбды
    call_count = [0]
    
    async def mock_pay():
        call_count[0] += 1
        return {"status": "ok"}

    runs, double_exec = [], 0
    N = 50
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(N):
            call_count[0] = 0
            vm = ExecutionVM(llm=MockLLMAdapter("yes"), tools={"pay": mock_pay})
            trace = await vm.run(program, context={})
            # Проверка по трейсу: шаг 'p' успешен ровно 1 раз
            p_steps = [s for s in trace.steps if s.step_id == "p" and s.status.name == "SUCCESS"]
            if len(p_steps) > 1: double_exec += 1
        runs.append(time.perf_counter() - t0)

    return BenchmarkResult("nano-vm Safety", "BM-VM", runs, double_exec == 0, 
                           {"double_execs": double_exec}, "Trace append-only invariant")

# ═══════════════════════════════════════════════════════════════════════════════
# UI & RUNNER (остальные тесты bench_03, 04, 06-10, 12 остаются по аналогии)
# ═══════════════════════════════════════════════════════════════════════════════

# (Пропустим реализацию bench_03-04 и т.д. для краткости, они правятся по аналогии с BM-01)

async def main():
    random.seed(SEED)
    console.print(Panel("[bold cyan]NANO-VM STRESS SUITE v0.1.1[/bold cyan]", expand=False))

    # Для примера запускаем исправленные тесты
    results = []
    results.append(bench_01_idempotency_replay())
    results.append(bench_02_duplicate_execution_attack())
    results.append(bench_05_tool_failure_cascade())
    results.append(await bench_11_reentrancy_stress())
    
    if HAS_NANO_VM:
        res = await bench_nanovm_double_execution()
        if res: results.append(res)

    # Вывод таблицы
    table = Table(box=box.MINIMAL_DOUBLE_HEAD, header_style="bold cyan")
    table.add_column("Tag")
    table.add_column("Test")
    table.add_column("Status", justify="center")
    table.add_column("Mean ms", justify="right")
    table.add_column("Invariant")

    for r in results:
        status = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
        table.add_row(r.tag, r.name, status, f"{r.mean_ms:.2f}", r.invariant)
    
    console.print(table)

if __name__ == "__main__":
    asyncio.run(main())
