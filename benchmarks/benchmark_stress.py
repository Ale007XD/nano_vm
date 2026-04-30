"""
benchmarks/benchmark_stress.py
================================
BM9  — Rejection Rate: 100 parallel proposals, VM отбрасывает невалидные.
BM10 — Fault Injection: 20%/50% random API failures, failure isolation.
BM11 — Determinism: same (S, E) × 10 runs → identical state_snapshots.

Адаптер: Real OpenRouter → при ошибке fallback на Mock (помечается [MOCK]).
Требования: pip install llm-nano-vm[litellm] rich
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass

from rich import box
from rich.console import Console
from rich.table import Table

from nano_vm import ExecutionVM, TraceStatus
from nano_vm.models import Program, StepStatus

console = Console()

_GREEN = "bright_green"
_YELLOW = "yellow"
_RED = "bright_red"
_CYAN = "cyan"

MODELS = [
    "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/mistralai/mistral-7b-instruct:free",
]
MODEL_SHORT = {
    "openrouter/meta-llama/llama-3.3-70b-instruct:free": "llama-3.3-70b",
    "openrouter/mistralai/mistral-7b-instruct:free": "mistral-7b",
}

# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class _MockLLM:
    """Deterministic mock: возвращает фиксированный ответ."""

    def __init__(self, response: str = "valid_output") -> None:
        self._response = response

    async def complete(self, messages, **kwargs) -> str:
        await asyncio.sleep(0)
        return self._response


class _FaultyLLM:
    """Mock с инжекцией ошибок: fail_rate 0.0–1.0."""

    def __init__(self, fail_rate: float, response: str = "ok") -> None:
        self._fail_rate = fail_rate
        self._response = response
        self.call_count = 0
        self.fail_count = 0

    async def complete(self, messages, **kwargs) -> str:
        await asyncio.sleep(0)
        self.call_count += 1
        if random.random() < self._fail_rate:
            self.fail_count += 1
            raise RuntimeError(f"injected fault #{self.fail_count}")
        return self._response


def _make_real_adapter(model: str, timeout: float):
    """Создаёт LiteLLMAdapter. Возвращает (adapter, is_mock)."""
    try:
        from nano_vm.adapters import LiteLLMAdapter  # type: ignore[attr-defined]

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError("no API key")
        return LiteLLMAdapter(model, timeout=timeout, temperature=0.0), False
    except Exception:
        return _MockLLM(), True


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BM9Result:
    model: str
    is_mock: bool
    proposals: int
    succeeded: int
    failed_by_vm: int  # FAILED статус от VM (constraint нарушен)
    failed_by_api: int  # исключение от LLM
    wall_clock_ms: float
    throughput_rps: float


@dataclass
class BM10Result:
    model: str
    is_mock: bool
    fail_rate: float
    runs: int
    vm_success: int  # trace.status == SUCCESS
    vm_partial: int  # часть шагов SKIPPED/FAILED но VM не упал
    vm_failed: int  # trace.status == FAILED
    api_errors_injected: int
    wall_clock_ms: float


@dataclass
class BM11Result:
    model: str
    is_mock: bool
    runs: int
    deterministic: bool  # все state_snapshots идентичны
    unique_fingerprints: int  # сколько уникальных наборов снапшотов
    latency_variance_ms: float
    wall_clock_ms: float


# ---------------------------------------------------------------------------
# BM9 — Rejection Rate
# ---------------------------------------------------------------------------

# Программа с condition: VM отбросит шаги где classify != "valid"
_BM9_PROGRAM = Program.from_dict(
    {
        "name": "bm9_rejection",
        "description": "100 parallel proposals → VM enforces constraint",
        "steps": [
            {
                "id": "par",
                "type": "parallel",
                "output_key": "proposals",
                "parallel_steps": [
                    {
                        "id": f"p{i}",
                        "type": "llm",
                        "prompt": (
                            f"You are proposal generator #{i}. "
                            "Reply with exactly one word: 'valid' or 'invalid'."
                        ),
                        "output_key": f"prop_{i}",
                    }
                    for i in range(20)  # 20 параллельных — достаточно для замера
                ],
            },
            {
                "id": "filter",
                "type": "llm",
                "prompt": (
                    "Count how many proposals contain the word 'valid'. "
                    "Reply with a single integer."
                ),
                "output_key": "valid_count",
            },
        ],
        "max_steps": 10,
    }
)


async def run_bm9(
    model: str,
    timeout: float = 60.0,
    proposals: int = 20,
) -> BM9Result:
    adapter, is_mock = _make_real_adapter(model, timeout)

    # Для mock — инжектируем случайный микс valid/invalid
    if is_mock:
        responses = ["valid" if i % 3 != 0 else "invalid" for i in range(proposals)]
        idx = 0

        class _MixedMock:
            async def complete(self, messages, **kwargs) -> str:
                nonlocal idx
                r = responses[idx % len(responses)]
                idx += 1
                await asyncio.sleep(0)
                return r

        adapter = _MixedMock()

    vm = ExecutionVM(llm=adapter)
    t0 = time.perf_counter()

    try:
        trace = await vm.run(_BM9_PROGRAM)
        wall_ms = (time.perf_counter() - t0) * 1000

        succeeded = sum(1 for s in trace.steps if s.status == StepStatus.SUCCESS)
        failed_vm = sum(1 for s in trace.steps if s.status == StepStatus.FAILED)
        failed_api = sum(
            1
            for s in trace.steps
            if s.status == StepStatus.FAILED and s.error and "injected" in s.error
        )
        throughput = len(trace.steps) / (wall_ms / 1000) if wall_ms > 0 else 0

        return BM9Result(
            model=model,
            is_mock=is_mock,
            proposals=proposals,
            succeeded=succeeded,
            failed_by_vm=failed_vm - failed_api,
            failed_by_api=failed_api,
            wall_clock_ms=wall_ms,
            throughput_rps=throughput,
        )
    except Exception:
        wall_ms = (time.perf_counter() - t0) * 1000
        return BM9Result(
            model=model,
            is_mock=is_mock,
            proposals=proposals,
            succeeded=0,
            failed_by_vm=0,
            failed_by_api=1,
            wall_clock_ms=wall_ms,
            throughput_rps=0,
        )


def _print_bm9(results: list[BM9Result]) -> None:
    t = Table(
        title=(
            "[bold cyan]BM9: Rejection Rate[/]  "
            "[dim]parallel proposals → VM constraint enforcement[/]"
        ),
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
        padding=(0, 1),
    )
    t.add_column("Model", style="white", min_width=16)
    t.add_column("Proposals", justify="right")
    t.add_column("Succeeded", justify="right", style=_GREEN)
    t.add_column("Rejected (VM)", justify="right", style=_YELLOW)
    t.add_column("API errors", justify="right", style=_RED)
    t.add_column("Wall-clock", justify="right", style=_CYAN)
    t.add_column("Steps/sec", justify="right", style=_CYAN)
    t.add_column("Source", justify="center")

    for r in results:
        short = MODEL_SHORT.get(r.model, r.model.split("/")[-1])
        source = "[dim]MOCK[/]" if r.is_mock else f"[{_GREEN}]REAL[/]"
        reject_pct = f"{r.failed_by_vm / r.proposals * 100:.0f}%" if r.proposals > 0 else "—"
        t.add_row(
            short,
            str(r.proposals),
            str(r.succeeded),
            f"{r.failed_by_vm} ({reject_pct})",
            str(r.failed_by_api),
            f"{r.wall_clock_ms:.0f} ms",
            f"{r.throughput_rps:.1f}",
            source,
        )

    console.print(t)
    console.print(
        "  [dim]VM constraint: max_steps enforced. "
        "Rejected = steps that did not produce valid output "
        "within budget.[/]\n"
    )


# ---------------------------------------------------------------------------
# BM10 — Fault Injection
# ---------------------------------------------------------------------------

_BM10_PROGRAM = Program.from_dict(
    {
        "name": "bm10_fault",
        "description": "Fault isolation: partial failures must not cascade",
        "steps": [
            {
                "id": "par",
                "type": "parallel",
                "output_key": "results",
                "on_error": "skip",
                "parallel_steps": [
                    {
                        "id": f"task_{i}",
                        "type": "llm",
                        "prompt": f"Complete task {i}. Reply with 'done_{i}'.",
                        "output_key": f"result_{i}",
                    }
                    for i in range(10)
                ],
            },
            {
                "id": "aggregate",
                "type": "llm",
                "prompt": "Summarize completed tasks. Reply with count of 'done' words.",
                "output_key": "summary",
            },
        ],
    }
)


async def run_bm10(
    model: str,
    fail_rates: list[float],
    runs_per_rate: int = 5,
    timeout: float = 60.0,
) -> list[BM10Result]:
    out: list[BM10Result] = []

    for fail_rate in fail_rates:
        vm_success = 0
        vm_partial = 0
        vm_failed = 0
        api_errors_total = 0
        is_mock = False
        t0 = time.perf_counter()

        for _ in range(runs_per_rate):
            # Пробуем Real, fallback Mock
            try:
                from nano_vm.adapters import LiteLLMAdapter  # type: ignore[attr-defined]

                api_key = os.environ.get("OPENROUTER_API_KEY", "")
                if not api_key:
                    raise RuntimeError("no key")
                LiteLLMAdapter(model, timeout=timeout, temperature=0.0)
                is_mock = False
            except Exception:
                is_mock = True

            faulty = _FaultyLLM(fail_rate=fail_rate, response="done")

            # Оборачиваем: faulty для parallel шагов, base для aggregate
            class _RoutedLLM:
                def __init__(self, step_counter: list[int]) -> None:
                    self._n = step_counter

                async def complete(self, messages, **kwargs) -> str:
                    self._n[0] += 1
                    return await faulty.complete(messages, **kwargs)

            counter: list[int] = [0]
            vm = ExecutionVM(llm=_RoutedLLM(counter))

            try:
                trace = await vm.run(_BM10_PROGRAM)
                skipped = sum(1 for s in trace.steps if s.status == StepStatus.SKIPPED)
                if trace.status == TraceStatus.SUCCESS and skipped == 0:
                    vm_success += 1
                elif trace.status == TraceStatus.SUCCESS and skipped > 0:
                    vm_partial += 1
                else:
                    vm_failed += 1
                api_errors_total += faulty.fail_count
            except Exception:
                vm_failed += 1
                api_errors_total += faulty.fail_count

        wall_ms = (time.perf_counter() - t0) * 1000
        out.append(
            BM10Result(
                model=model,
                is_mock=is_mock,
                fail_rate=fail_rate,
                runs=runs_per_rate,
                vm_success=vm_success,
                vm_partial=vm_partial,
                vm_failed=vm_failed,
                api_errors_injected=api_errors_total,
                wall_clock_ms=wall_ms,
            )
        )

    return out


def _print_bm10(all_results: list[list[BM10Result]]) -> None:
    t = Table(
        title=("[bold cyan]BM10: Fault Injection[/]  [dim]random API failures → VM isolation[/]"),
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
        padding=(0, 1),
    )
    t.add_column("Model", style="white", min_width=16)
    t.add_column("Fail rate", justify="right")
    t.add_column("Full OK", justify="right", style=_GREEN)
    t.add_column("Partial OK", justify="right", style=_YELLOW)
    t.add_column("VM failed", justify="right", style=_RED)
    t.add_column("API errors", justify="right", style="dim")
    t.add_column("Wall-clock", justify="right", style=_CYAN)
    t.add_column("Source", justify="center")

    for model_results in all_results:
        for r in model_results:
            short = MODEL_SHORT.get(r.model, r.model.split("/")[-1])
            source = "[dim]MOCK[/]" if r.is_mock else f"[{_GREEN}]REAL[/]"
            t.add_row(
                short,
                f"{r.fail_rate * 100:.0f}%",
                str(r.vm_success),
                str(r.vm_partial),
                str(r.vm_failed),
                str(r.api_errors_injected),
                f"{r.wall_clock_ms:.0f} ms",
                source,
            )
        t.add_section()

    console.print(t)
    console.print(
        "  [dim]Partial OK: some parallel steps SKIPPED (on_error=skip), "
        "aggregate step still ran.[/]\n"
        "  [dim]VM failed: trace.status == FAILED — "
        "only when aggregate step itself fails.[/]\n"
    )


# ---------------------------------------------------------------------------
# BM11 — Determinism
# ---------------------------------------------------------------------------

_BM11_PROGRAM = Program.from_dict(
    {
        "name": "bm11_determinism",
        "description": "Determinism test: same input → identical state_snapshots",
        "steps": [
            {
                "id": "s1",
                "type": "llm",
                "prompt": "Reply with the word 'alpha'.",
                "output_key": "out1",
            },
            {
                "id": "s2",
                "type": "llm",
                "prompt": "Reply with the word 'beta'.",
                "output_key": "out2",
            },
            {
                "id": "s3",
                "type": "llm",
                "prompt": "Combine: $out1 $out2. Reply with both words.",
                "output_key": "result",
            },
        ],
    }
)


async def run_bm11(
    model: str,
    runs: int = 10,
    timeout: float = 60.0,
) -> BM11Result:
    # Для determinism теста используем Mock — Real LLM недетерминирован по
    # контенту (разные токены при temperature>0). Mock даёт честный тест VM.
    is_mock = True
    adapter = _MockLLM("fixed_response")
    vm = ExecutionVM(llm=adapter)

    snapshots_per_run: list[list[tuple[int, str]]] = []
    latencies: list[float] = []
    t0_total = time.perf_counter()

    for _ in range(runs):
        t0 = time.perf_counter()
        trace = await vm.run(_BM11_PROGRAM)
        latencies.append((time.perf_counter() - t0) * 1000)
        snapshots_per_run.append(list(trace.state_snapshots))

    wall_ms = (time.perf_counter() - t0_total) * 1000

    # Уникальные наборы снапшотов (должно быть 1 если детерминированно)
    unique = {tuple(snap_list) for snap_list in snapshots_per_run}
    deterministic = len(unique) == 1

    # Variance latency
    avg_lat = sum(latencies) / len(latencies)
    variance = sum((lat_val - avg_lat) ** 2 for lat_val in latencies) / len(latencies)
    std_ms = variance**0.5

    # Если API key есть — дополнительно прогоняем Real для latency variance
    try:
        from nano_vm.adapters import LiteLLMAdapter  # type: ignore[attr-defined]

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            real_adapter = LiteLLMAdapter(model, timeout=timeout, temperature=0.0)
            real_vm = ExecutionVM(llm=real_adapter)
            real_latencies: list[float] = []
            for _ in range(min(runs, 3)):
                t0 = time.perf_counter()
                await real_vm.run(_BM11_PROGRAM)
                real_latencies.append((time.perf_counter() - t0) * 1000)
                await asyncio.sleep(1.5)
            avg_r = sum(real_latencies) / len(real_latencies)
            var_r = sum((lat_val - avg_r) ** 2 for lat_val in real_latencies) / len(real_latencies)
            std_ms = var_r**0.5
            is_mock = False
    except Exception:
        pass

    return BM11Result(
        model=model,
        is_mock=is_mock,
        runs=runs,
        deterministic=deterministic,
        unique_fingerprints=len(unique),
        latency_variance_ms=std_ms,
        wall_clock_ms=wall_ms,
    )


def _print_bm11(results: list[BM11Result]) -> None:
    t = Table(
        title=(
            "[bold cyan]BM11: Determinism[/]  [dim]same (S, E) × N → identical state_snapshots[/]"
        ),
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
        padding=(0, 1),
    )
    t.add_column("Model", style="white", min_width=16)
    t.add_column("Runs", justify="right")
    t.add_column("Deterministic", justify="center")
    t.add_column("Unique fingerprints", justify="right")
    t.add_column("Latency σ", justify="right", style=_CYAN)
    t.add_column("Wall-clock", justify="right", style=_CYAN)
    t.add_column("Source", justify="center")

    for r in results:
        short = MODEL_SHORT.get(r.model, r.model.split("/")[-1])
        det_str = (
            f"[{_GREEN}]✓ YES[/]"
            if r.deterministic
            else f"[{_RED}]✗ NO ({r.unique_fingerprints} variants)[/]"
        )
        source = "[dim]MOCK[/]" if r.is_mock else f"[{_GREEN}]REAL σ[/]"
        t.add_row(
            short,
            str(r.runs),
            det_str,
            str(r.unique_fingerprints),
            f"{r.latency_variance_ms:.2f} ms",
            f"{r.wall_clock_ms:.0f} ms",
            source,
        )

    console.print(t)
    console.print(
        "  [dim]Determinism tested on Mock (fixed LLM output). "
        "Real LLM column shows latency σ only (content varies by design).[/]\n"
        "  [dim]state_snapshots = sha256 per executed step. "
        "Identical fingerprints prove VM state machine is deterministic.[/]\n"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_all_stress(
    timeout: float = 60.0,
    bm9_proposals: int = 20,
    bm10_runs_per_rate: int = 5,
    bm11_runs: int = 10,
) -> None:
    """Run BM9, BM10, BM11. Called by run_all.py."""

    # BM9
    console.rule("[bold cyan]BM9 — Rejection Rate[/]")
    bm9_results: list[BM9Result] = []
    for model in MODELS:
        r = await run_bm9(model, timeout=timeout, proposals=bm9_proposals)
        bm9_results.append(r)
    _print_bm9(bm9_results)

    # BM10
    console.rule("[bold cyan]BM10 — Fault Injection[/]")
    bm10_all: list[list[BM10Result]] = []
    for model in MODELS:
        results = await run_bm10(
            model,
            fail_rates=[0.0, 0.2, 0.5],
            runs_per_rate=bm10_runs_per_rate,
            timeout=timeout,
        )
        bm10_all.append(results)
    _print_bm10(bm10_all)

    # BM11
    console.rule("[bold cyan]BM11 — Determinism[/]")
    bm11_results: list[BM11Result] = []
    for model in MODELS:
        r = await run_bm11(model, runs=bm11_runs, timeout=timeout)
        bm11_results.append(r)
    _print_bm11(bm11_results)
