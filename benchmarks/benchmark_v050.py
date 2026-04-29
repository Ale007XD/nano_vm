"""
benchmarks/benchmark_v050.py
=============================
BM8 — боевой бенчмарк v0.5.0: реальные LLM-вызовы через OpenRouter.
Два сценария × две модели free tier.

Сценарии:
  A) Planner: intent → Program (1 LLM-вызов Planner)
  B) VM.run: готовый Program → Trace (1–3 LLM-шага)

Модели (free tier OpenRouter):
  - meta-llama/llama-3.3-70b-instruct:free
  - mistralai/mistral-7b-instruct:free

Метрики:
  - ping до api.openrouter.ai (TCP + HTTP HEAD)
  - e2e latency (wall-clock на весь вызов)
  - TTFT (time to first token) — если провайдер возвращает
  - prompt / completion / total tokens
  - стоимость ($ per call)
  - статус (SUCCESS / FAILED / timeout)

Требования:
  pip install llm-nano-vm[litellm] rich

Переменная окружения:
  OPENROUTER_API_KEY=<ключ>

Запуск:
  python benchmarks/benchmark_v050.py
  python benchmarks/benchmark_v050.py --runs 3 --timeout 30
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Any

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("rich not installed. Run: pip install rich")
    sys.exit(1)

try:
    from nano_vm import ExecutionVM, Planner, Program, TraceStatus
    from nano_vm.adapters import LiteLLMAdapter
    from nano_vm.models import Step, StepType
except ImportError:
    print("nano-vm not installed. Run: pip install llm-nano-vm[litellm]")
    sys.exit(1)

console = Console()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = [
    "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/mistralai/mistral-7b-instruct:free",
]

MODEL_SHORT = {
    "openrouter/meta-llama/llama-3.3-70b-instruct:free": "llama-3.3-70b",
    "openrouter/mistralai/mistral-7b-instruct:free": "mistral-7b",
}

OPENROUTER_HOST = "openrouter.ai"
OPENROUTER_PORT = 443

# Fixed Program для сценария B — детерминированный, 2 LLM-шага
_BENCH_PROGRAM = Program.from_dict({
    "name": "bm8_bench",
    "description": "BM8 fixed program: classify then summarize",
    "steps": [
        {
            "id": "classify",
            "type": "llm",
            "prompt": (
                "Classify this text in one word: positive, negative, or neutral.\n"
                "Text: 'The product works as expected.'\n"
                "Reply with one word only."
            ),
            "output_key": "sentiment",
        },
        {
            "id": "summarize",
            "type": "llm",
            "prompt": (
                "Write one sentence summary based on sentiment: $sentiment.\n"
                "Be concise."
            ),
            "output_key": "result",
        },
    ],
})

# Intent для сценария A (Planner)
_BENCH_INTENT = (
    "Classify a customer review as positive, negative, or neutral, "
    "then write a one-sentence summary."
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    model: str
    scenario: str          # "A: Planner" | "B: VM.run"
    success: bool
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float | None = None
    error: str = ""
    planner_attempts: int = 0  # для сценария A
    steps_executed: int = 0    # для сценария B


@dataclass
class PingResult:
    host: str
    tcp_ms: float | None = None
    http_ms: float | None = None
    error: str = ""


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------


def _tcp_ping(host: str, port: int, timeout: float = 5.0) -> float | None:
    """TCP connect latency in ms."""
    try:
        t0 = time.perf_counter()
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return (time.perf_counter() - t0) * 1000
    except OSError:
        return None


async def _http_ping(host: str, timeout: float = 10.0) -> float | None:
    """HTTP HEAD latency to https://{host}/ in ms."""
    try:
        import httpx  # optional; falls back gracefully
        async with httpx.AsyncClient(timeout=timeout) as client:
            t0 = time.perf_counter()
            await client.head(f"https://{host}/")
            return (time.perf_counter() - t0) * 1000
    except Exception:
        return None


async def measure_ping(host: str, port: int = 443) -> PingResult:
    tcp_ms = await asyncio.get_event_loop().run_in_executor(
        None, _tcp_ping, host, port
    )
    http_ms = await _http_ping(host)
    return PingResult(host=host, tcp_ms=tcp_ms, http_ms=http_ms)


# ---------------------------------------------------------------------------
# Scenario A: Planner
# ---------------------------------------------------------------------------


async def run_scenario_a(
    model: str,
    timeout: float,
) -> CallResult:
    adapter = LiteLLMAdapter(model, timeout=timeout, temperature=0.0, max_retries=1)
    planner = Planner(llm=adapter, max_retries=2)

    t0 = time.perf_counter()
    try:
        program = await planner.generate(
            _BENCH_INTENT,
            available_tools=["send_email", "save_to_db"],
            context_keys=["review_text"],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # Planner-only сценарий: токены недоступны без доп. VM-вызова.
        # Для честности замера latency = только Planner, токены не считаем.
        return CallResult(
            model=model,
            scenario="A: Planner",
            success=True,
            latency_ms=latency_ms,
            steps_executed=len(program.steps),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return CallResult(
            model=model,
            scenario="A: Planner",
            success=False,
            latency_ms=latency_ms,
            error=str(exc)[:120],
        )


# ---------------------------------------------------------------------------
# Scenario B: VM.run
# ---------------------------------------------------------------------------


async def run_scenario_b(
    model: str,
    timeout: float,
) -> CallResult:
    adapter = LiteLLMAdapter(model, timeout=timeout, temperature=0.0, max_retries=1)
    vm = ExecutionVM(llm=adapter)

    t0 = time.perf_counter()
    try:
        trace = await vm.run(_BENCH_PROGRAM)
        latency_ms = (time.perf_counter() - t0) * 1000

        total_tokens = trace.total_tokens()
        cost = trace.total_cost_usd()

        # Разбиваем на prompt/completion из шагов
        prompt_t = sum(
            s.usage.prompt_tokens for s in trace.steps if s.usage
        )
        completion_t = sum(
            s.usage.completion_tokens for s in trace.steps if s.usage
        )

        return CallResult(
            model=model,
            scenario="B: VM.run",
            success=trace.status == TraceStatus.SUCCESS,
            latency_ms=latency_ms,
            prompt_tokens=prompt_t,
            completion_tokens=completion_t,
            total_tokens=total_tokens,
            cost_usd=cost,
            steps_executed=len(trace.steps),
            error="" if trace.status == TraceStatus.SUCCESS else (trace.error or ""),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return CallResult(
            model=model,
            scenario="B: VM.run",
            success=False,
            latency_ms=latency_ms,
            error=str(exc)[:120],
        )


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------


@dataclass
class AggResult:
    model: str
    scenario: str
    runs: int
    success_count: int
    latency_avg_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_p50_ms: float
    total_tokens_avg: float
    cost_usd_total: float | None
    errors: list[str] = field(default_factory=list)


def _aggregate(results: list[CallResult]) -> AggResult:
    assert results
    ok = [r for r in results if r.success]
    latencies = sorted(r.latency_ms for r in ok) if ok else [0.0]
    p50 = latencies[len(latencies) // 2]

    cost_values = [r.cost_usd for r in ok if r.cost_usd is not None]
    cost_total = sum(cost_values) if cost_values else None

    return AggResult(
        model=results[0].model,
        scenario=results[0].scenario,
        runs=len(results),
        success_count=len(ok),
        latency_avg_ms=sum(latencies) / len(latencies) if latencies else 0,
        latency_min_ms=min(latencies) if latencies else 0,
        latency_max_ms=max(latencies) if latencies else 0,
        latency_p50_ms=p50,
        total_tokens_avg=(
            sum(r.total_tokens for r in ok) / len(ok) if ok else 0
        ),
        cost_usd_total=cost_total,
        errors=[r.error for r in results if not r.success],
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_GREEN = "bright_green"
_YELLOW = "yellow"
_RED = "bright_red"
_CYAN = "cyan"


def _latency_color(ms: float) -> str:
    if ms < 2000:
        return _GREEN
    if ms < 5000:
        return _YELLOW
    return _RED


def _fmt_latency(ms: float) -> str:
    color = _latency_color(ms)
    return f"[{color}]{ms:.0f} ms[/]"


def _fmt_status(ok: int, total: int) -> str:
    if ok == total:
        return f"[{_GREEN}]✓ {ok}/{total}[/]"
    if ok == 0:
        return f"[{_RED}]✗ {ok}/{total}[/]"
    return f"[{_YELLOW}]⚠ {ok}/{total}[/]"


def _fmt_cost(cost: float | None) -> str:
    if cost is None:
        return "[dim]—[/]"
    if cost == 0.0:
        return "[dim]$0.000000[/]"
    return f"[dim]${cost:.6f}[/]"


def _print_ping(ping: PingResult) -> None:
    tcp = f"{ping.tcp_ms:.1f} ms" if ping.tcp_ms is not None else "—"
    http = f"{ping.http_ms:.1f} ms" if ping.http_ms is not None else "—"
    color_tcp = _latency_color(ping.tcp_ms or 9999)
    color_http = _latency_color(ping.http_ms or 9999)
    console.print(
        f"  [dim]Ping → {ping.host}[/]  "
        f"TCP [{color_tcp}]{tcp}[/]  "
        f"HTTP [{color_http}]{http}[/]\n"
    )


def _build_results_table(agg_list: list[AggResult]) -> Table:
    t = Table(
        title="[bold cyan]BM8: Real LLM Benchmark (OpenRouter)[/]  [dim]v0.5.0[/]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
        padding=(0, 1),
    )
    t.add_column("Model", style="white", min_width=16)
    t.add_column("Scenario", style="white", min_width=14)
    t.add_column("Status", justify="center")
    t.add_column("Avg Latency", justify="right")
    t.add_column("Min / Max", justify="right", style="dim")
    t.add_column("p50", justify="right", style="dim")
    t.add_column("Tokens (avg)", justify="right", style=_CYAN)
    t.add_column("Cost (total)", justify="right")

    for agg in agg_list:
        short = MODEL_SHORT.get(agg.model, agg.model.split("/")[-1])
        tokens_str = (
            f"{agg.total_tokens_avg:.0f}"
            if agg.total_tokens_avg > 0 else "[dim]—[/]"
        )
        t.add_row(
            short,
            agg.scenario,
            _fmt_status(agg.success_count, agg.runs),
            _fmt_latency(agg.latency_avg_ms),
            f"{agg.latency_min_ms:.0f} / {agg.latency_max_ms:.0f} ms",
            _fmt_latency(agg.latency_p50_ms),
            tokens_str,
            _fmt_cost(agg.cost_usd_total),
        )

    return t


def _build_comparison_table(agg_list: list[AggResult]) -> Table:
    """Side-by-side latency comparison between models per scenario."""
    t = Table(
        title="[bold cyan]Model Latency Comparison[/]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        title_justify="left",
        padding=(0, 1),
    )
    t.add_column("Scenario", style="white", min_width=14)

    models_seen: list[str] = []
    for agg in agg_list:
        short = MODEL_SHORT.get(agg.model, agg.model.split("/")[-1])
        if short not in models_seen:
            models_seen.append(short)
            t.add_column(short, justify="right")

    t.add_column("Faster", justify="center")

    # group by scenario
    scenarios: dict[str, dict[str, AggResult]] = {}
    for agg in agg_list:
        short = MODEL_SHORT.get(agg.model, agg.model.split("/")[-1])
        scenarios.setdefault(agg.scenario, {})[short] = agg

    for scenario, by_model in scenarios.items():
        latencies = {m: a.latency_avg_ms for m, a in by_model.items()}
        fastest = min(latencies, key=latencies.get)  # type: ignore[arg-type]

        row_vals: list[Any] = [scenario]
        for m in models_seen:
            if m in latencies:
                color = _GREEN if m == fastest else _YELLOW
                row_vals.append(f"[{color}]{latencies[m]:.0f} ms[/]")
            else:
                row_vals.append("[dim]—[/]")
        row_vals.append(f"[{_GREEN}]{fastest}[/]")
        t.add_row(*row_vals)

    return t


def _print_errors(agg_list: list[AggResult]) -> None:
    errors_found = [(a, e) for a in agg_list for e in a.errors]
    if not errors_found:
        return
    console.print("\n  [bold red]Errors:[/]")
    for agg, err in errors_found:
        short = MODEL_SHORT.get(agg.model, agg.model.split("/")[-1])
        console.print(f"  [{_RED}]{short} / {agg.scenario}:[/] [dim]{err}[/]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(runs: int, timeout: float) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        console.print(
            Panel(
                "[bold red]OPENROUTER_API_KEY not set.[/]\n"
                "Export it: [cyan]export OPENROUTER_API_KEY=sk-or-...[/]",
                box=box.HEAVY_EDGE,
                border_style="red",
                expand=False,
            )
        )
        sys.exit(1)

    os.environ.setdefault("OPENROUTER_API_KEY", api_key)

    console.print(
        Panel.fit(
            f"[bold cyan]llm-nano-vm v0.5.0 — Real LLM Benchmark (BM8)[/]\n"
            f"[dim]OS: {platform.system()} {platform.release()} | "
            f"CPU Cores: {cpu_count()} | "
            f"Python: {sys.version.split()[0]}[/]\n"
            f"[dim]Runs per scenario: {runs} | "
            f"Timeout: {timeout}s | "
            f"Models: {len(MODELS)}[/]",
            box=box.HEAVY,
            border_style="cyan",
        )
    )
    console.print()

    # Ping
    console.print("  [dim]Measuring network latency...[/]")
    ping = await measure_ping(OPENROUTER_HOST)
    _print_ping(ping)

    if ping.tcp_ms is None:
        console.print(
            f"  [yellow]⚠ TCP ping to {OPENROUTER_HOST} failed — "
            f"network issues possible[/]\n"
        )

    # Run scenarios
    all_results: list[CallResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[dim]{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as progress:
        total_calls = len(MODELS) * 2 * runs
        task = progress.add_task("Running BM8...", total=total_calls)

        for model in MODELS:
            short = MODEL_SHORT.get(model, model)

            # Scenario A: Planner
            for i in range(runs):
                progress.update(
                    task,
                    description=f"BM8 [{short}] A: Planner  run {i + 1}/{runs}",
                )
                result = await run_scenario_a(model, timeout)
                all_results.append(result)
                progress.advance(task)
                # Пауза между вызовами — free tier rate limits
                if i < runs - 1:
                    await asyncio.sleep(1.5)

            # Scenario B: VM.run
            for i in range(runs):
                progress.update(
                    task,
                    description=f"BM8 [{short}] B: VM.run  run {i + 1}/{runs}",
                )
                result = await run_scenario_b(model, timeout)
                all_results.append(result)
                progress.advance(task)
                if i < runs - 1:
                    await asyncio.sleep(1.5)

    # Aggregate
    agg_list: list[AggResult] = []
    for model in MODELS:
        for scenario_fn, scenario_label in [
            ("A: Planner", "A: Planner"),
            ("B: VM.run", "B: VM.run"),
        ]:
            subset = [
                r for r in all_results
                if r.model == model and r.scenario == scenario_label
            ]
            if subset:
                agg_list.append(_aggregate(subset))

    # Print tables
    console.print(_build_results_table(agg_list))
    console.print()
    console.print(_build_comparison_table(agg_list))

    _print_errors(agg_list)

    # Footer note
    console.print(
        "\n  [dim]Latency = wall-clock e2e (DNS + TCP + TLS + LLM inference + response).[/]"
    )
    console.print(
        "  [dim]Scenario A: Planner (1 LLM call). "
        "Scenario B: VM.run (2 LLM steps, fixed program).[/]"
    )
    console.print(
        "  [dim]free tier models may have cold-start latency spikes — "
        "p50 is more representative than avg.[/]\n"
    )

    success_total = sum(1 for r in all_results if r.success)
    total = len(all_results)
    if success_total == total:
        console.print(
            Panel(
                f"[bold bright_green]Benchmark Completed Successfully  "
                f"({success_total}/{total} calls)[/]",
                box=box.HEAVY_EDGE,
                border_style="bright_green",
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                f"[bold yellow]Benchmark Completed with Errors  "
                f"({success_total}/{total} calls succeeded)[/]",
                box=box.HEAVY_EDGE,
                border_style="yellow",
                expand=False,
            )
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="llm-nano-vm v0.5.0 real LLM benchmark (BM8)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Runs per scenario per model (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="LLM call timeout in seconds (default: 60)",
    )
    args = parser.parse_args()
    asyncio.run(main(runs=args.runs, timeout=args.timeout))
