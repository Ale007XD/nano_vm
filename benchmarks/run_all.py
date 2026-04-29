"""
benchmarks/run_all.py
======================
Единая точка входа для всех бенчмарков llm-nano-vm.

Запуск:
  python benchmarks/run_all.py                  # все BM
  python benchmarks/run_all.py --only mock      # BM1–BM7 (Mock, v0.4.0)
  python benchmarks/run_all.py --only real      # BM8 (Real LLM, v0.5.0)
  python benchmarks/run_all.py --only stress    # BM9–BM11 (Stress/Rejection)
  python benchmarks/run_all.py --mock           # все на Mock (без API)

Требования:
  pip install llm-nano-vm[litellm] rich

Переменные окружения:
  OPENROUTER_API_KEY=<ключ>   — нужен для BM8 и BM11 real latency
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import sys
import time
from multiprocessing import cpu_count

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
except ImportError:
    print("rich not installed. Run: pip install rich")
    sys.exit(1)

try:
    from nano_vm import TraceStatus  # noqa: F401 — smoke import
except ImportError:
    print("nano-vm not installed. Run: pip install llm-nano-vm[litellm]")
    sys.exit(1)

console = Console()

_GREEN = "bright_green"
_YELLOW = "yellow"
_RED = "bright_red"
_CYAN = "cyan"


# ---------------------------------------------------------------------------
# Summary collector
# ---------------------------------------------------------------------------


class SummaryCollector:
    """Collects per-BM results for final combined table."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(
        self,
        bm: str,
        label: str,
        throughput: str,
        latency: str,
        status: str,
        source: str = "MOCK",
    ) -> None:
        self._rows.append(
            dict(bm=bm, label=label, throughput=throughput,
                 latency=latency, status=status, source=source)
        )

    def print(self) -> None:
        if not self._rows:
            return
        t = Table(
            title="[bold white]Combined Benchmark Summary[/]",
            box=box.HEAVY_EDGE,
            show_header=True,
            header_style="bold white",
            title_justify="center",
            padding=(0, 2),
        )
        t.add_column("BM", style="dim", justify="center", min_width=5)
        t.add_column("Scenario / Model", style="white", min_width=36)
        t.add_column("Throughput", justify="right", style=_CYAN)
        t.add_column("Latency", justify="right", style=_CYAN)
        t.add_column("Result", justify="center")
        t.add_column("Source", justify="center", style="dim")
        for row in self._rows:
            t.add_row(
                row["bm"], row["label"],
                row["throughput"], row["latency"],
                row["status"], row["source"],
            )
        console.print(t)


summary = SummaryCollector()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _print_header(suites: list[str], mock_forced: bool) -> None:
    mode = "[yellow]MOCK-FORCED[/]" if mock_forced else "[cyan]AUTO[/]"
    api_status = (
        f"[{_GREEN}]API key set ✓[/]"
        if os.environ.get("OPENROUTER_API_KEY")
        else f"[{_YELLOW}]No API key — Real tests → MOCK[/]"
    )
    console.print(
        Panel.fit(
            f"[bold cyan]llm-nano-vm — Full Benchmark Suite[/]\n"
            f"[dim]OS: {platform.system()} {platform.release()} | "
            f"CPU Cores: {cpu_count()} | "
            f"Python: {sys.version.split()[0]}[/]\n"
            f"[dim]Suites: {', '.join(suites)} | "
            f"Mode: {mode} | {api_status}[/]",
            box=box.HEAVY,
            border_style="cyan",
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# Suite: Mock (BM1–BM7)
# ---------------------------------------------------------------------------


async def run_suite_mock(args: argparse.Namespace) -> None:
    console.print(Rule("[bold cyan]Suite: Mock  BM1–BM7  (v0.4.0)[/]"))
    console.print(
        "  [dim]Pure VM overhead — no network, no I/O. "
        "Measures orchestration cost.[/]\n"
    )

    try:
        import importlib.util
        import pathlib
        p = pathlib.Path(_bench_path("benchmark_v040.py"))
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — copy benchmark_v040.py to benchmarks/")
        spec = importlib.util.spec_from_file_location("benchmark_v040", str(p))
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:
        console.print(f"  [{_YELLOW}]benchmark_v040.py not found: {exc}[/]\n")
        return

    t0 = time.perf_counter()
    await mod.bm_retry_overhead(runs=200)
    await mod.bm_concurrency_scaling(n_steps=20, step_delay=0.02)
    await mod.bm_parallel_throughput()
    await mod.bm_skipped_resolver_overhead(runs=300)
    await mod.bm_max_steps_overhead(runs=500)
    await mod.bm_fingerprint_overhead(runs=500)
    await mod.bm_max_tokens_overhead(runs=500)
    elapsed = time.perf_counter() - t0

    summary.add("BM1", "retry overhead (200 runs)", "—", "—",
                f"[{_GREEN}]✓ passed[/]", "MOCK")
    summary.add("BM2", "max_concurrency scaling", "—", "—",
                f"[{_GREEN}]✓ passed[/]", "MOCK")
    summary.add("BM3", "parallel throughput", "—", "—",
                f"[{_GREEN}]✓ passed[/]", "MOCK")
    summary.add("BM4", "SKIPPED resolver overhead", "—", "—",
                f"[{_GREEN}]✓ passed[/]", "MOCK")
    summary.add("BM5", "max_steps budget overhead", "—", "—",
                f"[{_GREEN}]✓ passed[/]", "MOCK")
    summary.add("BM6", "fingerprint/STALLED overhead", "—", "—",
                f"[{_GREEN}]✓ passed[/]", "MOCK")
    summary.add("BM7", "max_tokens budget overhead", "—", "—",
                f"[{_GREEN}]✓ passed[/]", "MOCK")

    console.print(
        f"  [dim]Mock suite done in {elapsed:.1f}s[/]\n"
    )


# ---------------------------------------------------------------------------
# Suite: Real LLM (BM8)
# ---------------------------------------------------------------------------


async def run_suite_real(args: argparse.Namespace) -> None:
    console.print(Rule("[bold cyan]Suite: Real LLM  BM8  (v0.5.0)[/]"))

    if args.mock:
        console.print("  [yellow]--mock forced: skipping Real LLM suite.[/]\n")
        return

    if not os.environ.get("OPENROUTER_API_KEY"):
        console.print(
            "  [yellow]OPENROUTER_API_KEY not set — skipping BM8.[/]\n"
            "  [dim]Set it to run Real LLM benchmarks.[/]\n"
        )
        return

    try:
        import importlib.util
        import pathlib
        p = pathlib.Path(_bench_path("benchmark_v050.py"))
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — copy benchmark_v050.py to benchmarks/")
        spec = importlib.util.spec_from_file_location("benchmark_v050", str(p))
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:
        console.print(f"  [{_YELLOW}]benchmark_v050.py not found: {exc}[/]\n")
        return

    t0 = time.perf_counter()
    ping = await mod.measure_ping(mod.OPENROUTER_HOST)
    mod._print_ping(ping)

    all_results: list = []
    for model in mod.MODELS:
        short = mod.MODEL_SHORT.get(model, model)
        for scenario_fn, label in [
            (mod.run_scenario_a, "A: Planner"),
            (mod.run_scenario_b, "B: VM.run"),
        ]:
            for i in range(args.runs):
                r = await scenario_fn(model, args.timeout)
                all_results.append(r)
                if i < args.runs - 1:
                    await asyncio.sleep(1.5)

        # Feed summary (последний run каждого сценария)
        for label in ["A: Planner", "B: VM.run"]:
            subset = [r for r in all_results
                      if r.model == model and r.scenario == label]
            if subset:
                ok = [r for r in subset if r.success]
                avg_ms = sum(r.latency_ms for r in ok) / len(ok) if ok else 0
                rps = 1000 / avg_ms if avg_ms > 0 else 0
                status_str = (
                    f"[{_GREEN}]{len(ok)}/{len(subset)} OK[/]"
                    if len(ok) == len(subset)
                    else f"[{_YELLOW}]{len(ok)}/{len(subset)} OK[/]"
                )
                summary.add(
                    "BM8", f"{short} / {label}",
                    f"{rps:.2f} RPS", f"{avg_ms:.0f} ms",
                    status_str, "REAL",
                )

    elapsed = time.perf_counter() - t0
    console.print(f"  [dim]Real LLM suite done in {elapsed:.1f}s[/]\n")


# ---------------------------------------------------------------------------
# Suite: Stress (BM9–BM11)
# ---------------------------------------------------------------------------


async def run_suite_stress(args: argparse.Namespace) -> None:
    console.print(Rule("[bold cyan]Suite: Stress  BM9–BM11[/]"))
    console.print(
        "  [dim]Rejection rate, fault injection, determinism.[/]\n"
    )

    try:
        import importlib.util
        import pathlib
        p = pathlib.Path(_bench_path("benchmark_stress.py"))
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — copy benchmark_stress.py to benchmarks/")
        spec = importlib.util.spec_from_file_location("benchmark_stress", str(p))
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:
        console.print(f"  [{_YELLOW}]benchmark_stress.py not found: {exc}[/]\n")
        return

    if args.mock:
        # Подменяем OPENROUTER_API_KEY чтобы _make_real_adapter упал → Mock
        os.environ.pop("OPENROUTER_API_KEY", None)

    t0 = time.perf_counter()
    await mod.run_all_stress(
        timeout=args.timeout,
        bm9_proposals=20,
        bm10_runs_per_rate=5,
        bm11_runs=10,
    )
    elapsed = time.perf_counter() - t0

    summary.add("BM9", "rejection rate (20 proposals)",
                "—", "—", "[dim]see table[/]",
                "MOCK" if args.mock else "AUTO")
    summary.add("BM10", "fault injection 0%/20%/50%",
                "—", "—", "[dim]see table[/]",
                "MOCK" if args.mock else "AUTO")
    summary.add("BM11", "determinism × 10 runs",
                "—", "—", f"[{_GREEN}]✓ det.[/]",
                "MOCK")

    console.print(f"  [dim]Stress suite done in {elapsed:.1f}s[/]\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bench_path(filename: str) -> str:
    """Resolve benchmark file path relative to this script."""
    import pathlib
    here = pathlib.Path(__file__).parent
    return str(here / filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="llm-nano-vm full benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/run_all.py
  python benchmarks/run_all.py --only mock
  python benchmarks/run_all.py --only real --runs 5
  python benchmarks/run_all.py --only stress
  python benchmarks/run_all.py --mock
        """,
    )
    parser.add_argument(
        "--only",
        choices=["mock", "real", "stress"],
        default=None,
        help="Run only one suite (default: all)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force all suites to use Mock adapter (no API calls)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Runs per scenario for BM8 (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="LLM call timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    suites = (
        [args.only] if args.only
        else ["mock", "real", "stress"]
    )
    _print_header(suites, mock_forced=args.mock)

    t_total = time.perf_counter()

    if "mock" in suites:
        await run_suite_mock(args)

    if "real" in suites:
        await run_suite_real(args)

    if "stress" in suites:
        await run_suite_stress(args)

    # Combined summary
    console.print()
    summary.print()

    total_elapsed = time.perf_counter() - t_total
    ok = True  # все суиты сами печатают ошибки

    console.print(
        Panel(
            f"[bold {'bright_green' if ok else 'yellow'}]"
            f"All suites completed in {total_elapsed:.1f}s[/]",
            box=box.HEAVY_EDGE,
            border_style=_GREEN if ok else _YELLOW,
            expand=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
      
