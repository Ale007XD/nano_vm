"""
benchmarks/benchmark_v030.py
=============================
Бенчмарки v0.3.0: retry overhead, max_concurrency scaling, semaphore cap.

Запуск:
    python benchmarks/benchmark_v030.py

Выводит таблицу результатов и сравнение с baseline (без ограничений).
"""

from __future__ import annotations

import asyncio
import platform
import sys
import time
from multiprocessing import cpu_count

try:
    from nano_vm import ExecutionVM, Program, TraceStatus
    from nano_vm.models import OnError, Step, StepType
except ImportError:
    print("❌ nano-vm not installed. Run: pip install llm-nano-vm")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Shared mock infrastructure
# ---------------------------------------------------------------------------


class MockLLM:
    async def complete(self, messages):
        await asyncio.sleep(0)  # yield to event loop, no I/O
        return "ok", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


async def fast_tool(**_) -> str:
    await asyncio.sleep(0)
    return "done"


async def slow_tool(delay: float = 0.01, **_) -> str:
    """Имитирует IO-bound sub-step (LLM-вызов, HTTP)."""
    await asyncio.sleep(delay)
    return "done"


async def flaky_tool(fail_times: int = 0, **_) -> str:
    """Падает первые fail_times вызовов, потом отдаёт результат."""
    if not hasattr(flaky_tool, "_counts"):
        flaky_tool._counts = {}  # type: ignore[attr-defined]
    key = id(asyncio.get_event_loop())
    flaky_tool._counts[key] = flaky_tool._counts.get(key, 0) + 1  # type: ignore[attr-defined]
    if flaky_tool._counts[key] <= fail_times:  # type: ignore[attr-defined]
        raise RuntimeError("transient failure")
    flaky_tool._counts[key] = 0  # type: ignore[attr-defined]
    return "recovered"


def make_vm() -> ExecutionVM:
    return ExecutionVM(
        llm=MockLLM(),
        tools={
            "fast": fast_tool,
            "slow": slow_tool,
            "flaky": flaky_tool,
        },
    )


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


async def timed(coro) -> tuple[float, bool]:
    """Возвращает (duration_sec, success)."""
    t0 = time.perf_counter()
    trace = await coro
    return time.perf_counter() - t0, trace.status == TraceStatus.SUCCESS


def row(label: str, runs: int, elapsed: float, success: int, extra: str = "") -> str:
    rps = runs / elapsed if elapsed > 0 else 0
    avg_ms = elapsed / runs * 1000
    ok = "✓" if success == runs else f"⚠ {success}/{runs}"
    extra_col = f"  {extra}" if extra else ""
    return f"  {label:<38} {avg_ms:>8.2f} ms   {rps:>8.1f} RPS   {ok}{extra_col}"


# ---------------------------------------------------------------------------
# BM1 — Retry overhead: success on Nth attempt
# ---------------------------------------------------------------------------


async def bm_retry_overhead(runs: int = 200) -> None:
    print("\n── BM1: Retry overhead (mock sleep disabled) ─────────────────────")

    async def _patch_sleep(delay):  # noqa: ARG001
        pass  # skip backoff в бенчмарке

    original_sleep = asyncio.sleep

    results = {}
    for fail_times, label in [
        (0, "0 retries (success 1st)"),
        (1, "1 retry  (success 2nd)"),
        (2, "2 retries (success 3rd)"),
    ]:
        vm = make_vm()
        program = Program(
            name="retry_bench",
            steps=[
                Step(
                    id="s",
                    type=StepType.TOOL,
                    tool="flaky",
                    args={"fail_times": fail_times},
                    on_error=OnError.RETRY,
                    max_retries=3,
                )
            ],
        )

        # Патчим sleep чтобы не ждать backoff
        asyncio.sleep = _patch_sleep  # type: ignore[assignment]
        try:
            t0 = time.perf_counter()
            tasks = [vm.run(program) for _ in range(runs)]
            traces = await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - t0
        finally:
            asyncio.sleep = original_sleep  # type: ignore[assignment]

        success = sum(1 for t in traces if t.status == TraceStatus.SUCCESS)
        results[label] = (elapsed, success)
        print(row(label, runs, elapsed, success))

    # Overhead: разница между 0 и 2 retries
    base = results["0 retries (success 1st)"][0]
    two = results["2 retries (success 3rd)"][0]
    overhead_pct = (two - base) / base * 100
    print(f"\n  Retry overhead (2 retries vs 0): +{overhead_pct:.1f}%")


# ---------------------------------------------------------------------------
# BM2 — max_concurrency scaling
# ---------------------------------------------------------------------------


async def bm_concurrency_scaling(
    n_steps: int = 20,
    step_delay: float = 0.02,
    caps: list[int | None] = None,
) -> None:
    if caps is None:
        caps = [None, 10, 5, 2, 1]

    print(f"\n── BM2: max_concurrency scaling ({n_steps} steps × {step_delay * 1000:.0f}ms each) ─")

    vm = make_vm()
    baseline_elapsed = None

    for cap in caps:
        label = f"max_concurrency={cap if cap is not None else 'None (no cap)'}"
        parallel_steps = [
            Step(
                id=f"s{i}",
                type=StepType.TOOL,
                tool="slow",
                args={"delay": step_delay},
            )
            for i in range(n_steps)
        ]
        program = Program(
            name="concurrency_bench",
            steps=[
                Step(
                    id="par",
                    type=StepType.PARALLEL,
                    max_concurrency=cap,
                    parallel_steps=parallel_steps,
                )
            ],
        )

        RUNS = 10
        t0 = time.perf_counter()
        traces = await asyncio.gather(*[vm.run(program) for _ in range(RUNS)])
        elapsed = time.perf_counter() - t0

        success = sum(1 for t in traces if t.status == TraceStatus.SUCCESS)
        avg_ms = elapsed / RUNS * 1000

        if baseline_elapsed is None:
            baseline_elapsed = elapsed
            extra = "(baseline)"
        else:
            slowdown = elapsed / baseline_elapsed
            extra = f"×{slowdown:.1f} vs baseline" if slowdown > 1.05 else "≈ baseline"

        print(row(label, RUNS, elapsed, success, extra))

    # Теоретические минимумы
    ideal_no_cap = step_delay * 1000
    ideal_cap1 = step_delay * n_steps * 1000
    print(f"\n  Theory: no cap ~{ideal_no_cap:.0f}ms | cap=1 ~{ideal_cap1:.0f}ms")


# ---------------------------------------------------------------------------
# BM3 — Parallel throughput: steps/sec при разном N
# ---------------------------------------------------------------------------


async def bm_parallel_throughput() -> None:
    print("\n── BM3: Parallel throughput (fast_tool, no I/O) ──────────────────")
    print(f"  {'Steps':>6}   {'cap':>8}   {'avg ms':>8}   {'steps/sec':>10}")

    vm = make_vm()

    configs = [
        (10, None),
        (20, None),
        (50, None),
        (100, None),
        (100, 10),
        (100, 5),
    ]

    for n_steps, cap in configs:
        parallel_steps = [Step(id=f"s{i}", type=StepType.TOOL, tool="fast") for i in range(n_steps)]
        program = Program(
            name="throughput_bench",
            steps=[
                Step(
                    id="par",
                    type=StepType.PARALLEL,
                    max_concurrency=cap,
                    parallel_steps=parallel_steps,
                )
            ],
        )

        RUNS = 30
        t0 = time.perf_counter()
        traces = await asyncio.gather(*[vm.run(program) for _ in range(RUNS)])
        elapsed = time.perf_counter() - t0

        avg_ms = elapsed / RUNS * 1000
        steps_per_sec = (n_steps * RUNS) / elapsed
        cap_str = str(cap) if cap is not None else "None"
        success = sum(1 for t in traces if t.status == TraceStatus.SUCCESS)
        ok = "✓" if success == RUNS else f"⚠ {success}/{RUNS}"
        print(f"  {n_steps:>6}   {cap_str:>8}   {avg_ms:>8.2f}   {steps_per_sec:>10.1f}   {ok}")


# ---------------------------------------------------------------------------
# BM4 — SKIPPED contract overhead в downstream resolver
# ---------------------------------------------------------------------------


async def bm_skipped_resolver_overhead(runs: int = 300) -> None:
    print("\n── BM4: SKIPPED→None resolver overhead ───────────────────────────")

    vm = make_vm()

    async def fail_tool(**_):
        raise RuntimeError("intentional")

    vm2 = ExecutionVM(
        llm=MockLLM(),
        tools={"fast": fast_tool, "fail": fail_tool},
    )

    # Baseline: все успешны
    prog_ok = Program(
        name="baseline",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                parallel_steps=[
                    Step(id=f"s{i}", type=StepType.TOOL, tool="fast") for i in range(5)
                ],
            ),
            Step(id="next", type=StepType.LLM, prompt="$s0.output $s1.output"),
        ],
    )

    # С SKIPPED: часть None в контексте
    prog_skip = Program(
        name="with_skipped",
        steps=[
            Step(
                id="par",
                type=StepType.PARALLEL,
                on_error=OnError.SKIP,
                parallel_steps=[
                    Step(id="s0", type=StepType.TOOL, tool="fast"),
                    Step(id="s1", type=StepType.TOOL, tool="fail"),
                    Step(id="s2", type=StepType.TOOL, tool="fast"),
                    Step(id="s3", type=StepType.TOOL, tool="fail"),
                    Step(id="s4", type=StepType.TOOL, tool="fast"),
                ],
            ),
            Step(
                id="next",
                type=StepType.LLM,
                prompt="$s0.output $s1.output $s2.output $s3.output $s4.output",
            ),
        ],
    )

    # Run baseline
    t0 = time.perf_counter()
    await asyncio.gather(*[vm.run(prog_ok) for _ in range(runs)])
    baseline_elapsed = time.perf_counter() - t0

    # Run with skipped
    t0 = time.perf_counter()
    await asyncio.gather(*[vm2.run(prog_skip) for _ in range(runs)])
    skip_elapsed = time.perf_counter() - t0

    overhead = (skip_elapsed - baseline_elapsed) / baseline_elapsed * 100
    print(row("all success (baseline)", runs, baseline_elapsed, runs))
    print(
        row(
            "2/5 SKIPPED → None in resolver", runs, skip_elapsed, runs, f"+{overhead:.1f}% overhead"
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 65)
    print("  llm-nano-vm v0.3.0 — benchmark suite")
    print(f"  OS:     {platform.system()} {platform.release()}")
    print(f"  CPU:    {platform.processor() or 'unknown'} ({cpu_count()} cores)")
    print(f"  Python: {sys.version.split()[0]}")
    print("=" * 65)

    await bm_retry_overhead(runs=200)
    await bm_concurrency_scaling(n_steps=20, step_delay=0.02)
    await bm_parallel_throughput()
    await bm_skipped_resolver_overhead(runs=300)

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
