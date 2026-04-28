"""
benchmarks/benchmark_v040.py
=============================
Бенчмарки v0.4.0: регрессия v0.3.0 + overhead новых механизмов.

BM1 — retry overhead (regression)
BM2 — max_concurrency scaling (regression)
BM3 — parallel throughput (regression)
BM4 — SKIPPED resolver overhead (regression)
BM5 — max_steps check overhead
BM6 — fingerprint/STALLED detection overhead
BM7 — max_tokens check overhead

Запуск:
    python benchmarks/benchmark_v040.py

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
    from nano_vm.models import LLMUsage, OnError, Step, StepStatus, StepType
except ImportError:
    print("❌ nano-vm not installed. Run: pip install llm-nano-vm")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Shared mock infrastructure
# ---------------------------------------------------------------------------


class MockLLM:
    async def complete(self, messages):
        await asyncio.sleep(0)
        return "ok"


class MockLLMWithUsage:
    """Returns usage metadata — required for max_tokens benchmark."""

    def __init__(self, tokens_per_call: int = 10):
        self.tokens_per_call = tokens_per_call

    async def complete(self, messages):
        await asyncio.sleep(0)
        return "ok"


class _CountingVM(ExecutionVM):
    """Injects LLMUsage into LLM step results to simulate token consumption."""

    def __init__(self, tokens_per_call: int = 10, **kwargs):
        super().__init__(llm=MockLLMWithUsage(tokens_per_call), **kwargs)
        self._tokens_per_call = tokens_per_call

    async def _run_step(self, step, state):
        result, new_state, sub_results = await super()._run_step(step, state)
        if step.type == StepType.LLM and result.status == StepStatus.SUCCESS:
            usage = LLMUsage(
                prompt_tokens=self._tokens_per_call // 2,
                completion_tokens=self._tokens_per_call // 2,
                total_tokens=self._tokens_per_call,
            )
            result = result.model_copy(update={"usage": usage})
        return result, new_state, sub_results


async def fast_tool(**_) -> str:
    await asyncio.sleep(0)
    return "done"


async def slow_tool(delay: float = 0.01, **_) -> str:
    await asyncio.sleep(delay)
    return "done"


async def flaky_tool(fail_times: int = 0, **_) -> str:
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
        tools={"fast": fast_tool, "slow": slow_tool, "flaky": flaky_tool},
    )


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


async def timed(coro) -> tuple[float, bool]:
    t0 = time.perf_counter()
    trace = await coro
    return time.perf_counter() - t0, trace.status == TraceStatus.SUCCESS


def row(label: str, runs: int, elapsed: float, success: int, extra: str = "") -> str:
    rps = runs / elapsed if elapsed > 0 else 0
    avg_ms = elapsed / runs * 1000
    ok = "✓" if success == runs else f"⚠ {success}/{runs}"
    extra_col = f"  {extra}" if extra else ""
    return f"  {label:<42} {avg_ms:>8.2f} ms   {rps:>8.1f} RPS   {ok}{extra_col}"


# ---------------------------------------------------------------------------
# BM1 — Retry overhead (regression from v0.3.0)
# ---------------------------------------------------------------------------


async def bm_retry_overhead(runs: int = 200) -> None:
    print("\n── BM1: Retry overhead (mock sleep disabled) ─────────────────────────")

    async def _patch_sleep(delay):  # noqa: ARG001
        pass

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

    base = results["0 retries (success 1st)"][0]
    two = results["2 retries (success 3rd)"][0]
    overhead_pct = (two - base) / base * 100
    print(f"\n  Retry overhead (2 retries vs 0): +{overhead_pct:.1f}%")


# ---------------------------------------------------------------------------
# BM2 — max_concurrency scaling (regression from v0.3.0)
# ---------------------------------------------------------------------------


async def bm_concurrency_scaling(
    n_steps: int = 20,
    step_delay: float = 0.02,
    caps: list[int | None] = None,
) -> None:
    if caps is None:
        caps = [None, 10, 5, 2, 1]

    print(
        f"\n── BM2: max_concurrency scaling "
        f"({n_steps} steps × {step_delay * 1000:.0f}ms each) ──────────"
    )

    vm = make_vm()
    baseline_elapsed = None

    for cap in caps:
        label = f"max_concurrency={cap if cap is not None else 'None (no cap)'}"
        parallel_steps = [
            Step(id=f"s{i}", type=StepType.TOOL, tool="slow", args={"delay": step_delay})
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

        if baseline_elapsed is None:
            baseline_elapsed = elapsed
            extra = "(baseline)"
        else:
            slowdown = elapsed / baseline_elapsed
            extra = f"×{slowdown:.1f} vs baseline" if slowdown > 1.05 else "≈ baseline"

        print(row(label, RUNS, elapsed, success, extra))

    ideal_no_cap = step_delay * 1000
    ideal_cap1 = step_delay * n_steps * 1000
    print(f"\n  Theory: no cap ~{ideal_no_cap:.0f}ms | cap=1 ~{ideal_cap1:.0f}ms")


# ---------------------------------------------------------------------------
# BM3 — Parallel throughput (regression from v0.3.0)
# ---------------------------------------------------------------------------


async def bm_parallel_throughput() -> None:
    print("\n── BM3: Parallel throughput (fast_tool, no I/O) ──────────────────────")
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
        parallel_steps = [
            Step(id=f"s{i}", type=StepType.TOOL, tool="fast") for i in range(n_steps)
        ]
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
# BM4 — SKIPPED resolver overhead (regression from v0.3.0)
# ---------------------------------------------------------------------------


async def bm_skipped_resolver_overhead(runs: int = 300) -> None:
    print("\n── BM4: SKIPPED→None resolver overhead ───────────────────────────────")

    vm = make_vm()

    async def fail_tool(**_):
        raise RuntimeError("intentional")

    vm2 = ExecutionVM(llm=MockLLM(), tools={"fast": fast_tool, "fail": fail_tool})

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

    t0 = time.perf_counter()
    await asyncio.gather(*[vm.run(prog_ok) for _ in range(runs)])
    baseline_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    await asyncio.gather(*[vm2.run(prog_skip) for _ in range(runs)])
    skip_elapsed = time.perf_counter() - t0

    overhead = (skip_elapsed - baseline_elapsed) / baseline_elapsed * 100
    print(row("all success (baseline)", runs, baseline_elapsed, runs))
    print(
        row(
            "2/5 SKIPPED → None in resolver",
            runs,
            skip_elapsed,
            runs,
            f"+{overhead:.1f}% overhead",
        )
    )


# ---------------------------------------------------------------------------
# BM5 — max_steps check overhead (v0.4.0)
# ---------------------------------------------------------------------------


async def bm_max_steps_overhead(runs: int = 500) -> None:
    print("\n── BM5: max_steps check overhead (v0.4.0) ────────────────────────────")

    vm = make_vm()
    n_steps = 10

    steps = [Step(id=f"s{i}", type=StepType.LLM, prompt="hi") for i in range(n_steps)]

    # Baseline: no budget
    prog_baseline = Program(name="baseline", steps=steps)

    # With max_steps=None (disabled, explicit)
    prog_none = Program(name="max_steps_none", max_steps=None, steps=steps)

    # With max_steps=1000 (large cap, never fires)
    prog_large = Program(name="max_steps_large", max_steps=1000, steps=steps)

    # With max_steps=10 (equals step count — fires right after last step on next iter)
    prog_exact = Program(name="max_steps_exact", max_steps=n_steps, steps=steps)

    configs = [
        ("no max_steps (baseline)", prog_baseline),
        ("max_steps=None (explicit)", prog_none),
        ("max_steps=1000 (never fires)", prog_large),
        ("max_steps=10 (= step count)", prog_exact),
    ]

    baseline_elapsed = None
    for label, prog in configs:
        t0 = time.perf_counter()
        traces = await asyncio.gather(*[vm.run(prog) for _ in range(runs)])
        elapsed = time.perf_counter() - t0
        success = sum(1 for t in traces if t.status == TraceStatus.SUCCESS)

        if baseline_elapsed is None:
            baseline_elapsed = elapsed
            extra = "(baseline)"
        else:
            delta_pct = (elapsed - baseline_elapsed) / baseline_elapsed * 100
            sign = "+" if delta_pct >= 0 else ""
            extra = f"{sign}{delta_pct:.1f}% vs baseline"

        print(row(label, runs, elapsed, success, extra))

    print(
        "\n  max_steps adds one integer comparison per loop iteration. "
        "Expected overhead: < 1%."
    )


# ---------------------------------------------------------------------------
# BM6 — fingerprint/STALLED detection overhead (v0.4.0)
# ---------------------------------------------------------------------------


async def bm_fingerprint_overhead(runs: int = 500) -> None:
    print("\n── BM6: Fingerprint/STALLED detection overhead (v0.4.0) ──────────────")

    vm = make_vm()
    n_steps = 10
    steps = [Step(id=f"s{i}", type=StepType.LLM, prompt="hi") for i in range(n_steps)]

    prog_baseline = Program(name="baseline", steps=steps)
    prog_stalled_none = Program(name="stalled_none", max_stalled_steps=None, steps=steps)
    prog_stalled_large = Program(name="stalled_large", max_stalled_steps=1000, steps=steps)

    configs = [
        ("no max_stalled_steps (baseline)", prog_baseline),
        ("max_stalled_steps=None (disabled)", prog_stalled_none),
        ("max_stalled_steps=1000 (never fires)", prog_stalled_large),
    ]

    baseline_elapsed = None
    for label, prog in configs:
        t0 = time.perf_counter()
        traces = await asyncio.gather(*[vm.run(prog) for _ in range(runs)])
        elapsed = time.perf_counter() - t0
        success = sum(1 for t in traces if t.status == TraceStatus.SUCCESS)

        if baseline_elapsed is None:
            baseline_elapsed = elapsed
            extra = "(baseline)"
        else:
            delta_pct = (elapsed - baseline_elapsed) / baseline_elapsed * 100
            sign = "+" if delta_pct >= 0 else ""
            extra = f"{sign}{delta_pct:.1f}% vs baseline"

        print(row(label, runs, elapsed, success, extra))

    # Snapshot overhead (sha256 per step)
    print("\n  Snapshot (sha256) overhead per step:")
    prog_snapshot = Program(name="snapshot", steps=steps)

    # Warm up
    await asyncio.gather(*[vm.run(prog_snapshot) for _ in range(50)])

    t0 = time.perf_counter()
    await asyncio.gather(*[vm.run(prog_snapshot) for _ in range(runs)])
    elapsed_snap = time.perf_counter() - t0

    t0 = time.perf_counter()
    await asyncio.gather(*[vm.run(prog_baseline) for _ in range(runs)])
    elapsed_base = time.perf_counter() - t0

    snap_overhead_ms = (elapsed_snap - elapsed_base) / runs / n_steps * 1000
    print(
        f"  sha256 per step: ~{snap_overhead_ms:.4f} ms "
        f"(total across {n_steps} steps: ~{snap_overhead_ms * n_steps:.4f} ms/program)"
    )
    print(
        "\n  Fingerprint check: hash(frozenset) in-process comparison. "
        "Snapshot: sha256 per step (storage path only)."
    )


# ---------------------------------------------------------------------------
# BM7 — max_tokens check overhead (v0.4.0)
# ---------------------------------------------------------------------------


async def bm_max_tokens_overhead(runs: int = 500) -> None:
    print("\n── BM7: max_tokens check overhead (v0.4.0) ───────────────────────────")

    n_steps = 10
    steps = [Step(id=f"s{i}", type=StepType.LLM, prompt="hi") for i in range(n_steps)]

    # Baseline: no token budget, no usage injection
    vm_plain = make_vm()
    prog_baseline = Program(name="baseline", steps=steps)

    # With max_tokens=None (disabled, explicit)
    prog_none = Program(name="tokens_none", max_tokens=None, steps=steps)

    # With max_tokens=1_000_000 (large cap, never fires) + usage injection
    prog_large = Program(name="tokens_large", max_tokens=1_000_000, steps=steps)

    # With max_tokens=None but usage injected (cost of total_tokens() call)
    vm_counting = _CountingVM(tokens_per_call=10)
    prog_with_usage = Program(name="tokens_with_usage", max_tokens=None, steps=steps)
    prog_budget_active = Program(name="tokens_budget", max_tokens=1_000_000, steps=steps)

    configs: list[tuple[str, ExecutionVM, Program]] = [
        ("no max_tokens, no usage (baseline)", vm_plain, prog_baseline),
        ("max_tokens=None (explicit)", vm_plain, prog_none),
        ("max_tokens=1M (never fires, no usage)", vm_plain, prog_large),
        ("max_tokens=None + usage injected", vm_counting, prog_with_usage),
        ("max_tokens=1M + usage injected", vm_counting, prog_budget_active),
    ]

    baseline_elapsed = None
    for label, vm, prog in configs:
        t0 = time.perf_counter()
        traces = await asyncio.gather(*[vm.run(prog) for _ in range(runs)])
        elapsed = time.perf_counter() - t0
        success = sum(1 for t in traces if t.status == TraceStatus.SUCCESS)

        if baseline_elapsed is None:
            baseline_elapsed = elapsed
            extra = "(baseline)"
        else:
            delta_pct = (elapsed - baseline_elapsed) / baseline_elapsed * 100
            sign = "+" if delta_pct >= 0 else ""
            extra = f"{sign}{delta_pct:.1f}% vs baseline"

        print(row(label, runs, elapsed, success, extra))

    print(
        "\n  max_tokens adds one total_tokens() aggregation per loop iteration "
        "when enabled. total_tokens() is O(N steps)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("=" * 70)
    print("  llm-nano-vm v0.4.0 — benchmark suite")
    print(f"  OS:     {platform.system()} {platform.release()}")
    print(f"  CPU:    {platform.processor() or 'unknown'} ({cpu_count()} cores)")
    print(f"  Python: {sys.version.split()[0]}")
    print("=" * 70)
    print()
    print("  Regression (BM1–BM4): verifies v0.3.0 mechanisms unchanged.")
    print("  New (BM5–BM7):        measures v0.4.0 budget/guard overhead.")

    await bm_retry_overhead(runs=200)
    await bm_concurrency_scaling(n_steps=20, step_delay=0.02)
    await bm_parallel_throughput()
    await bm_skipped_resolver_overhead(runs=300)
    await bm_max_steps_overhead(runs=500)
    await bm_fingerprint_overhead(runs=500)
    await bm_max_tokens_overhead(runs=500)

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
