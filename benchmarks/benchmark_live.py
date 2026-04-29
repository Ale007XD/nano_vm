"""
benchmarks/benchmark_live.py
=============================
Боевой тест llm-nano-vm через реальный API (OpenRouter).
Измеряет сетевой overhead, реальный TTFT (Time-To-First-Token) / Latency,
работу параллельных шагов и подсчет стоимости.

Требует:
    pip install llm-nano-vm[litellm] rich
    export OPENROUTER_API_KEY="sk-or-..."

Запуск:
    python benchmarks/benchmark_live.py
"""

import asyncio
import json
import os
import sys
import time
import urllib.request

# Подавляем лишний спам от litellm при сетевых ошибках
import litellm

litellm.suppress_debug_info = True
litellm.set_verbose = False

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    print("rich not installed. Run: pip install rich")
    sys.exit(1)

from nano_vm import ExecutionVM, Program  # noqa: E402
from nano_vm.models import OnError, Step, StepType  # noqa: E402

# Используем реальный адаптер
try:
    from nano_vm.adapters import LiteLLMAdapter
except ImportError:
    print("LiteLLMAdapter not found. Run: pip install llm-nano-vm[litellm]")
    sys.exit(1)


console = Console()

# Используем самую стабильную модель OpenAI
MODEL_ID = "openrouter/openai/gpt-4o-mini"


def check_openrouter_auth(api_key: str) -> dict:
    """Пинг API OpenRouter для проверки ключа и лимитов."""
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            return data.get("data", {})
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        return {"error": f"HTTP {e.code}: {e.reason} - {error_body}"}
    except Exception as e:
        return {"error": str(e)}


async def run_live_benchmark():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[bold red]Ошибка:[/] Не задан OPENROUTER_API_KEY")
        sys.exit(1)

    # ---------------------------------------------------------
    # 0. Пинг и проверка авторизации
    # ---------------------------------------------------------
    console.print("[dim]Пинг OpenRouter и проверка ключа...[/]")
    auth_info = check_openrouter_auth(api_key)

    auth_status_str = ""
    if "error" in auth_info:
        auth_status_str = f"[bold red]ОШИБКА АВТОРИЗАЦИИ:[/] {auth_info['error']}"
    else:
        limit = auth_info.get("limit")
        usage = auth_info.get("usage")
        is_free = auth_info.get("is_free_tier")
        auth_status_str = (
            f"[bright_green]Ключ ВАЛИДЕН[/] | Free Tier: {is_free}"
            f" | Лимит: {limit} | Использовано: {usage}"
        )

    console.print(
        Panel.fit(
            f"[bold bright_green]Live API Benchmark[/]\n"
            f"Модель: [cyan]{MODEL_ID}[/]\n"
            f"Статус: {auth_status_str}",
            box=box.HEAVY,
            border_style="bright_green",
        )
    )

    # Инициализация боевого VM
    vm = ExecutionVM(llm=LiteLLMAdapter(model=MODEL_ID))

    # ---------------------------------------------------------
    # Сценарий 1: Серийный вызов (3 шага)
    # ---------------------------------------------------------
    console.print("\n[bold yellow]1. Серийное выполнение (3 шага)...[/]")
    prog_serial = Program(
        name="Live_Serial",
        steps=[
            Step(id="s1", type=StepType.LLM, prompt="Say 'ping 1' and nothing else."),
            Step(id="s2", type=StepType.LLM, prompt="Say 'ping 2' and nothing else."),
            Step(id="s3", type=StepType.LLM, prompt="Say 'ping 3' and nothing else."),
        ],
    )

    t0 = time.perf_counter()
    trace_serial = await vm.run(prog_serial)
    serial_time = time.perf_counter() - t0

    # ---------------------------------------------------------
    # Сценарий 2: Конкурентный вызов (5 шагов параллельно)
    # ---------------------------------------------------------
    console.print(
        "[bold yellow]2. Конкурентное выполнение (5 sub-steps in parallel)...[/]"
    )
    N_PARALLEL = 5
    prog_parallel = Program(
        name="Live_Parallel",
        steps=[
            Step(
                id="par_fetch",
                type=StepType.PARALLEL,
                on_error=OnError.SKIP,
                parallel_steps=[
                    Step(
                        id=f"p{i}",
                        type=StepType.LLM,
                        prompt=f"Return exactly the number {i}.",
                    )
                    for i in range(N_PARALLEL)
                ],
            )
        ],
    )

    t0 = time.perf_counter()
    trace_parallel = await vm.run(prog_parallel)
    parallel_time = time.perf_counter() - t0

    # ---------------------------------------------------------
    # Вывод результатов
    # ---------------------------------------------------------
    t = Table(
        title="[bold white]Результаты Live API[/]",
        box=box.SIMPLE_HEAVY,
        show_header=True,
    )
    t.add_column("Сценарий", style="cyan")
    t.add_column("Статус", justify="center")
    t.add_column("Время (Wall-clock)", justify="right")
    t.add_column("Токены (Total)", justify="right")
    t.add_column("Стоимость ($)", justify="right", style="bright_green")

    def _add_trace_row(label, t_time, trace):
        status_val = (
            trace.status.value if hasattr(trace.status, "value") else str(trace.status)
        )
        status_color = (
            "[bright_green]SUCCESS[/]"
            if str(status_val).upper() == "SUCCESS"
            else f"[red]{status_val}[/]"
        )

        tokens = trace.total_tokens()
        tokens_str = str(tokens) if tokens is not None else "N/A"

        cost_str = "N/A"
        if hasattr(trace, "total_cost_usd"):
            cost_val = trace.total_cost_usd()
            if cost_val is not None:
                cost_str = f"${cost_val:.6f}"

        t.add_row(label, status_color, f"{t_time:.2f} s", tokens_str, cost_str)

    _add_trace_row("Серийный (3 шага)", serial_time, trace_serial)
    _add_trace_row(f"Параллельный ({N_PARALLEL} шагов)", parallel_time, trace_parallel)

    console.print("\n")
    console.print(t)

    # Детализация параллельного блока
    if getattr(trace_parallel, "steps", None):
        console.print("\n[dim]Детализация параллельного блока (Trace Steps):[/]")
        for step in trace_parallel.steps:
            status_val = (
                step.status.value if hasattr(step.status, "value") else str(step.status)
            )
            color = (
                "bright_green"
                if str(status_val).upper() == "SUCCESS"
                else "yellow"
                if str(status_val).upper() == "SKIPPED"
                else "red"
            )
            status_fmt = f"[{color}]{status_val}[/]"

            output_preview = (
                str(getattr(step, "output", "None")).strip().replace("\n", " ")[:40]
            )
            dur = getattr(step, "duration_ms", 0)
            prefix = "►" if step.step_id == "par_fetch" else "  ├─"

            console.print(
                f"{prefix} {step.step_id:9} | {status_fmt:15}"
                f" | {dur:5} ms | Output: {output_preview}"
            )


if __name__ == "__main__":
    asyncio.run(run_live_benchmark())
    
