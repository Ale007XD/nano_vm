import asyncio
import hashlib
import json
import platform
import random
import time
from collections import Counter
from datetime import datetime

# Импорты для красивого форматирования (Rich)
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from nano_vm.adapters import MockLLMAdapter
from nano_vm.models import Program, Step, StepType, TraceStatus

# Импорты из ядра (llm-nano-vm v0.6.0)
from nano_vm.vm import ExecutionVM

# 1. Настройка программы
BENCHMARK_PROGRAM = Program(
    name="benchmark_10k_5steps",
    steps=[
        Step(
            id="step_1_llm", type=StepType.LLM, prompt="Analyze field $value", output_key="analysis"
        ),
        Step(
            id="step_2_tool",
            type=StepType.TOOL,
            tool="mock_processor",
            parameters={"data": "$analysis"},
        ),
        # ИСПРАВЛЕНИЕ: Безопасное выражение без использования запрещенной функции float()
        Step(
            id="step_3_cond",
            type=StepType.CONDITION,
            condition="$value > 0.9",
            then="step_error",
            otherwise="step_4_success",
        ),
        Step(id="step_4_success", type=StepType.TOOL, tool="mock_finalize", parameters={}),
        Step(
            id="step_error",
            type=StepType.TOOL,
            tool="unregistered_tool_to_force_error",
            parameters={},
        ),
    ],
)


# Простые мок-инструменты для ВМ
async def mock_processor(**kwargs):
    return {"processed": True}


async def mock_finalize(**kwargs):
    return {"status": "done"}


async def process_item(
    item: dict, vm: ExecutionVM, sem: asyncio.Semaphore, error_log_file, progress, task_id
) -> str:
    """Обработка одного элемента с ограничением конкурентности"""
    async with sem:
        # ИСПРАВЛЕНИЕ: Передаем число напрямую, песочница сама безопасно подставит его в $value
        trace = await vm.run(BENCHMARK_PROGRAM, context={"value": item["value"]})

        result_status = "SUCCESS"
        if trace.status == TraceStatus.FAILED:
            error_msg = str(trace.error)
            error_hash = hashlib.md5(error_msg.encode("utf-8")).hexdigest()[:8]

            # Пишем лог конкретного прогона
            log_entry = json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "item_id": item["id"],
                    "trace_id": getattr(trace, "trace_id", "unknown"),
                    "error_hash": error_hash,
                    "error_msg": error_msg,
                }
            )
            error_log_file.write(log_entry + "\n")
            result_status = f"ERROR_{error_hash}"

        # Обновляем UI
        progress.advance(task_id)
        return result_status


async def run_benchmark():
    console = Console()
    TOTAL_ITEMS = 10000
    RUNS = 5
    CONCURRENCY_LIMIT = 200

    console.print(
        Panel(
            f"[bold cyan]Запуск стресс-теста транзакционного ядра v0.6.0[/bold cyan]\n"
            f"Размер выборки: [bold]{TOTAL_ITEMS}[/bold] записей\n"
            f"Количество прогонов: [bold]{RUNS}[/bold]\n"
            f"Уровень параллелизма: [bold]{CONCURRENCY_LIMIT}[/bold] задач",
            border_style="cyan",
        )
    )

    # Генерация массива (ГЕНЕРИРУЕТСЯ ОДИН РАЗ ДЛЯ ЧИСТОТЫ ЭКСПЕРИМЕНТА)
    console.print("[dim]Генерация эталонного массива данных...[/dim]")
    dataset = [{"id": i, "value": round(random.uniform(0.0, 1.0), 3)} for i in range(TOTAL_ITEMS)]

    run_metrics = []

    # Запускаем прогоны
    for run_idx in range(1, RUNS + 1):
        # Изолированная ВМ для каждого прогона (защита от роста массива calls в моке)
        vm = ExecutionVM(llm=MockLLMAdapter())
        vm.register_tool("mock_processor", mock_processor)
        vm.register_tool("mock_finalize", mock_finalize)

        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
        log_filename = f"benchmark_run_{run_idx}_{int(time.time())}.log"

        # Настройка прогресс-бара для текущего прогона
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Прогон {run_idx}/{RUNS}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("process", total=TOTAL_ITEMS)
            start_time = time.time()

            with open(log_filename, "w", encoding="utf-8") as error_log_file:
                tasks = [
                    process_item(item, vm, sem, error_log_file, progress, task_id)
                    for item in dataset
                ]
                results = await asyncio.gather(*tasks)

            duration = time.time() - start_time

        # Агрегация результатов прогона
        stats = Counter(results)
        success_count = stats.pop("SUCCESS", 0)
        error_count = TOTAL_ITEMS - success_count
        throughput = TOTAL_ITEMS / duration

        run_metrics.append(
            {
                "run": run_idx,
                "duration": duration,
                "throughput": throughput,
                "success": success_count,
                "errors": error_count,
                "log_file": log_filename,
            }
        )

    # ==========================================
    # ИТОГОВЫЙ ОТЧЕТ
    # ==========================================
    console.print("\n")
    table = Table(title="📊 Результаты бенчмарка", show_header=True, header_style="bold magenta")
    table.add_column("Прогон", justify="center")
    table.add_column("Время (сек)", justify="right")
    table.add_column("Скорость", justify="right")
    table.add_column("Успех", justify="right", style="green")
    table.add_column("Ошибки", justify="right", style="red")
    table.add_column("Файл логов JSON", justify="left", style="dim")

    total_duration = sum(m["duration"] for m in run_metrics)
    avg_duration = total_duration / RUNS
    avg_throughput = sum(m["throughput"] for m in run_metrics) / RUNS

    for m in run_metrics:
        table.add_row(
            str(m["run"]),
            f"{m['duration']:.2f}",
            f"{m['throughput']:.0f} ит/с",
            str(m["success"]),
            str(m["errors"]),
            m["log_file"],
        )

    table.add_section()
    table.add_row(
        "[bold]СРЕДНЕЕ[/bold]",
        f"[bold]{avg_duration:.2f}[/bold]",
        f"[bold cyan]{avg_throughput:.0f} ит/с[/bold cyan]",
        "-",
        "-",
        f"[bold]Общее время: {total_duration:.2f} сек[/bold]",
    )

    console.print(table)

    sys_panel = Panel(
        f"[bold]Интерпретатор:[/bold] Python {platform.python_version()}\n"
        f"[bold]Платформа:[/bold] {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"[bold]Контекст выполнения:[/bold] 100% CPU Bound (MockLLM), Asynchronous I/O simulation",
        title="⚙️ Системные данные",
        border_style="blue",
    )
    console.print(sys_panel)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
