import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from nano_vm.adapters import MockLLMAdapter
from nano_vm.models import Program, Step, StepType, TraceStatus
from nano_vm.vm import ExecutionVM

# ─── Конфигурация ────────────────────────────────────────────────
INCIDENT_DIR = Path(__file__).parent / "incidents" / "pocketos_cursor_db_delete"
REPLAY_ITERATIONS = 10_000      # Количество повторов за прогон
RUNS = 5                        # Количество независимых прогонов
CONCURRENCY_LIMIT = 200         # Лимит параллельных задач (как в эталонном бенчмарке)

# ─── Адаптер, возвращающий конкретные ответы для шагов ─────────
class IncidentMockLLMAdapter(MockLLMAdapter):
    """Подменяет ответ LLM на основе поля mock_response в трассе инцидента."""

    def __init__(self, step_responses: Dict[str, str]):
        super().__init__()
        self.step_responses = step_responses  # ключ – step_id, значение – ответ

    async def complete(self, prompt: str, **kwargs) -> str:
        # kwargs может содержать step_id, если пробросить из VM (зависит от реализации)
        # Альтернатива: ищем ответ по полному совпадению промпта.
        # Для простоты будем искать по сохранённому prompt.
        # В реальном коде надо адаптировать под ваш API.
        for step_id, expected_prompt in kwargs.get("step_prompts", {}).items():
            if prompt == expected_prompt:
                return self.step_responses.get(step_id, "OK (default)")
        # Если промпт не распознан – возвращаем безопасное значение
        return "OK (fallback)"


# ─── Загрузка инцидента ─────────────────────────────────────────
def load_incident(path: Path) -> dict:
    with open(path / "incident.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_program(incident: dict) -> Program:
    """Создаёт Program из шагов инцидента."""
    steps = []
    step_prompts = {}
    for step_data in incident["steps"]:
        step_type = getattr(StepType, step_data["type"].upper(), StepType.TOOL)
        kwargs = {"id": step_data["step_id"], "type": step_type}

        if step_type == StepType.LLM:
            kwargs["prompt"] = step_data["prompt"]
            kwargs["output_key"] = f"out_{step_data['step_id']}"
            # Сохраняем промпт для адаптера
            step_prompts[step_data["step_id"]] = step_data["prompt"]
        elif step_type == StepType.TOOL:
            kwargs["tool"] = step_data["tool"]
            # Параметры можно передать как пустой dict или из expected_output
        elif step_type == StepType.CONDITION:
            kwargs["condition"] = step_data.get("condition", "true")
            kwargs["then"] = step_data.get("expected_output", {}).get("then", "next")
            kwargs["otherwise"] = step_data.get("expected_output", {}).get("otherwise", "next")

        steps.append(Step(**kwargs))

    return Program(name=f"incident_{incident['incident_id']}", steps=steps), step_prompts


# ─── Анализ инвариантов (заглушка под реальную логику) ─────────
def check_invariants(incident: dict, trace) -> List[Dict[str, Any]]:
    """Возвращает список нарушенных инвариантов."""
    violations = []
    # Пока проверяем только наличие expected_violations в трассе и статус TraceStatus.FAILED
    if trace.status == TraceStatus.FAILED:
        for v in incident.get("expected_violations", []):
            violations.append({
                "invariant": v.split(":")[0].strip(),
                "description": v,
                "evidence": {"trace_status": "FAILED", "error": str(trace.error)[:200]}
            })
    return violations


# ─── Главная функция реплея ────────────────────────────────────
async def replay_incident(incident_dir: Path, iterations: int, runs: int, concurrency: int):
    console = Console()
    incident = load_incident(incident_dir)
    program, step_prompts = build_program(incident)

    # Подготавливаем маппинг ответов LLM
    step_responses = {
        step["step_id"]: step["mock_response"]
        for step in incident["steps"] if step["type"] == "llm"
    }

    console.print(
        Panel(
            f"[bold cyan]Forensic Replay[/bold cyan] — {incident['title']}\n"
            f"Шагов: [bold]{len(program.steps)}[/bold] | "
            f"Итераций: {iterations} × {runs} прогонов\n"
            f"Ожидаемые нарушения: {len(incident.get('expected_violations', []))}",
            border_style="cyan",
        )
    )

    # Итоговая статистика
    total_success = 0
    total_violations = 0
    deterministic_match = True
    first_step_ids = None

    for run_idx in range(1, runs + 1):
        # Каждый прогон – новый инстанс VM и адаптера
        adapter = IncidentMockLLMAdapter(step_responses)
        # Передаём step_prompts как дополнительный контекст
        adapter.step_prompts = step_prompts  # кастомное поле
        vm = ExecutionVM(llm=adapter)
        # Регистрируем инструменты-заглушки (чтобы не падало на tool-шагах)
        vm.register_tool("execute_sql", lambda **kw: {"status": "ok"})
        vm.register_tool("mock_processor", lambda **kw: {"processed": True})
        vm.register_tool("mock_finalize", lambda **kw: {"status": "done"})
        # (можно добавить другие из трассы)

        sem = asyncio.Semaphore(concurrency)
        start_time = time.time()
        success_count = 0
        violation_count = 0

        async def run_one(_):
            nonlocal success_count, violation_count
            async with sem:
                trace = await vm.run(program, context=incident.get("initial_context", {}))
                if trace.status == TraceStatus.SUCCESS:
                    success_count += 1
                else:
                    violation_count += 1
                return trace

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Прогон {run_idx}/{runs}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("replay", total=iterations)
            # Последовательный запуск для сохранения детерминизма (можно асинхронно, но осторожно)
            for i in range(iterations):
                trace = await run_one(None)
                # Проверяем структуру шагов (детерминизм)
                step_ids = [s.step_id for s in trace.steps]
                if first_step_ids is None:
                    first_step_ids = step_ids
                elif step_ids != first_step_ids:
                    deterministic_match = False
                progress.advance(task_id)

        duration = time.time() - start_time
        total_success += success_count
        total_violations += violation_count
        console.print(f"  [{run_idx}] Успешно: {success_count}, Ошибок: {violation_count}, "
                      f"Длительность: {duration:.1f}с")

    # Генерация forensic_report.json
    report = {
        "incident_id": incident["incident_id"],
        "total_replays": runs * iterations,
        "total_success": total_success,
        "total_violations": total_violations,
        "deterministic_match": deterministic_match,
        "violations": incident.get("expected_violations", [])
    }
    report_path = incident_dir / "forensic_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    console.print(f"\n[bold green]Отчёт сохранён в {report_path}[/bold green]")

    # Вывод сводки
    console.print(
        Panel(
            f"Детерминированная структура: [bold]{'✅ ДА' if deterministic_match else '❌ НЕТ'}[/bold]\n"
            f"Всего успешных реплеев: {total_success}\n"
            f"Всего реплеев с нарушениями: {total_violations}\n"
            f"Ожидаемые инварианты: {', '.join(incident.get('expected_violations', ['нет']))}",
            title="Forensic Summary",
            border_style="red" if total_violations > 0 else "green",
        )
    )


if __name__ == "__main__":
    asyncio.run(replay_incident(INCIDENT_DIR, REPLAY_ITERATIONS, RUNS, CONCURRENCY_LIMIT))
