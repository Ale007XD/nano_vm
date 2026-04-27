import asyncio
import platform
import sys
import time
from multiprocessing import cpu_count

# Пытаемся импортировать ядро, чтобы выдать понятную ошибку
try:
    from nano_vm import (
        ExecutionVM,
        Program,
        Step,
        StepType,
        TraceStatus,
    )
    from nano_vm.adapters.base import LLMAdapter
except ImportError:
    print("❌ Ошибка: nano-vm не установлена. Запустите 'pip install nano-vm'.")
    sys.exit(1)


# 1. Заглушка для LLM (Network-free)
class MockLLMAdapter(LLMAdapter):
    async def complete(self, messages: list[dict[str, str]]) -> tuple[str, dict]:
        # Минимальная задержка для имитации асинхронного перехода
        await asyncio.sleep(0.001)
        return "mock_response", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


# 2. Тестовый инструмент (CPU-bound имитация)
async def workload_tool(data_size: int = 100):
    # Имитируем небольшую работу с данными
    _ = "x" * data_size
    return "ok"


def create_stress_program(steps_count: int = 20) -> Program:
    """Генерирует цепочку последовательных шагов и финальное условие."""
    steps = [
        Step(
            id=f"step_{i}",
            type=StepType.TOOL,
            tool="workload",
            args={"data_size": 1000},
            output_key=f"data_{i}",
        )
        for i in range(steps_count)
    ]

    # Сложное условие для проверки Resolver и Eval
    steps.append(
        Step(
            id="check_logic",
            type=StepType.CONDITION,
            condition="'o' in '$step_0.output'",
            then=f"step_{steps_count - 1}",
            otherwise="step_0",
        )
    )

    return Program(name="performance_benchmark", steps=steps)


async def run_worker(vm: ExecutionVM, program: Program, context: dict):
    start = time.perf_counter()
    trace = await vm.run(program, context=context)
    duration = time.perf_counter() - start
    return trace.status == TraceStatus.SUCCESS, duration


async def main():
    # Настройки масштабирования
    # На мощных машинах можно поднять до 500+
    CONCURRENT_RUNS = 100
    STEPS_PER_PROG = 20

    # Сбор информации о системе
    sys_info = {
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": f"{platform.processor() or 'Unknown'} ({cpu_count()} cores)",
        "Python": sys.version.split()[0],
    }

    print("🧬 Инициализация бенчмарка nano-vm...")
    print(f"🖥️  Система: {sys_info['OS']}")
    print(f"⚙️  Процессор: {sys_info['CPU']}")
    print(f"🐍 Python: {sys_info['Python']}")
    print(f"🧪 Тест: {CONCURRENT_RUNS} запусков по {STEPS_PER_PROG} шагов\n")

    adapter = MockLLMAdapter()
    vm = ExecutionVM(llm=adapter, tools={"workload": workload_tool})
    program = create_stress_program(STEPS_PER_PROG)

    heavy_context = {f"ctx_key_{i}": "payload_data" for i in range(50)}

    start_time = time.perf_counter()

    # Запуск всех ВМ параллельно
    tasks = [run_worker(vm, program, heavy_context) for _ in range(CONCURRENT_RUNS)]
    results = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - start_time
    success_runs = sum(1 for r in results if r[0])
    avg_run_time = sum(r[1] for r in results) / CONCURRENT_RUNS

    print("═" * 45)
    print("📊 ФИНАЛЬНЫЕ МЕТРИКИ")
    print(f"⏱️  Общее время теста: {total_time:.4f} s")
    print(f"🚀 Ср. время исполнения (1 Program): {avg_run_time * 1000:.2f} ms")
    print(f"💎 Успешность: {success_runs}/{CONCURRENT_RUNS}")
    print(f"📈 Производительность: {CONCURRENT_RUNS / total_time:.2f} RPS")
    print("═" * 45)


if __name__ == "__main__":
    asyncio.run(main())
