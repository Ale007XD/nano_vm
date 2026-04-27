"""
llm-nano-vm benchmark script (v0.2.0)
--------------------------------------
Этот скрипт замеряет производительность параллельного выполнения шагов
через сетевой OpenRouter API.

Установка:
    pip install llm-nano-vm[litellm]

Запуск:
    export OPENROUTER_API_KEY="your_key_here"
    python benchmarks/stress_test_openrouter.py
"""

import asyncio
import contextlib
import logging
import os
import platform
import time

from nano_vm import ExecutionVM, Program, Step
from nano_vm.adapters.litellm_adapter import LiteLLMAdapter

# Глобальное подавление логов библиотек
logging.basicConfig(level=logging.CRITICAL)


async def main():
    # Проверка наличия ключа
    if "OPENROUTER_API_KEY" not in os.environ:
        print("❌ Ошибка: OPENROUTER_API_KEY не обнаружен в окружении.")
        return

    # Блок подавления лишнего вывода (litellm warnings, provider list и т.д.)
    with open(os.devnull, "w") as f, contextlib.redirect_stderr(f), contextlib.redirect_stdout(f):
        adapter = LiteLLMAdapter(model="openrouter/meta-llama/llama-3.1-8b-instruct")
        vm = ExecutionVM(llm=adapter)

        # Конфигурация: 20 параллельных сетевых вызовов
        CONCURRENCY = 20
        program = Program(
            steps=[
                Step(
                    id="stress_job",
                    type="parallel",
                    parallel_steps=[
                        Step(id=f"task_{i}", type="llm", prompt="Say 'OK'")
                        for i in range(CONCURRENCY)
                    ],
                    on_error="skip",
                )
            ]
        )

        t0 = time.perf_counter()
        trace = await vm.run(program)
        t1 = time.perf_counter()

    # Сбор данных после выполнения
    duration = t1 - t0
    outputs = getattr(trace, "final_output", {}) or {}
    success_count = len(outputs)
    rps = success_count / duration if duration > 0 else 0

    # Системная информация
    cores = os.cpu_count() or "Unknown"
    py_version = platform.python_version()
    is_android = any(x in os.environ for x in ["ANDROID_DATA", "ANDROID_ROOT"])
    sys_name = "Android" if is_android else platform.system()

    # Форматированный вывод для README/Скриншотов
    print("\n")
    print("🧬 Инициализация бенчмарка llm-nano-vm (OpenRouter Network)...")
    print(f"📱 Система: {sys_name} {platform.release()}")
    print(f"⚙️ Процессор: {platform.machine()} ({cores} cores)")
    print(f"🐍 Python: {py_version}")
    print(f"🧪 Тест: 1 запуск по {CONCURRENCY} параллельных шагов\n")
    print("═" * 54)
    print("📊 ФИНАЛЬНЫЕ МЕТРИКИ")
    print(f"⏱️ Общее время теста: {duration:.4f} s")
    print(f"🚀 Ср. время исполнения (1 Program): {duration:.2f} s")
    print(f"💎 Успешность: {success_count}/{CONCURRENCY}")
    print(f"📈 Производительность: {rps:.2f} RPS")
    print("═" * 54)
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
