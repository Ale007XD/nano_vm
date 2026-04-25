"""
Пример 1: Программа написана вручную (без Planner).
Детерминированное исполнение — никаких сюрпризов.
"""

import asyncio

from nano_vm import ExecutionVM, Program
from nano_vm.adapters import LiteLLMAdapter


async def main():
    # 1. Описать программу
    program = Program.from_dict(
        {
            "name": "summarize_and_translate",
            "description": "Суммаризировать текст и перевести на английский",
            "steps": [
                {
                    "id": "summarize",
                    "type": "llm",
                    "prompt": "Суммаризируй этот текст в 2 предложения: $text",
                    "output_key": "summary",
                },
                {
                    "id": "translate",
                    "type": "llm",
                    "prompt": "Translate to English: $summary",
                    "output_key": "translation",
                },
            ],
        }
    )

    # 2. Создать VM с адаптером
    vm = ExecutionVM(
        llm=LiteLLMAdapter(
            model="groq/llama-3.3-70b-versatile",
            fallbacks=["openrouter/llama-3.3-70b-instruct:free"],
            temperature=0.0,  # детерминированный вывод
        )
    )

    # 3. Запустить
    trace = await vm.run(
        program,
        context={"text": "Ваш длинный текст здесь..."},
    )

    print(f"Статус: {trace.status}")
    print(f"Результат: {trace.final_output}")
    print(f"Время: {trace.duration_ms:.0f}ms")

    for step in trace.steps:
        print(f"  [{step.step_id}] {step.status} → {str(step.output)[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
