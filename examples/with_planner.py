"""
Пример 2: Planner генерирует Program из свободного текста.

Архитектура:
    user_input → Planner (1 LLM-вызов, недетерминирован)
               → Program (DSL)
               → ExecutionVM (детерминирован)
               → Trace
"""

import asyncio
from nano_vm import ExecutionVM, Planner
from nano_vm.adapters import LiteLLMAdapter


async def search(query: str) -> str:
    """Заглушка инструмента поиска."""
    return f"Результаты поиска по запросу '{query}': ..."


async def main():
    adapter = LiteLLMAdapter(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.0,
    )

    # Planner использует более умную модель для планирования
    planner_adapter = LiteLLMAdapter(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.2,
    )

    vm = ExecutionVM(
        llm=adapter,
        tools={"search": search},
    )

    planner = Planner(
        llm=planner_adapter,
        tools=["search"],
    )

    # Planner генерирует Program из свободного текста
    user_input = "Найди информацию о Python 3.13 и напиши краткое саммари"
    print(f"Запрос: {user_input}\n")

    program = await planner.generate(user_input)
    print(f"Сгенерированная программа: {program.name}")
    for step in program.steps:
        print(f"  [{step.id}] {step.type.value}")
    print()

    # ExecutionVM исполняет детерминированно
    trace = await vm.run(program, context={"user_input": user_input})

    print(f"Статус: {trace.status}")
    print(f"Результат: {trace.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
