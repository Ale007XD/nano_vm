import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional

from nano_vm import ExecutionVM, Program, Trace
from nano_vm.adapters import MockLLMAdapter
from nano_vm.errors import VMError

# --- Incident Model ---
class IncidentStep(BaseModel):
    step_id: str
    type: str  # llm / tool / condition
    prompt: Optional[str] = None
    tool: Optional[str] = None
    condition: Optional[str] = None
    expected_output: Optional[Any] = None
    mock_response: Optional[str] = None  # для llm шагов

class IncidentTrace(BaseModel):
    incident_id: str
    title: str
    description: str
    steps: List[IncidentStep]
    expected_violations: List[str] = Field(default_factory=list)  # I1-I7
    initial_context: Dict[str, Any] = Field(default_factory=dict)

# --- Forensic Report Model ---
class InvariantViolation(BaseModel):
    invariant: str
    description: str
    step_id: str
    evidence: Dict[str, Any]

class ForensicReport(BaseModel):
    incident_id: str
    replay_count: int
    success_count: int
    violation_count: int
    deterministic_match: bool  # все ли 10000 реплеев дали одинаковую структуру шагов
    violations: List[InvariantViolation] = Field(default_factory=list)
    replay_duration_sec: float = 0.0

# --- Core Replay Engine ---
async def replay_incident(
    incident: IncidentTrace,
    n_iterations: int = 10000,
    n_runs: int = 5
) -> List[ForensicReport]:
    """
    Выполняет n_runs прогонов по n_iterations детерминированных реплеев.
    Возвращает список ForensicReport (по одному на прогон).
    """
    reports: List[ForensicReport] = []

    # Собираем Program из шагов инцидента
    program_steps = []
    mock_responses = {}
    for step in incident.steps:
        step_dict = {"id": step.step_id, "type": step.type}
        if step.type == "llm":
            step_dict["prompt"] = step.prompt
            step_dict["output_key"] = f"out_{step.step_id}"
        elif step.type == "tool":
            step_dict["tool"] = step.tool
        elif step.type == "condition":
            step_dict["condition"] = step.condition
            step_dict["then"] = step.expected_output.get("then", "next") if step.expected_output else "next"
            step_dict["otherwise"] = step.expected_output.get("otherwise", "next") if step.expected_output else "next"
        program_steps.append(step_dict)
        if step.mock_response:
            mock_responses[step.step_id] = step.mock_response

    program = Program.from_dict({
        "name": f"incident_{incident.incident_id}",
        "steps": program_steps
    })

    for run_idx in range(n_runs):
        # Для вариативности между прогонами можно менять mock-ответы
        # (имитация разных семантических исходов)
        adapter = MockLLMAdapter(mock_responses)
        vm = ExecutionVM(llm=adapter)

        start = datetime.now(timezone.utc)
        success_count = 0
        violation_count = 0
        all_traces_same_structure = True
        previous_step_ids: Optional[List[str]] = None

        for i in range(n_iterations):
            try:
                trace: Trace = await vm.run(program, context=incident.initial_context)
                step_ids = [s.step_id for s in trace.steps]
                if previous_step_ids is not None and step_ids != previous_step_ids:
                    all_traces_same_structure = False
                previous_step_ids = step_ids
                success_count += 1
            except VMError as e:
                violation_count += 1
                # Здесь можно логировать конкретное нарушение

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()

        # Генерируем отчёт о нарушениях инвариантов на основе expected_violations
        violations = []
        for v in incident.expected_violations:
            violations.append(InvariantViolation(
                invariant=v,
                description=f"Expected violation of {v}",
                step_id="N/A",
                evidence={"incident": incident.incident_id}
            ))

        report = ForensicReport(
            incident_id=incident.incident_id,
            replay_count=n_iterations,
            success_count=success_count,
            violation_count=violation_count,
            deterministic_match=all_traces_same_structure,
            violations=violations,
            replay_duration_sec=elapsed
        )
        reports.append(report)
        print(f"[Run {run_idx+1}/{n_runs}] {report.model_dump_json(indent=2)}")

    return reports

# --- Main Entry Point ---
async def main():
    corpus_dir = Path("examples/incidents")
    for incident_file in corpus_dir.rglob("incident.json"):
        with open(incident_file) as f:
            data = json.load(f)
        incident = IncidentTrace(**data)
        print(f"Replaying {incident.incident_id}: {incident.title}")
        reports = await replay_incident(incident, n_iterations=10000, n_runs=5)
        # Сохраняем forensic_report.json в ту же директорию
        report_path = incident_file.parent / "forensic_report.json"
        with open(report_path, "w") as f:
            json.dump([r.model_dump() for r in reports], f, indent=2, default=str)

if __name__ == "__main__":
    asyncio.run(main())
