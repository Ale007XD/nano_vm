#!/usr/bin/env bash
# gate.sh — канонический гейт nano-vm на чистом окружении.
# Идиотент: пересоздаёт .venv с нуля на каждом запуске.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-3.12}"
VENV=".venv"
REQ="tools/requirements-gate.txt"
LOCK="tools/requirements-gate.lock"

# --- 0. Python нужной версии -------------------------------------------------
if ! command -v "python$PY" >/dev/null 2>&1; then
  echo "python$PY не найден. Ubuntu/WSL: sudo apt install python3.12 python3.12-venv"
  exit 1
fi

# --- 1. Чистый venv ----------------------------------------------------------
rm -rf "$VENV"
"python$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip wheel

# --- 2. Зависимости: pinned-лок, обновление только по команде ----------------
if [[ -f "$LOCK" ]]; then
  pip install --quiet -r "$LOCK"
else
  pip install --quiet -e ".[dev]"
  echo "⚠ $LOCK отсутствует — установлен плавающий [dev]. Сгенерируй: tools/gate-lock.sh"
fi

pip install --quiet pytest-timeout  # CI-parity: в [dev] его нет, CI ставит отдельно
pip install --quiet -e "."
# (первый вызов с локом ставит и litellm==1.97.x, и пакет из локальной директории)

# --- 3. Гейт -------------------------------------------------------------------
echo "== commit =="
echo "$(git rev-parse --short HEAD) ($(git status --porcelain | wc -l) dirty)"

echo "== import smoke =="
python - <<'PY'
import nano_vm
from nano_vm import ExecutionVM, Program, ProgramValidator, TraceAnalyzer
print("import OK ·", "v" + nano_vm.__version__)
PY

echo "== ruff =="
ruff check .

echo "== mypy =="
mypy nano_vm/

echo "== pytest =="
pytest tests/ --timeout=10 -q

echo "== GATE GREEN =="