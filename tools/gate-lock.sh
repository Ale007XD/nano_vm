#!/usr/bin/env bash
# gate-lock.sh — (пере)генерировать tools/requirements-gate.lock.
# Запускать: после изменения зависимостей, или раз в месяц, или когда
# upstream-баг (как litellm<1.98.0) требует осознанного сдвига.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-3.12}"
rm -rf .venv
"python$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip wheel
pip install --quiet -e ".[dev]"
pip freeze --exclude-editable > tools/requirements-gate.lock
echo "OK → tools/requirements-gate.lock $(wc -l < tools/requirements-gate.lock) пакетов"
echo "Закоммить его вместе с изменением зависимостей."