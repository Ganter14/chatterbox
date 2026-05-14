#!/usr/bin/env bash
# Создаёт/обновляет baseline venv с upstream chatterbox-tts для регрессионных тестов.
#
# Переменные окружения (можно переопределить):
#   CHATTERBOX_BASELINE_VENV  — путь к venv (по умолчанию рядом с форком)
#   CHATTERBOX_UPSTREAM_REF   — тег/коммит/ветка апстрима (по умолчанию: master)
#
# Пример:
#   bash setup_baseline.sh
#   CHATTERBOX_UPSTREAM_REF=v0.1.2 bash setup_baseline.sh
#
# После выполнения экспортируйте путь к python для verify_regression.py:
#   export CHATTERBOX_REGRESSION_BASELINE_PYTHON=<BASELINE_VENV>/bin/python

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINE_VENV="${CHATTERBOX_BASELINE_VENV:-${SCRIPT_DIR}/../chatterbox-baseline-venv}"
UPSTREAM_REF="${CHATTERBOX_UPSTREAM_REF:-3f35dfc8fbe63e5b29793289dc68f1875bb317a5}"
UPSTREAM_URL="git+https://github.com/resemble-ai/chatterbox.git@${UPSTREAM_REF}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

echo "==> Baseline venv : ${BASELINE_VENV}"
echo "==> Upstream ref  : ${UPSTREAM_REF}"
echo ""

uv venv "${BASELINE_VENV}" --python 3.12

uv pip install \
  --python "${BASELINE_VENV}/bin/python" \
  --extra-index-url "${TORCH_INDEX}" \
  --index-strategy unsafe-best-match \
  "chatterbox-tts @ ${UPSTREAM_URL}"

# torch==2.6.0 (требование upstream) не поддерживает Blackwell (sm_120).
# Принудительно переустанавливаем torch/torchaudio из cu128-индекса с поддержкой sm_120.
echo "==> Переустановка torch/torchaudio с поддержкой sm_120 (Blackwell)..."
uv pip install \
  --python "${BASELINE_VENV}/bin/python" \
  --index-url "${TORCH_INDEX}" \
  --reinstall \
  "torch==2.10.0" "torchaudio==2.10.0"

PYTHON_PATH="$(cd "${BASELINE_VENV}/bin" && pwd)/python"

echo ""
echo "==> Готово. Для запуска регрессии:"
echo ""
echo "    export CHATTERBOX_REGRESSION_BASELINE_PYTHON=${PYTHON_PATH}"
echo "    uv run python verify_regression.py"
echo ""
