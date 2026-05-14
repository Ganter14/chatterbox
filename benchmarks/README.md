# benchmarks/

Инфраструктура измерения производительности и регрессионного тестирования для форка Chatterbox TTS.

## Назначение

Модуль отвечает на два независимых вопроса:

| Скрипт / модуль | Вопрос |
|---|---|
| `benchmarks/cuda.py` | Насколько CUDA Graphs ускорили генерацию? |
| `benchmarks/rtf.py` | Работает ли стриминг в реальном времени (RTF < 1.0)? |
| `verify_regression.py` (корень) | Не сломала ли оптимизация аудиовыход? (MSE) |

---

## Структура файлов

```
benchmarks/
  __init__.py
  _utils.py                    — общие примитивы: set_seed, free_cuda, Timer,
                                  capture_vram_peak, measure_ttfc
  _results.py                  — JSON-персистентность: save(), load_history(),
                                  git-метаданные, GPU info
  _baseline_waveform_worker.py — subprocess-воркер под upstream-venv;
                                  сохраняет .npy с волной (для verify_regression.py)
  _baseline_timing_worker.py   — subprocess-воркер под upstream-venv;
                                  сохраняет timing JSON (для benchmarks/cuda.py)
  cuda.py                      — бенчмарк full-gen time, TTFC, VRAM, опц. speedup
  rtf.py                       — бенчмарк RTF, TTFC p50/p95, avg chunk decode
  results/                     — сохранённые JSON-прогоны (в .gitignore)
```

---

## Запуск

Из корня репозитория (через `uv run`):

```bash
# Бенчмарк CUDA Graphs (full-gen time, TTFC, VRAM)
uv run python benchmark_cuda.py

# Бенчмарк RTF стриминга
uv run python benchmark_rtf.py

# Регрессия: MSE форка vs upstream baseline
uv run python verify_regression.py

# Отдельная фаза регрессии
uv run python verify_regression.py mtl_ru
uv run python verify_regression.py streaming

# Все фазы в одном процессе (для отладки)
CHATTERBOX_REGRESSION_INPROC=1 uv run python verify_regression.py
```

---

## Переменные окружения

| Переменная | Кто использует | Описание |
|---|---|---|
| `CHATTERBOX_MTL_MODEL_DIR` | все бенчмарки, `verify_regression.py` | Путь к локальной директории весов MTL-модели (`ResembleAI/chatterbox`). Если задан — HuggingFace не используется. |
| `CHATTERBOX_TURBO_MODEL_DIR` | `benchmark_cuda.py` | Путь к локальной директории весов Turbo-модели (`ResembleAI/chatterbox-turbo`). |
| `CHATTERBOX_REGRESSION_BASELINE_PYTHON` | `verify_regression.py`, `benchmark_cuda.py` | Путь к `python` из upstream-venv. **Обязателен** для фазы `upstream_quality` и для speedup-колонки в CUDA-бенчмарке. |
| `CHATTERBOX_REGRESSION_INPROC` | `verify_regression.py` | Если `1` — все фазы в одном процессе (без очистки VRAM между фазами). Только для отладки. |
| `CHATTERBOX_BASELINE_VENV` | `setup_baseline.sh` | Путь к venv baseline (по умолчанию `../chatterbox-baseline-venv`). |
| `CHATTERBOX_UPSTREAM_REF` | `setup_baseline.sh` | Тег/коммит/ветка upstream (по умолчанию `master`). |

### Офлайн-режим: загрузка весов локально

При нестабильном или заблокированном доступе к Hugging Face загрузите веса один раз:

```bash
# Загрузить обе модели (~5–8 ГБ) в ~/.local/share/chatterbox-models/
uv run python download_models.py

# Или в произвольную директорию
uv run python download_models.py --output-dir /data/models

# Только MTL или только Turbo
uv run python download_models.py --model mtl
uv run python download_models.py --model turbo

# Скрипт выведет строки export — добавьте их в ~/.bashrc / ~/.zshrc
export CHATTERBOX_MTL_MODEL_DIR=~/.local/share/chatterbox-models/chatterbox
export CHATTERBOX_TURBO_MODEL_DIR=~/.local/share/chatterbox-models/chatterbox-turbo
```

После этого все бенчмарки и `verify_regression.py` автоматически используют локальные веса.
Для upstream-baseline переменные `CHATTERBOX_MTL_MODEL_DIR` / `CHATTERBOX_TURBO_MODEL_DIR`
также прокидываются в воркер-subprocess.

### Подготовка upstream baseline одной командой

```bash
bash setup_baseline.sh
export CHATTERBOX_REGRESSION_BASELINE_PYTHON=<путь из вывода скрипта>
```

---

## Сохранение результатов

Каждый прогон `benchmark_cuda.py` и `benchmark_rtf.py` автоматически сохраняет JSON в `benchmarks/results/`:

```
benchmarks/results/
  2026-05-14T23-58-00_abc1234_cuda.json
  2026-05-14T23-59-00_abc1234_rtf.json
```

Структура файла:

```json
{
  "timestamp": "2026-05-14T23:58:00",
  "git_commit": "abc1234",
  "git_branch": "feat/cuda-graphs",
  "gpu": "NVIDIA GeForce RTX 3090",
  "results": { ... }
}
```

Для загрузки истории прогонов в скрипте:

```python
from benchmarks._results import load_history

for record in load_history("rtf"):
    print(record["git_commit"], record["results"]["Short"]["rtf_avg"])
```

---

## Архитектурное ограничение воркеров

Файлы `_baseline_waveform_worker.py` и `_baseline_timing_worker.py` запускаются
**под интерпретатором из upstream-venv** (`CHATTERBOX_REGRESSION_BASELINE_PYTHON`).
В этом окружении пакет `benchmarks` форка физически отсутствует.

**Поэтому они намеренно не импортируют ничего из форка**, включая `benchmarks._utils`.
Дублирование `set_seed` и `gc`-логики в этих файлах — осознанный компромисс,
а не техдолг. Не нужно «исправлять» это дублирование.

---

## Добавление нового бенчмарка

1. Создать `benchmarks/my_bench.py` — использовать примитивы из `_utils.py`.
2. Сохранять результаты через `from benchmarks._results import save; save("my_bench", data)`.
3. Добавить корневой wrapper `benchmark_my_bench.py` из двух строк.
4. Задокументировать в этом README.
