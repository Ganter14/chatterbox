# AGENTS.md — Контекст ИИ-разработчика (Fork: Chatterbox)

## Краткое описание проекта

Этот репозиторий — форк оригинального **Chatterbox TTS** от Resemble AI.
Проект предназначен для высокопроизводительного, мультиязычного синтеза речи (TTS) и конверсии голоса с ультранизкой задержкой (Streaming Mode).
Основной фокус форка — глубокие оптимизации инференса на GPU NVIDIA для использования моделей в production-агентах реального времени.

---

## Окружение и Hardware

### Требования к среде
- **GPU**: NVIDIA (архитектура Ampere и новее рекомендуется для стабильных CUDA Graphs). Минимум 8 ГБ VRAM (12+ ГБ для батчей > 2).
- **CUDA**: Строго 12.8 (совместимость с жестко закрепленными версиями `nvidia-*` в `pyproject.toml`).
- **Python**: 3.12+

### Быстрый старт
Для подготовки окружения используйте `uv`:
```bash
uv sync --frozen
```
Это гарантирует установку зависимостей в соответствии с `uv.lock` без попыток обновления версий.

---

## Карта кода

Пути от корня репозитория `chatterbox/`:

| Область | Пути |
|--------|------|
| Публичный API TTS | `src/chatterbox/tts.py` (`ChatterboxTTS`), `src/chatterbox/tts_turbo.py` (`ChatterboxTurboTTS`), `src/chatterbox/mtl_tts.py` (`ChatterboxMultilingualTTS`) |
| T3 и оптимизации | `src/chatterbox/models/t3/t3.py`, `src/chatterbox/models/t3/t3_graph.py`, `src/chatterbox/models/t3/inference/t3_hf_backend.py` |
| Аудио-декодер (токены → звук) | `src/chatterbox/models/s3gen/` (`s3gen.py`, `flow*.py`, `hifigan.py` и др.); стриминг: `src/chatterbox/models/s3gen/s3gen_streamer.py` (`S3GenStreamer`) |
| Токенизация текста | `src/chatterbox/models/s3tokenizer/` |
| Voice conversion | `src/chatterbox/vc.py`, `src/chatterbox/models/voice_encoder/` |
| Примеры и UI | `example_streaming.py`, `gradio_*.py`, `multilingual_app.py` в корне репозитория |

**`T3Graph` и batch:** при `use_cuda_graph=True` для `ChatterboxMultilingualTTS` в `mtl_tts.py` создаётся `T3Graph(..., batch_size=2)` (ветка CFG: условный и безусловный forward в одном батче). Для `ChatterboxTurboTTS` в `tts_turbo.py` — `batch_size=1`.

---

## Архитектура инференса (после оптимизаций)

**Пайплайн данных:** текст проходит лингвистическую и аудио-токенизацию (S3Tokenizer и код в `models/s3tokenizer`). Далее **T3** — авторегрессивная модель на базе GPT-2 / Llama, выдающая аудио-токены; интеграция с Hugging Face реализована в `t3_hf_backend`. Затем **S3Gen** (flow-matching, HiFT-GAN и связанные модули) преобразует токены в волновую форму.

**Оптимизированный путь T3:** на prefill используется стандартный проход с кэшем; затем KV переносится в **`StaticCache`** из `transformers`. На шаге decode одного токена при `use_cuda_graph=True` выполняется **`T3Graph`**: заранее захваченный CUDA graph воспроизводится через `replay`, статические буферы для входов/выходов; маски внимания и позиции подготовлены так, чтобы не выполнять тяжёлую логику на CPU внутри горячего цикла (подробности — в walkthrough по CUDA graphs).

**Streaming:** на стороне T3 используются генераторы вроде `inference_stream` в `t3.py` — токены отдаются по одному, при этом decode-шаг совместим с тем же graph-путём, что и нестриминговый режим. Класс **`S3GenStreamer`** накапливает декод с окном и lookahead, чтобы чанки аудио стыковались без щелчков (см. walkthrough по streaming).

**CFG (мультиязычность):** ускорение завязано не на «два независимых вызова модели подряд», а на совместной обработке условного и безусловного прохода в рамках конфигурации `T3Graph` с `batch_size=2` в `ChatterboxMultilingualTTS`.

```mermaid
flowchart LR
  textNode[Text] --> tokNode[Tokenization]
  tokNode --> t3Node[T3]
  t3Node --> s3Node[S3Gen]
  s3Node --> wavNode[Waveform]
```

Ниже в разделе «Core-оптимизации» перечислены инварианты, которые нельзя нарушать при правках; этот раздел задаёт *где* в коде это живёт.

---

## Документация для разработчиков (walkthrough)

Читать **до** правок в соответствующей подсистеме:

- [`agent-docs/cuda-graphs-optimization-walkthrough.md`](agent-docs/cuda-graphs-optimization-walkthrough.md) — `StaticCache`, жизненный цикл `T3Graph` (prefill → копирование KV → decode через graph), маскирование, **эталонные таблицы** времени полной генерации (Turbo EN и MTL RU, baseline vs оптимизированная версия).
- [`agent-docs/streaming-walkthrough.md`](agent-docs/streaming-walkthrough.md) — методы `generate_streaming`, TTFC, связка T3-streaming + `S3GenStreamer`, проверки согласованности с полной генерацией.
- [`agent-docs/streaming-performance-optimization.md`](agent-docs/streaming-performance-optimization.md) — кеширование промпта (prompt caching), состояние вокодера (stateful vocoder) и отключение водяных знаков для достижения RTF < 1.0.

Перед изменениями в `t3_graph.py`, цикле decode или кэше — cuda-graphs walkthrough. Перед изменениями в `generate_streaming`, `S3GenStreamer` или chunked API — streaming walkthrough.

---

## Менеджер пакетов: uv

В проекте зависимости задаются в [`pyproject.toml`](pyproject.toml) и предполагается использование **[uv](https://github.com/astral-sh/uv)**.

- В `pyproject.toml` настроены `[[tool.uv.index]]` с именем `pytorch-cu128` (URL `https://download.pytorch.org/whl/cu128`, `explicit = true`) и `[tool.uv.sources]`: пакеты `torch` и `torchaudio` берутся **только** с этого индекса.
- Рекомендуемый рабочий цикл из каталога `chatterbox/`: `uv sync`. Запуск скриптов верификации: `uv run python verify_regression.py`.

**Правило для ИИ-агента:** не предлагать замену uv на «проще через pip» без веской причины; не обходить явный индекс PyTorch; новые зависимости добавлять через `pyproject.toml` и политику uv, а не размытыми командами вроде `pip install <latest>`.

---

## Версии в pyproject — не трогать

Следующие и **связанные с ними жёсткие пины** в [`pyproject.toml`](pyproject.toml) задают совместимость CUDA graphs, API `transformers`, ABI и воспроизводимость инференса:

- `requires-python = ">=3.12"`
- `torch==2.10.0`, `torchaudio==2.10.0` (согласованная пара)
- `transformers==4.57.3`
- CUDA 12.8: `nvidia-cuda-runtime-cu12==12.8.90`, `nvidia-cublas-cu12==12.8.4.1`

**Правило:** эти версии **не изменять вообще** без **явного, недвусмысленного запроса пользователя**. Агент не должен сам «обновлять» зависимости. Если пользователь **явно** потребовал смену версий — только тогда, и обязательно с полным прогоном верификации.

---

## Верификация изменений

Запуск из каталога `chatterbox/` (предпочтительно через `uv run python …`).

1. **`verify_regression.py`**: Три независимые фазы, каждая — отдельный подпроцесс (сброс VRAM).

   | Фаза | Что проверяет | Метрика / Порог |
   |------|--------------|-----------------|
   | `mtl_ru` | Паритет CUDA graphs: `use_cuda_graph=False` vs `True` | MSE < 1e-3 |
   | `upstream_quality` | Деградация качества vs upstream resemble-ai/chatterbox | SQUIM-STOI ≥ 0.55 (fork и не хуже base на 0.1); WER fork ≤ max(35%, WER base + 15 п.п.); MCD — информационно |
   | `streaming` | Качество стримингового аудио | SQUIM-STOI ≥ 0.55; длина в пределах ±5% от полной генерации; MSE — информационно |

   `upstream_quality` пропускается (SKIP) если `CHATTERBOX_REGRESSION_BASELINE_PYTHON` не задан.
   Подготовка upstream-venv: `bash setup_baseline.sh && export CHATTERBOX_REGRESSION_BASELINE_PYTHON=<путь>`.

   Отладка в одном процессе: `CHATTERBOX_REGRESSION_INPROC=1 uv run python verify_regression.py`.
   Одна фаза вручную: `uv run python verify_regression.py mtl_ru` (или `upstream_quality`, `streaming`).

   > **Примечание по ODE-шуму**: форк намеренно использует `z = torch.zeros_like(mu)` вместо `randn`
   > для детерминированности CUDA graphs. Из-за этого побитовое совпадение с upstream невозможно.
   > **MCD** без DTW-выравнивания между системами с разными ODE-стартами всегда 20–40 dB независимо
   > от качества — метрика выводится информационно и не входит в pass/fail.
   > **WER** проверяется относительно: fork ≤ max(35%, WER baseline + 15 п.п.) — это нейтрализует
   > ошибки Whisper на сложных русских словах (одинаковые у обоих систем).
   >
   > **Примечание по streaming MSE**: `S3GenStreamer` использует prompt caching (раздельное кодирование
   > промпта для RTF) и stateful vocoder (`last_s`, для бесшовных переходов между чанками). Из-за этих
   > оптимизаций MSE между streaming и `generate()` ≈ 0.026 — норма, не регрессия. Критерием качества
   > стриминга служит SQUIM-STOI ≥ 0.55 и совпадение длины ±5%.

2. **`benchmark_cuda.py`**: Замер времени `generate` и TTFC. Не допускать относительной просадки FPS.
3. **`verify_update.py`**: Сквозная проверка (TTS + Whisper). Семантическое соответствие текста.

---

## Внедрённые Core-оптимизации (ВАЖНО)

При генерации или изменении кода инференса ИИ обязан строго следовать этим правилам:

1. **Static KV-Cache (`StaticCache`)**:
   * **Инвариант:** Исключить динамические аллокации памяти во время генерации.
   * **Ограничение:** `StaticCache` жестко аллоцирует память под `max_seq_len` при создании. Увеличение этого параметра повышает потребление VRAM.

2. **Захват CUDA Graphs (`T3Graph`)**:
   * **Инвариант:** Внутри цикла генерации (между `graph.replay()`) запрещены любые операции, вызывающие синхронизацию CPU/GPU: `.item()`, `.tolist()`, `print()`, условия на значениях тензоров.
   * **Ограничение:** Любое изменение архитектуры `tfmr` (T3) требует обновления логики в `T3Graph._decode_step()`.

3. **Classifier-Free Guidance (CFG)**:
   * **Инвариант:** Для Multilingual использовать `batch_size=2` в графах. Не разделять проходы на два графа, это убьет производительность.

4. **Streaming Mode**:
   * **Инвариант:** Сохранять обратную совместимость с `generate_streaming`. Новые аргументы API должны иметь значения по умолчанию.

---

## Чек-лист для ИИ-агента (перед PR)

> [!IMPORTANT]
> Перед выполнением любых правок в логике инференса убедитесь, что:

- [ ] Вы не изменили версии в `pyproject.toml`.
- [ ] Изменения в модели T3 отражены в `src/chatterbox/models/t3/t3_graph.py` (статические буферы, размерность логитов).
- [ ] В "горячем цикле" генерации не появилось вызовов, блокирующих CUDA Graph.
- [ ] Прогнан `uv run python verify_regression.py`: фаза `mtl_ru` (MSE < 1e-3) и `streaming` прошли; `upstream_quality` не упала ниже порогов (SQUIM-STOI / WER относительный).
- [ ] `StaticCache` корректно сбрасывается (`.reset()`) перед новым prefill.

---

## Troubleshooting (Типичные проблемы)

- **"Illegal memory access" после изменения T3**: Скорее всего, размер буфера `logits_buf` или `input_buf` в `T3Graph` больше не соответствует выходным тензорам модели.
- **Производительность упала до уровня baseline**: Проверьте, не вызывается ли `capture()` на каждом шаге. Граф должен захватываться один раз при инициализации.
- **`uv sync` падает с ошибкой индекса**: Убедитесь, что вы не удалили `[[tool.uv.index]]` из `pyproject.toml`. PyTorch для CUDA 12.8 берется с отдельного URL.
- **OOM (Out of Memory)**: Проверьте `max_seq_len` в конфиге. Для `batch_size=2` и длинных контекстов потребление памяти растет линейно.
