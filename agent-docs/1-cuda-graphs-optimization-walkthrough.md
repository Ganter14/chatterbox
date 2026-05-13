# Оптимизация Chatterbox: CUDA Graphs и StaticCache

Мы успешно реализовали оптимизацию инференса для моделей T3 (Turbo и Multilingual), что позволило сократить время генерации речи в **2 раза**.

## Что было сделано

1.  **Интеграция StaticCache**: Перевели модели T3 (GPT-2 и Llama) на использование статического KV-кэша, что исключило постоянные переаллокации памяти.
2.  **Захват CUDA Graphs**: Реализовали класс `T3Graph`, который захватывает шаг декодирования одного токена в виде графа CUDA. Это сводит накладные расходы CPU к минимуму.
3.  **Поддержка мультиязычности**: Оптимизация адаптирована для работы с Classifier-Free Guidance (CFG), что позволило ускорить русский язык в `ChatterboxMultilingualTTS`.
4.  **Разделение веток**: Все работы выполнены в ветке `feat/cuda-graphs`, что позволяет легко сравнивать производительность с `master`.

## Результаты бенчмарков

Замеры проводились на GPU (CUDA) для полной цепочки синтеза (T3 + S3Gen + Vocoder).

| Модель | Базовая версия (s) | Оптимизированная (s) | Ускорение |
| :--- | :--- | :--- | :--- |
| **Chatterbox-Turbo (EN)** | 1.7870 | **0.8891** | **2.01x** |
| **Chatterbox-MTL (RU)** | 4.2120 | **1.9531** | **2.16x** |

> [!TIP]
> Скорость генерации токенов (T3 loop) выросла с **~47 токенов/сек** до **~105 токенов/сек**.

## Как использовать

Для включения оптимизации при загрузке модели используйте флаг `use_cuda_graph=True`:

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Инициализация с захватом графа
model = ChatterboxMultilingualTTS.from_pretrained(
    device="cuda", 
    use_cuda_graph=True
)

# Синтез будет происходить на ускоренном движке
wav = model.generate("Привет, это ускоренная версия Chatterbox!", language_id="ru")
```

## Технические детали

### T3Graph
Класс `T3Graph` управляет жизненным циклом графа:
- **Prefill**: После начальной обработки текста KV-кэш копируется из `DynamicCache` в `StaticCache`.
- **Decode**: Каждый последующий токен генерируется через `graph.replay()`, используя статические буферы для входов и выходов.
- **Masking**: Маски внимания предварительно рассчитаны для всех позиций до 2048, чтобы избежать вычислений на CPU внутри цикла.

### Файлы
- [t3_graph.py](file:///e:/Projects/chatterbox-work/chatterbox/src/chatterbox/models/t3/t3_graph.py) — Ядро оптимизации.
- [t3.py](file:///e:/Projects/chatterbox-work/chatterbox/src/chatterbox/models/t3/t3.py) — Поддержка кэша в модели.
- [benchmark_cuda.py](file:///e:/Projects/chatterbox-work/chatterbox/benchmark_cuda.py) — Скрипт для верификации.
