"""
Общие примитивы для бенчмарков и регрессионных тестов.

ВАЖНО: этот модуль импортируется только из кода форка.
Воркеры _baseline_*_worker.py запускаются под upstream-venv,
где этот пакет недоступен — они намеренно самодостаточны.
"""
from __future__ import annotations

import contextlib
import gc
import random
import time
from typing import Any, Callable, Generator, Iterator

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Детерминизм
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Управление памятью
# ---------------------------------------------------------------------------

def cuda_sync() -> None:
    """Синхронизация GPU если доступен."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def free_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def release_models(*models: Any) -> None:
    for m in models:
        if m is not None:
            del m
    free_cuda_memory()


def waveform_cpu_clone(wav: torch.Tensor) -> torch.Tensor:
    """Копия волны на CPU с освобождением исходного тензора."""
    out = wav.detach().cpu().clone()
    del wav
    return out


# ---------------------------------------------------------------------------
# VRAM
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def capture_vram_peak() -> Generator[dict[str, int], None, None]:
    """
    Контекстный менеджер, фиксирующий пиковое потребление VRAM (байт).

    Использование:
        with capture_vram_peak() as vram:
            model.generate(...)
        print(vram["peak_bytes"])
    """
    info: dict[str, int] = {"peak_bytes": 0}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        yield info
    finally:
        if torch.cuda.is_available():
            info["peak_bytes"] = torch.cuda.max_memory_allocated()


# ---------------------------------------------------------------------------
# Таймер
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def Timer() -> Generator[dict[str, float], None, None]:
    """
    Контекстный менеджер для замера времени.

    Использование:
        with Timer() as t:
            model.generate(...)
        print(t["elapsed_ms"])
    """
    result: dict[str, float] = {"elapsed_ms": 0.0}
    cuda_sync()
    start = time.perf_counter()
    try:
        yield result
    finally:
        cuda_sync()
        result["elapsed_ms"] = (time.perf_counter() - start) * 1000.0


# ---------------------------------------------------------------------------
# TTFC
# ---------------------------------------------------------------------------

def measure_ttfc(generate_streaming_fn: Callable[..., Iterator[Any]], **kwargs: Any) -> float:
    """
    Измеряет Time To First Chunk (мс) для потокового генератора.

    Гарантирует cuda_sync() после получения первого чанка,
    чтобы зафиксировать реальное время завершения GPU-работы.

    Args:
        generate_streaming_fn: вызываемый объект, возвращающий итератор
                               (chunk, sr, timing). Обычно model.generate_streaming.
        **kwargs: аргументы, передаваемые в generate_streaming_fn.

    Returns:
        TTFC в миллисекундах.
    """
    cuda_sync()
    start = time.perf_counter()
    for _chunk, _sr, _timing in generate_streaming_fn(**kwargs):
        cuda_sync()
        ttfc_ms = (time.perf_counter() - start) * 1000.0
        break
    else:
        ttfc_ms = 0.0
    return ttfc_ms
