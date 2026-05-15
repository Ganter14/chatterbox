"""
Бенчмарк: «Работает ли стриминг в реальном времени?»

Измеряет RTF, TTFC (median, P95) и средний decode-time чанка
для трёх длин фраз. Требует RTF < 1.0 для production-готовности.

Запуск:
    uv run python benchmark_rtf.py
    uv run python -m benchmarks.rtf
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from benchmarks._model_loader import load_mtl_model
from benchmarks._utils import (
    capture_vram_peak,
    cuda_sync,
    free_cuda_memory,
    release_models,
    set_seed,
)
from benchmarks._results import get_git_meta, save

ITERATIONS = 5

TEST_PHRASES: list[tuple[str, str, str]] = [
    (
        "Short",
        "Солнце светит ярко.",
        "ru",
    ),
    (
        "Medium",
        "В чащах юга жил бы цитрус? Да, но фальшивый экземпляр!",
        "ru",
    ),
    (
        "Long",
        (
            "Привет! Как твои дела? Надеюсь, у тебя всё отлично и ты готов к новым "
            "свершениям в области искусственного интеллекта и синтеза речи."
        ),
        "ru",
    ),
]


# ---------------------------------------------------------------------------
# Замер одной фразы
# ---------------------------------------------------------------------------

def _measure_phrase(
    model: ChatterboxMultilingualTTS,
    text: str,
    language_id: str,
    chunk_size: int,
    seed: int,
) -> dict[str, Any]:
    """Полный стриминг-прогон, ITERATIONS раз. TTFC берётся из времени до первого
    yield генератора (а не из отдельного measure_ttfc). Это важно: отдельный
    pre-run, прерванный break после первого чанка, не закрывает background
    daemon-threads `t3_worker` / S3Gen-воркера, и они продолжают съедать GPU
    в следующей итерации, искусственно завышая RTF.
    """
    gen_kwargs = dict(language_id=language_id, chunk_size=chunk_size, skip_watermark=True)

    ttfc_samples: list[float] = []
    rtf_samples: list[float] = []
    decode_ms_all: list[float] = []
    vram_peak_bytes: list[int] = []

    for _ in range(ITERATIONS):
        set_seed(seed)
        chunks_info: list[dict] = []
        total_samples = 0
        sr = 22050
        ttfc_ms: float | None = None

        with capture_vram_peak() as vram:
            cuda_sync()
            t0 = time.perf_counter()
            for chunk, sr, timing in model.generate_streaming(text=text, **gen_kwargs):
                if ttfc_ms is None:
                    cuda_sync()
                    ttfc_ms = (time.perf_counter() - t0) * 1000.0
                total_samples += len(chunk)
                chunks_info.append(timing)
            cuda_sync()
            gen_time = time.perf_counter() - t0
        vram_peak_bytes.append(vram["peak_bytes"])

        audio_duration = total_samples / sr
        rtf_samples.append(gen_time / audio_duration if audio_duration > 0 else 0.0)
        if ttfc_ms is not None:
            ttfc_samples.append(ttfc_ms)
        decode_ms_all.extend(c["decode_ms"] for c in chunks_info if "decode_ms" in c)

        # Дать background daemon-threads streaming pipeline спокойно завершиться
        # перед следующей итерацией, иначе они конкурируют за GPU и пессимизируют
        # как RTF, так и TTFC следующего прогона.
        free_cuda_memory()

    return {
        "rtf_avg": float(np.mean(rtf_samples)),
        "rtf_pass": bool(np.mean(rtf_samples) < 1.0),
        "ttfc_p50_ms": float(np.percentile(ttfc_samples, 50)) if ttfc_samples else 0.0,
        "ttfc_p95_ms": float(np.percentile(ttfc_samples, 95)) if ttfc_samples else 0.0,
        "avg_chunk_decode_ms": float(np.mean(decode_ms_all)) if decode_ms_all else 0.0,
        "vram_peak_mb": round(max(vram_peak_bytes) / 1024 / 1024, 1),
        "iterations": ITERATIONS,
    }


# ---------------------------------------------------------------------------
# Вывод таблицы
# ---------------------------------------------------------------------------

def _print_table(rows: list[tuple[str, dict[str, Any]]]) -> None:
    git = get_git_meta()
    branch_info = f" branch={git['branch']} commit={git['commit']}" if git["commit"] else ""
    print(f"\n{'='*85}")
    print(f"RTF STREAMING BENCHMARK{branch_info}")
    print(f"{'='*85}")
    header = (
        f"{'Phrase':<8} | {'RTF':<8} | {'Pass':<5} | "
        f"{'TTFC p50(ms)':<13} | {'TTFC p95(ms)':<13} | "
        f"{'Chunk(ms)':<10} | {'VRAM(MB)'}"
    )
    print(header)
    print("-" * len(header))
    for label, r in rows:
        status = "OK" if r["rtf_pass"] else "FAIL"
        print(
            f"{label:<8} | {r['rtf_avg']:<8.4f} | {status:<5} | "
            f"{r['ttfc_p50_ms']:<13.1f} | {r['ttfc_p95_ms']:<13.1f} | "
            f"{r['avg_chunk_decode_ms']:<10.2f} | {r['vram_peak_mb']}"
        )
    print("=" * len(header))
    all_pass = all(r["rtf_pass"] for _, r in rows)
    verdict = "ALL PASSED (RTF < 1.0)" if all_pass else "SOME FAILED (RTF >= 1.0)"
    print(f"VERDICT: {verdict}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA недоступна, RTF < 1.0 маловероятен.")

    chunk_size = 12
    seed = 42

    print("Загрузка модели...")
    model = load_mtl_model(device, use_cuda_graph=True)

    print("Warmup...")
    model.generate("Warmup.", language_id="ru", skip_watermark=True)

    rows: list[tuple[str, dict[str, Any]]] = []
    all_results: dict[str, Any] = {}

    for label, text, lang in TEST_PHRASES:
        print(f"\n[{label}] '{text[:50]}...'")
        result = _measure_phrase(model, text, lang, chunk_size, seed)
        rows.append((label, result))
        all_results[label] = result
        free_cuda_memory()

    release_models(model)

    _print_table(rows)

    out_path = save("rtf", all_results)
    print(f"\nРезультаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
