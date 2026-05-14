"""
Бенчмарк: «Насколько CUDA Graphs ускорили генерацию?»

Измеряет full-gen time, TTFC и VRAM для оптимизированной версии форка.
Если задана переменная CHATTERBOX_REGRESSION_BASELINE_PYTHON, запускает
upstream-воркер для получения baseline-таймингов и вычисляет speedup.

Запуск:
    uv run python benchmark_cuda.py
    uv run python -m benchmarks.cuda

Env:
    CHATTERBOX_REGRESSION_BASELINE_PYTHON — python из upstream-venv.
                                            Без него speedup не выводится.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks._model_loader import load_mtl_model, load_turbo_model
from benchmarks._utils import (
    Timer,
    capture_vram_peak,
    free_cuda_memory,
    measure_ttfc,
    release_models,
    set_seed,
)
from benchmarks._results import get_git_meta, save

ITERATIONS = 5

_WORKER = Path(__file__).parent / "_baseline_timing_worker.py"

BENCHMARK_CASES: list[dict[str, Any]] = [
    {
        "label": "Turbo-EN",
        "model_type": "turbo",
        "text": "The quick brown fox jumps over the lazy dog.",
        "language_id": None,
        "gen_kwargs": {},
    },
    {
        "label": "MTL-RU",
        "model_type": "mtl",
        "text": "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "language_id": "ru",
        "gen_kwargs": {"language_id": "ru"},
    },
]


# ---------------------------------------------------------------------------
# Baseline через upstream subprocess
# ---------------------------------------------------------------------------

def _run_baseline(model_type: str, text: str, language_id: str | None, device: str, seed: int) -> dict[str, float] | None:
    baseline_py = os.environ.get("CHATTERBOX_REGRESSION_BASELINE_PYTHON", "").strip()
    if not baseline_py:
        return None

    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        cmd = [
            baseline_py,
            str(_WORKER),
            "--output", tmp,
            "--device", device,
            "--seed", str(seed),
            "--model", model_type,
            "--iterations", str(ITERATIONS),
            "--text", text,
        ]
        if language_id:
            cmd += ["--language-id", language_id]
        if model_type == "mtl":
            mtl_dir = os.environ.get("CHATTERBOX_MTL_MODEL_DIR", "").strip()
            if mtl_dir:
                cmd += ["--mtl-model-dir", mtl_dir]
        elif model_type == "turbo":
            turbo_dir = os.environ.get("CHATTERBOX_TURBO_MODEL_DIR", "").strip()
            if turbo_dir:
                cmd += ["--turbo-model-dir", turbo_dir]

        proc = subprocess.run(cmd, cwd=str(_WORKER.parent), env=os.environ.copy())
        if proc.returncode != 0:
            print(f"[WARNING] baseline subprocess exited with {proc.returncode}", file=sys.stderr)
            return None

        with open(tmp) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Оптимизированный прогон (форк, use_cuda_graph=True)
# ---------------------------------------------------------------------------

def _run_optimized(case: dict[str, Any], device: str, seed: int) -> dict[str, float]:
    if case["model_type"] == "turbo":
        model = load_turbo_model(device, use_cuda_graph=True)
    else:
        model = load_mtl_model(device, use_cuda_graph=True)

    # warmup
    model.generate(case["text"], **case["gen_kwargs"])

    gen_times_ms: list[float] = []
    ttfc_times_ms: list[float] = []
    vram_peak_bytes: list[int] = []

    for _ in range(ITERATIONS):
        set_seed(seed)
        with capture_vram_peak() as vram, Timer() as t:
            model.generate(case["text"], **case["gen_kwargs"])
        gen_times_ms.append(t["elapsed_ms"])
        vram_peak_bytes.append(vram["peak_bytes"])

        ttfc_ms = measure_ttfc(model.generate_streaming, **case["gen_kwargs"], text=case["text"])
        ttfc_times_ms.append(ttfc_ms)

    release_models(model)

    return {
        "gen_time_avg_ms": float(np.mean(gen_times_ms)),
        "gen_time_std_ms": float(np.std(gen_times_ms)),
        "ttfc_avg_ms": float(np.mean(ttfc_times_ms)),
        "ttfc_p95_ms": float(np.percentile(ttfc_times_ms, 95)),
        "vram_peak_mb": round(max(vram_peak_bytes) / 1024 / 1024, 1),
    }


# ---------------------------------------------------------------------------
# Вывод таблицы
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict[str, Any]]) -> None:
    git = get_git_meta()
    branch_info = f" branch={git['branch']} commit={git['commit']}" if git["commit"] else ""
    print(f"\n{'='*90}")
    print(f"CUDA GRAPHS BENCHMARK{branch_info}")
    print(f"{'='*90}")

    has_baseline = any(r.get("baseline") for r in rows)
    if has_baseline:
        header = f"{'Model':<12} | {'Opt gen(ms)':<13} | {'Base gen(ms)':<14} | {'Speedup':<9} | {'TTFC avg(ms)':<14} | {'TTFC p95(ms)':<13} | {'VRAM(MB)'}"
    else:
        header = f"{'Model':<12} | {'Opt gen(ms)':<13} | {'TTFC avg(ms)':<14} | {'TTFC p95(ms)':<13} | {'VRAM(MB)'}"
    print(header)
    print("-" * len(header))

    for r in rows:
        opt = r["optimized"]
        if has_baseline and r.get("baseline"):
            base = r["baseline"]
            speedup = base["gen_time_avg_ms"] / opt["gen_time_avg_ms"]
            print(
                f"{r['label']:<12} | {opt['gen_time_avg_ms']:<13.1f} | "
                f"{base['gen_time_avg_ms']:<14.1f} | {speedup:<9.2f}x | "
                f"{opt['ttfc_avg_ms']:<14.1f} | {opt['ttfc_p95_ms']:<13.1f} | "
                f"{opt['vram_peak_mb']}"
            )
        else:
            print(
                f"{r['label']:<12} | {opt['gen_time_avg_ms']:<13.1f} | "
                f"{opt['ttfc_avg_ms']:<14.1f} | {opt['ttfc_p95_ms']:<13.1f} | "
                f"{opt['vram_peak_mb']}"
            )
    print("=" * len(header))

    if not has_baseline:
        print("NOTE: задайте CHATTERBOX_REGRESSION_BASELINE_PYTHON для вывода speedup.")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA недоступна, результаты не репрезентативны для GPU-оптимизаций.")

    seed = 42
    rows: list[dict[str, Any]] = []
    all_results: dict[str, Any] = {}

    for case in BENCHMARK_CASES:
        print(f"\n[{case['label']}] Запуск baseline (upstream)...")
        baseline = _run_baseline(case["model_type"], case["text"], case["language_id"], device, seed)
        if baseline is None:
            print(f"[{case['label']}] Baseline пропущен (CHATTERBOX_REGRESSION_BASELINE_PYTHON не задан).")

        print(f"[{case['label']}] Запуск optimized (форк, use_cuda_graph=True)...")
        optimized = _run_optimized(case, device, seed)
        free_cuda_memory()

        row: dict[str, Any] = {"label": case["label"], "optimized": optimized}
        if baseline:
            row["baseline"] = baseline
        rows.append(row)

        all_results[case["label"]] = row

    _print_table(rows)

    out_path = save("cuda", all_results)
    print(f"\nРезультаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
