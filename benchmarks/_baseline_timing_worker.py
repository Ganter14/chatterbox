"""
Subprocess-воркер для замера времени генерации upstream chatterbox-tts.

ВАЖНО — АРХИТЕКТУРНОЕ ОГРАНИЧЕНИЕ:
    Этот файл запускается под интерпретатором из CHATTERBOX_REGRESSION_BASELINE_PYTHON
    (отдельный venv с upstream chatterbox-tts). Пакет benchmarks форка там недоступен,
    поэтому этот модуль намеренно самодостаточен и НЕ импортирует ничего из форка.
    Дублирование set_seed / gc-логики здесь — осознанный компромисс.

Вызывается из benchmarks/cuda.py через subprocess.
Сохраняет результаты в JSON-файл по пути --output.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time

import numpy as np
import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_model(model_type: str, device: str, mtl_model_dir: str | None, turbo_model_dir: str | None):
    if model_type == "turbo":
        if turbo_model_dir:
            return ChatterboxTurboTTS.from_local(turbo_model_dir, device=device)
        return ChatterboxTurboTTS.from_pretrained(device=device)
    if model_type == "mtl":
        if mtl_model_dir:
            return ChatterboxMultilingualTTS.from_local(mtl_model_dir, device=device)
        return ChatterboxMultilingualTTS.from_pretrained(device=device)
    raise ValueError(f"Unknown model type: {model_type!r}. Expected 'turbo' or 'mtl'.")


def _generate(model, model_type: str, text: str, language_id: str | None):
    kwargs: dict = {}
    if model_type == "mtl" and language_id:
        kwargs["language_id"] = language_id
    return model.generate(text, **kwargs)


def _generate_streaming(model, model_type: str, text: str, language_id: str | None):
    kwargs: dict = {}
    if model_type == "mtl" and language_id:
        kwargs["language_id"] = language_id
    return model.generate_streaming(text, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upstream baseline timing worker for benchmarks/cuda.py"
    )
    parser.add_argument("--output", required=True, help="Путь к выходному .json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="mtl", choices=["turbo", "mtl"])
    parser.add_argument("--language-id", default="ru")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--mtl-model-dir",
        default=None,
        help="Путь к локальной директории весов MTL-модели (обходит HuggingFace)",
    )
    parser.add_argument(
        "--turbo-model-dir",
        default=None,
        help="Путь к локальной директории весов Turbo-модели (обходит HuggingFace)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    model = _load_model(args.model, args.device, args.mtl_model_dir, args.turbo_model_dir)

    # warmup
    _generate(model, args.model, args.text, args.language_id)
    cuda_sync()

    gen_times: list[float] = []
    ttfc_times: list[float] = []

    for _ in range(args.iterations):
        # full generation
        cuda_sync()
        t0 = time.perf_counter()
        _generate(model, args.model, args.text, args.language_id)
        cuda_sync()
        gen_times.append((time.perf_counter() - t0) * 1000.0)

        # TTFC
        cuda_sync()
        t1 = time.perf_counter()
        for _chunk, _sr, _timing in _generate_streaming(model, args.model, args.text, args.language_id):
            cuda_sync()
            ttfc_times.append((time.perf_counter() - t1) * 1000.0)
            break

    result = {
        "gen_time_avg_ms": float(np.mean(gen_times)),
        "gen_time_std_ms": float(np.std(gen_times)),
        "ttfc_avg_ms": float(np.mean(ttfc_times)),
        "ttfc_p95_ms": float(np.percentile(ttfc_times, 95)),
        "iterations": args.iterations,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
