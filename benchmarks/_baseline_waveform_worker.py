"""
Subprocess-воркер для получения эталонной волны из upstream chatterbox-tts.

ВАЖНО — АРХИТЕКТУРНОЕ ОГРАНИЧЕНИЕ:
    Этот файл запускается под интерпретатором из CHATTERBOX_REGRESSION_BASELINE_PYTHON
    (отдельный venv с upstream chatterbox-tts). Пакет benchmarks форка там недоступен,
    поэтому этот модуль намеренно самодостаточен и НЕ импортирует ничего из форка.
    Дублирование set_seed / gc-логики здесь — осознанный компромисс.

Вызывается из verify_regression.py через subprocess.
Сохраняет волну в .npy (float32) по пути --output.
"""
from __future__ import annotations

import argparse
import gc
import random

import numpy as np
import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


TEXT_DEFAULT = "Быстрая коричневая лиса прыгает через ленивую собаку."


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Upstream baseline waveform for verify_regression.py")
    parser.add_argument("--output", required=True, help="Путь к выходному .npy (массив float32)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language-id", default="ru")
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--text", default=TEXT_DEFAULT)
    parser.add_argument(
        "--mtl-model-dir",
        default=None,
        help="Путь к локальной директории весов MTL-модели (обходит HuggingFace)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.mtl_model_dir:
        model = ChatterboxMultilingualTTS.from_local(args.mtl_model_dir, device=args.device)
    else:
        model = ChatterboxMultilingualTTS.from_pretrained(device=args.device)
    wav = model.generate(args.text, language_id=args.language_id, temperature=args.temperature)
    wav_np = wav.detach().cpu().to(dtype=torch.float32).numpy()
    np.save(args.output, wav_np)

    del model, wav
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
