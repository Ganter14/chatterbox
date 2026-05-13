"""
Регрессия baseline vs use_cuda_graph и стриминг vs полная генерация.

По умолчанию фазы (Turbo-EN, MTL-RU, стриминг) выполняются в отдельных дочерних процессах,
чтобы после каждой фазы драйвер полностью освобождал VRAM (иначе на Windows/WSL возможен
монотонный рост из-за кэша аллокатора CUDA и графов).

Один процесс (как раньше): CHATTERBOX_REGRESSION_INPROC=1 uv run python verify_regression.py
Отдельная фаза: uv run python verify_regression.py turbo_en
"""
from __future__ import annotations

import gc
import os
import random
import subprocess
import sys
from typing import Callable

import numpy as np
import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS

PHASE_TURBO_EN = "turbo_en"
PHASE_MTL_RU = "mtl_ru"
PHASE_STREAMING = "streaming"
_PHASE_ORDER = (PHASE_TURBO_EN, PHASE_MTL_RU, PHASE_STREAMING)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def free_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def release_models(*models: object) -> None:
    for m in models:
        if m is not None:
            del m
    free_cuda_memory()


def waveform_cpu_clone(wav: torch.Tensor) -> torch.Tensor:
    """Сохранить копию волны на CPU и освободить VRAM от исходного тензора."""
    out = wav.detach().cpu().clone()
    del wav
    return out


def compare_wavs(wav1: torch.Tensor, wav2: torch.Tensor, name: str) -> bool:
    wav1_np = wav1.detach().cpu().numpy()
    wav2_np = wav2.detach().cpu().numpy()

    shape_match = wav1_np.shape == wav2_np.shape

    min_len = min(wav1_np.shape[-1], wav2_np.shape[-1])
    mse = np.mean((wav1_np[..., :min_len] - wav2_np[..., :min_len]) ** 2)

    print(f"\n--- Result for {name} ---")
    print(f"Shape match: {shape_match} (Base: {wav1_np.shape}, Opt: {wav2_np.shape})")
    print(f"Waveform MSE: {mse:.2e}")

    if mse < 1e-3:
        print(f"[SUCCESS]: {name} regression test passed.")
        return True
    print(f"[FAILURE]: {name} regression test failed.")
    return False


def phase_turbo_en(device: str = "cuda", seed: int = 42) -> bool:
    print("\n[Testing Turbo TTS (English)]")
    text_en = "The quick brown fox jumps over the lazy dog."

    set_seed(seed)
    model_turbo_base = ChatterboxTurboTTS.from_pretrained(device=device, use_cuda_graph=False)
    wav_turbo_base_cpu = waveform_cpu_clone(
        model_turbo_base.generate(text_en, temperature=0.001)
    )
    release_models(model_turbo_base)

    set_seed(seed)
    model_turbo_opt = ChatterboxTurboTTS.from_pretrained(device=device, use_cuda_graph=True)
    wav_turbo_opt_cpu = waveform_cpu_clone(
        model_turbo_opt.generate(text_en, temperature=0.001)
    )
    release_models(model_turbo_opt)

    ok = compare_wavs(wav_turbo_base_cpu, wav_turbo_opt_cpu, "Turbo-EN")
    del wav_turbo_base_cpu, wav_turbo_opt_cpu
    free_cuda_memory()
    return ok


def phase_mtl_ru(device: str = "cuda", seed: int = 42) -> bool:
    print("\n[Testing Multilingual TTS (Russian)]")
    text_ru = "Быстрая коричневая лиса прыгает через ленивую собаку."

    set_seed(seed)
    model_mtl_base = ChatterboxMultilingualTTS.from_pretrained(device=device, use_cuda_graph=False)
    wav_mtl_base_cpu = waveform_cpu_clone(
        model_mtl_base.generate(text_ru, language_id="ru", temperature=0.001)
    )
    release_models(model_mtl_base)

    set_seed(seed)
    model_mtl_opt = ChatterboxMultilingualTTS.from_pretrained(device=device, use_cuda_graph=True)
    wav_mtl_opt_cpu = waveform_cpu_clone(
        model_mtl_opt.generate(text_ru, language_id="ru", temperature=0.001)
    )
    release_models(model_mtl_opt)

    ok = compare_wavs(wav_mtl_base_cpu, wav_mtl_opt_cpu, "MTL-RU")
    del wav_mtl_base_cpu, wav_mtl_opt_cpu
    free_cuda_memory()
    return ok


def phase_streaming(device: str = "cuda", seed: int = 42) -> bool:
    print("\n[Testing Streaming vs Non-streaming (Turbo)]")
    text_st = "Streaming should be consistent with full generation."
    set_seed(seed)
    model_turbo_stream = ChatterboxTurboTTS.from_pretrained(device=device, use_cuda_graph=True)
    wav_full_cpu = waveform_cpu_clone(
        model_turbo_stream.generate(text_st, temperature=0.001)
    )
    free_cuda_memory()

    set_seed(seed)
    chunks = []
    last_timing = None
    for chunk, sr, timing in model_turbo_stream.generate_streaming(text_st, temperature=0.001):
        chunks.append(chunk)
        last_timing = timing
    assert last_timing is not None and last_timing["is_final"] is True, (
        "streaming API: last chunk timing must include is_final=True"
    )
    wav_stream = torch.from_numpy(np.concatenate(chunks)).unsqueeze(0)

    ok = compare_wavs(wav_full_cpu, wav_stream, "Streaming-Consistency")
    del wav_full_cpu, wav_stream
    release_models(model_turbo_stream)
    return ok


_PHASE_RUNNERS: dict[str, Callable[[], bool]] = {
    PHASE_TURBO_EN: lambda: phase_turbo_en(),
    PHASE_MTL_RU: lambda: phase_mtl_ru(),
    PHASE_STREAMING: lambda: phase_streaming(),
}


def verify_in_process() -> bool:
    """Все фазы в одном процессе (для отладки или при CHATTERBOX_REGRESSION_INPROC=1)."""
    results = [phase_turbo_en(), phase_mtl_ru(), phase_streaming()]
    print("\n" + "=" * 40)
    if all(results):
        print("FINAL VERDICT: ALL REGRESSION TESTS PASSED")
    else:
        print("FINAL VERDICT: REGRESSION DETECTED")
    print("=" * 40)
    return all(results)


def verify_subprocess() -> bool:
    """Каждая фаза — новый интерпретатор: VRAM сбрасывается драйвером между фазами."""
    script = os.path.abspath(__file__)
    cwd = os.path.dirname(script)
    py = sys.executable
    for phase in _PHASE_ORDER:
        print(f"\n{'=' * 10} Regression subprocess: {phase} {'=' * 10}")
        proc = subprocess.run(
            [py, script, phase],
            cwd=cwd,
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            print("\n" + "=" * 40)
            print(f"FINAL VERDICT: REGRESSION DETECTED (phase {phase} exit {proc.returncode})")
            print("=" * 40)
            return False
    print("\n" + "=" * 40)
    print("FINAL VERDICT: ALL REGRESSION TESTS PASSED")
    print("=" * 40)
    return True


def verify() -> bool:
    if os.environ.get("CHATTERBOX_REGRESSION_INPROC") == "1":
        return verify_in_process()
    return verify_subprocess()


def _cli_child_phase(phase: str) -> bool:
    runner = _PHASE_RUNNERS.get(phase)
    if runner is None:
        print(f"Unknown phase {phase!r}. Expected one of: {list(_PHASE_RUNNERS)}", file=sys.stderr)
        return False
    ok = runner()
    return ok


def main() -> None:
    if len(sys.argv) >= 2:
        phase = sys.argv[1]
        ok = _cli_child_phase(phase)
        sys.exit(0 if ok else 1)
    ok = verify()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
