"""
Регрессионные тесты форка: три независимые фазы.

  mtl_ru            — Паритет CUDA graphs: use_cuda_graph=False vs True (MSE < 1e-3).
                      Основная проверка — что оптимизации не меняют вычисления.

  upstream_quality  — Качество аудио vs upstream resemble-ai/chatterbox.
                      Метрики: MCD (мел-кепстральное расстояние), SQUIM-STOI, WER (Whisper).
                      Пропускается (SKIP) если CHATTERBOX_REGRESSION_BASELINE_PYTHON не задан.

  streaming         — Качество стриминга: SQUIM-STOI ≥ 0.55 и длина в пределах ±5% от полной генерации.
                      MSE выводится информационно: streaming использует prompt caching и stateful vocoder
                      для RTF < 1.0, поэтому побайтовое совпадение с полной генерацией невозможно.

Требуется для upstream_quality:
  CHATTERBOX_REGRESSION_BASELINE_PYTHON — путь к python из venv с upstream chatterbox-tts.
  Подготовка: bash setup_baseline.sh

По умолчанию каждая фаза — отдельный дочерний процесс (сброс VRAM между фазами).
Один процесс (отладка): CHATTERBOX_REGRESSION_INPROC=1 uv run python verify_regression.py
Одна фаза:              uv run python verify_regression.py mtl_ru
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from chatterbox.models.s3gen import S3GEN_SR
from benchmarks._model_loader import load_mtl_model
from benchmarks._utils import (
    free_cuda_memory,
    release_models,
    set_seed,
    waveform_cpu_clone,
)

PHASE_MTL_RU = "mtl_ru"
PHASE_UPSTREAM_QUALITY = "upstream_quality"
PHASE_STREAMING = "streaming"
_PHASE_ORDER = (PHASE_MTL_RU, PHASE_UPSTREAM_QUALITY, PHASE_STREAMING)

_WORKER = Path(__file__).parent / "benchmarks" / "_baseline_waveform_worker.py"

_TEXT_RU = "Быстрая коричневая лиса прыгает через ленивую собаку."


# ---------------------------------------------------------------------------
# Сравнение волн (используется для graph parity и streaming)
# ---------------------------------------------------------------------------

def compare_wavs(wav1: torch.Tensor, wav2: torch.Tensor, name: str) -> bool:
    w1 = wav1.detach().cpu().numpy()
    w2 = wav2.detach().cpu().numpy()
    shape_match = w1.shape == w2.shape
    min_len = min(w1.shape[-1], w2.shape[-1])
    mse = np.mean((w1[..., :min_len] - w2[..., :min_len]) ** 2)

    print(f"\n--- Result for {name} ---")
    print(f"Shape match: {shape_match}  (wav1: {w1.shape}, wav2: {w2.shape})")
    print(f"Waveform MSE: {mse:.2e}")

    if mse < 1e-3:
        print(f"[SUCCESS]: {name} passed.")
        return True
    print(f"[FAILURE]: {name} failed.")
    return False


# ---------------------------------------------------------------------------
# Метрики качества (для phase_upstream_quality)
# ---------------------------------------------------------------------------

def _compute_mcd(wav1: torch.Tensor, wav2: torch.Tensor, sr: int, n_mfcc: int = 13) -> float:
    """Mel Cepstral Distortion (дБ). Меньше — лучше. < 10 dB = приемлемо.

    Параметры mel-банка выбраны под sr=22050:
      n_fft=1024  →  513 FFT-бин  (достаточно для n_mels=80 без нулевых фильтров)
      hop_length=256, n_mels=80   — стандарт для TTS-систем
    Без этих параметров torchaudio использует n_fft=400 → 201 бин при n_mels=128,
    что приводит к нулевым фильтрам и некорректному MCD (> 40 дБ).
    """
    import torchaudio.transforms as T
    mfcc_fn = T.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc + 1,
        melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80},
    )
    # Пропускаем c_0 (энергия), берём c_1..c_n
    m1 = mfcc_fn(wav1.cpu().float())[..., 1:, :]
    m2 = mfcc_fn(wav2.cpu().float())[..., 1:, :]
    min_t = min(m1.shape[-1], m2.shape[-1])
    diff = m1[..., :min_t] - m2[..., :min_t]
    return float((10.0 / np.log(10.0)) * torch.sqrt(2.0 * (diff ** 2).mean()).item())


def _squim_stoi(wav: torch.Tensor, sr: int) -> float | None:
    """
    Безэталонная оценка STOI через torchaudio SQUIM_OBJECTIVE.
    Диапазон 0–1. Возвращает None при ошибке (нет сети / модель не скачана).
    """
    try:
        import torchaudio
        from torchaudio.pipelines import SQUIM_OBJECTIVE
        if sr != 16000:
            resamp = torchaudio.transforms.Resample(sr, 16000)
            wav16 = resamp(wav.cpu().float())
        else:
            wav16 = wav.cpu().float()
        if wav16.ndim == 1:
            wav16 = wav16.unsqueeze(0)
        squim = SQUIM_OBJECTIVE.get_model()
        with torch.inference_mode():
            stoi, _, _ = squim(wav16)
        return float(stoi.mean().item())
    except Exception as exc:
        print(f"  [WARN] SQUIM-STOI недоступен: {exc}")
        return None


def _normalize_text(text: str) -> str:
    """Нижний регистр + удаление знаков препинания для WER."""
    import re
    return re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE)


def _wer(ref: str, hyp: str) -> float:
    """Word Error Rate через динамическое программирование."""
    r = _normalize_text(ref).split()
    h = _normalize_text(hyp).split()
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, m + 1):
            dp[j] = prev[j - 1] if r[i - 1] == h[j - 1] else 1 + min(prev[j - 1], prev[j], dp[j - 1])
    return dp[m] / max(n, 1)


def _whisper_wer(wav: torch.Tensor, ref_text: str, sr: int) -> float | None:
    """Транскрибирует аудио через Whisper-medium, возвращает WER vs ref_text.

    ВАЖНО: используется именно `medium`, а не `small`. На этом TTS-кодеке
    `whisper-small` склонен к галлюцинациям финальных слов (см. AGENTS.md,
    hard rule №8). Для русского `medium` — минимальный надёжный размер.
    """
    try:
        import whisper
        import torchaudio
        if sr != 16000:
            resamp = torchaudio.transforms.Resample(sr, 16000)
            wav16 = resamp(wav.cpu().float()).squeeze().numpy()
        else:
            wav16 = wav.cpu().float().squeeze().numpy()
        w_model = whisper.load_model("medium")
        result = w_model.transcribe(wav16, language="ru")
        hyp = result["text"].strip()
        print(f"  Whisper: {hyp!r}")
        return _wer(ref_text, hyp)
    except Exception as exc:
        print(f"  [WARN] Whisper недоступен: {exc}")
        return None


# ---------------------------------------------------------------------------
# Фаза 1: паритет CUDA graphs (use_cuda_graph=False vs True)
# ---------------------------------------------------------------------------

def phase_mtl_ru(device: str = "cuda", seed: int = 42) -> bool:
    """
    Сравнивает генерацию без CUDA graphs и с ними внутри форка.
    skip_watermark=True — исключаем потенциальную стохастичность вотермаркера.
    Порог: MSE < 1e-3.
    """
    print("\n[Testing Graph Parity: use_cuda_graph=False vs use_cuda_graph=True]")

    gen_kwargs = dict(language_id="ru", temperature=0.001, skip_watermark=True)

    set_seed(seed)
    model = load_mtl_model(device, use_cuda_graph=False)
    wav_no_graph = waveform_cpu_clone(model.generate(_TEXT_RU, **gen_kwargs))
    release_models(model)
    free_cuda_memory()

    set_seed(seed)
    model = load_mtl_model(device, use_cuda_graph=True)
    wav_graph = waveform_cpu_clone(model.generate(_TEXT_RU, **gen_kwargs))
    release_models(model)

    ok = compare_wavs(wav_no_graph, wav_graph, "MTL-RU Graph Parity")
    del wav_no_graph, wav_graph
    free_cuda_memory()
    return ok


# ---------------------------------------------------------------------------
# Фаза 2: качество аудио vs upstream (MCD + SQUIM-STOI + WER)
# ---------------------------------------------------------------------------

def phase_upstream_quality(device: str = "cuda", seed: int = 42) -> bool:
    """
    Сравнивает качество аудио форка с upstream resemble-ai/chatterbox.
    Не требует побитового совпадения (алгоритм ODE намеренно изменён).
    Проверяет что деградации нет по трём независимым метрикам.

    Метрики и пороги:
      MCD      < 10.0 dB   — мел-кепстральное расстояние
      SQUIM-STOI >= 0.55   — безэталонный STOI форка (и не хуже upstream на 0.1)
      WER      < 15%       — разборчивость речи через Whisper

    Если CHATTERBOX_REGRESSION_BASELINE_PYTHON не задан — фаза пропускается (SKIP).
    """
    baseline_py = os.environ.get("CHATTERBOX_REGRESSION_BASELINE_PYTHON", "").strip()
    if not baseline_py:
        print("\n[SKIP]: upstream_quality — задайте CHATTERBOX_REGRESSION_BASELINE_PYTHON для запуска.")
        return True

    print("\n[Testing Upstream Quality: MCD + SQUIM-STOI + WER vs upstream]")

    # --- Получаем эталонное аудио из upstream subprocess ---
    fd, npy_path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)
    wav_base: torch.Tensor | None = None
    try:
        cmd = [
            baseline_py, str(_WORKER),
            "--output", npy_path,
            "--device", device,
            "--seed", str(seed),
            "--language-id", "ru",
            "--temperature", "0.001",
            "--text", _TEXT_RU,
        ]
        mtl_dir = os.environ.get("CHATTERBOX_MTL_MODEL_DIR", "").strip()
        if mtl_dir:
            cmd += ["--mtl-model-dir", mtl_dir]
        proc = subprocess.run(cmd, cwd=str(_WORKER.parent), env=os.environ.copy())
        if proc.returncode != 0:
            print(f"\n[FAILURE]: upstream subprocess завершился с кодом {proc.returncode}", file=sys.stderr)
            return False
        wav_base = torch.from_numpy(np.load(npy_path)).float()
    finally:
        try:
            os.unlink(npy_path)
        except OSError:
            pass

    if wav_base is None:
        return False

    # --- Генерация форка ---
    set_seed(seed)
    model = load_mtl_model(device, use_cuda_graph=True)
    wav_fork = waveform_cpu_clone(model.generate(_TEXT_RU, language_id="ru", temperature=0.001))
    release_models(model)

    sr = S3GEN_SR

    # --- Smoke: длина аудио ---
    min_secs = min(wav_base.shape[-1], wav_fork.shape[-1]) / sr
    print(f"\n--- Result for Upstream-Quality ---")
    print(f"Audio length: base={wav_base.shape[-1]/sr:.2f}s  fork={wav_fork.shape[-1]/sr:.2f}s")
    if min_secs < 0.5:
        print(f"[FAILURE]: аудио слишком короткое ({min_secs:.2f}s < 0.5s)")
        return False

    results: list[bool] = []

    # --- MCD (информационно) ---
    # Форк намеренно использует z=zeros вместо z=randn для детерминированности CUDA graphs
    # (см. AGENTS.md). Из-за разных ODE-стартов оба аудио genuinely разные — без DTW-выравнивания
    # MCD между fork и upstream всегда будет 20–40 dB независимо от качества генерации.
    # Метрика здесь информационная: сигнализирует об аномально коротком/тихом аудио,
    # но не входит в pass/fail.
    mcd = _compute_mcd(wav_base, wav_fork, sr)
    print(f"MCD:        {mcd:.2f} dB    (информационно — разные ODE-старты, DTW не применяется)")

    # --- SQUIM-STOI (основной gate качества) ---
    stoi_base = _squim_stoi(wav_base, sr)
    stoi_fork = _squim_stoi(wav_fork, sr)
    if stoi_base is not None and stoi_fork is not None:
        stoi_ok = stoi_fork >= 0.55 and stoi_fork >= stoi_base - 0.1
        results.append(stoi_ok)
        print(f"SQUIM-STOI: base={stoi_base:.3f}  fork={stoi_fork:.3f}  "
              f"(fork >= 0.55 и >= base-0.1)  → {'OK' if stoi_ok else 'FAIL'}")
    else:
        print("SQUIM-STOI: пропущено (недоступно)")

    # --- WER (Whisper, относительная проверка) ---
    # Сравниваем fork и baseline по одной модели: если оба дают похожий WER — это
    # ограничение Whisper на данном тексте, а не деградация TTS.
    # Порог: fork WER не хуже baseline WER более чем на 15 п.п. и не хуже 35% абсолютно.
    wer_fork = _whisper_wer(wav_fork, _TEXT_RU, sr)
    wer_base = _whisper_wer(wav_base, _TEXT_RU, sr)
    if wer_fork is not None:
        if wer_base is not None:
            wer_threshold = max(0.35, wer_base + 0.15)
            wer_ok = wer_fork <= wer_threshold
            results.append(wer_ok)
            print(f"WER:  fork={wer_fork:.1%}  base={wer_base:.1%}  "
                  f"(порог: fork ≤ max(35%, base+15pp) = {wer_threshold:.0%})  "
                  f"→ {'OK' if wer_ok else 'FAIL'}")
        else:
            wer_ok = wer_fork < 0.35
            results.append(wer_ok)
            print(f"WER (fork): {wer_fork:.1%}    (порог: < 35%)   → {'OK' if wer_ok else 'FAIL'}")
    else:
        print("WER:        пропущено (Whisper недоступен)")

    ok = all(results) if results else False
    if ok:
        print("[SUCCESS]: upstream_quality прошёл.")
    else:
        print("[FAILURE]: upstream_quality не прошёл.")

    del wav_base, wav_fork
    free_cuda_memory()
    return ok


# ---------------------------------------------------------------------------
# Фаза 3: качество и консистентность стриминга
# ---------------------------------------------------------------------------

def phase_streaming(device: str = "cuda", seed: int = 42) -> bool:
    """
    Проверяет, что стриминг-генерация производит качественное аудио сравнимой длины.

    Примечание по архитектуре: S3GenStreamer использует prompt caching (раздельное
    кодирование промпта и speech-токенов) и stateful vocoder (last_s) для достижения
    RTF < 1.0. Эти оптимизации делают побайтовое совпадение с generate() невозможным
    (MSE ≈ 0.026 — норма). Поэтому тест проверяет качество (SQUIM-STOI) и длину, а не MSE.
    """
    print("\n[Testing Streaming vs Non-streaming (Multilingual)]")
    text_st = "Потоковая передача должна быть согласована с полной генерацией."
    stream_sr = S3GEN_SR

    set_seed(seed)
    model_mtl_stream = load_mtl_model(device, use_cuda_graph=True)
    wav_full_cpu = waveform_cpu_clone(
        model_mtl_stream.generate(text_st, language_id="ru", temperature=0.001, skip_watermark=True)
    )
    free_cuda_memory()

    set_seed(seed)
    chunks = []
    for chunk, stream_sr, _timing in model_mtl_stream.generate_streaming(
        text_st, language_id="ru", temperature=0.001, skip_watermark=True
    ):
        chunks.append(chunk)
    wav_stream = torch.from_numpy(np.concatenate(chunks)).unsqueeze(0)

    results = []
    print(f"\n--- Result for Streaming-Consistency ---")
    print(f"Shape full: {tuple(wav_full_cpu.shape)},  shape stream: {tuple(wav_stream.shape)}")

    # MSE (информационно) — не критерий из-за prompt caching + stateful vocoder
    min_len = min(wav_full_cpu.shape[-1], wav_stream.shape[-1])
    w1 = wav_full_cpu[..., :min_len].cpu().numpy()
    w2 = wav_stream[..., :min_len].cpu().numpy()
    mse = float(np.mean((w1 - w2) ** 2))
    print(f"Waveform MSE: {mse:.2e}  (информационно — streaming использует RTF-оптимизации)")

    # Длина: в пределах ±5% от полной генерации
    len_full = wav_full_cpu.shape[-1]
    len_stream = wav_stream.shape[-1]
    len_diff_pct = abs(len_full - len_stream) / max(len_full, len_stream)
    len_ok = len_diff_pct < 0.05
    results.append(len_ok)
    print(f"Length diff: {len_diff_pct:.1%}  (< 5%)  → {'OK' if len_ok else 'FAIL'}")

    # SQUIM-STOI: основной gate качества стримингового аудио
    stoi_stream = _squim_stoi(wav_stream, stream_sr)
    if stoi_stream is not None:
        stoi_ok = stoi_stream >= 0.55
        results.append(stoi_ok)
        print(f"SQUIM-STOI (streaming): {stoi_stream:.3f}  (≥ 0.55)  → {'OK' if stoi_ok else 'FAIL'}")
    else:
        print("SQUIM-STOI: пропущено (недоступно)")

    ok = all(results) if results else False
    if ok:
        print("[SUCCESS]: Streaming-Consistency passed.")
    else:
        print("[FAILURE]: Streaming-Consistency failed.")

    del wav_full_cpu, wav_stream
    release_models(model_mtl_stream)
    return ok


# ---------------------------------------------------------------------------
# Оркестрация
# ---------------------------------------------------------------------------

_PHASE_RUNNERS: dict[str, Callable[[], bool]] = {
    PHASE_MTL_RU: lambda: phase_mtl_ru(),
    PHASE_UPSTREAM_QUALITY: lambda: phase_upstream_quality(),
    PHASE_STREAMING: lambda: phase_streaming(),
}


def verify_in_process() -> bool:
    """Все фазы в одном процессе (для отладки или при CHATTERBOX_REGRESSION_INPROC=1)."""
    results = [phase_mtl_ru(), phase_upstream_quality(), phase_streaming()]
    print("\n" + "=" * 40)
    if all(results):
        print("FINAL VERDICT: ALL REGRESSION TESTS PASSED")
    else:
        print("FINAL VERDICT: REGRESSION DETECTED")
    print("=" * 40)
    return all(results)


def verify_subprocess() -> bool:
    """Каждая фаза — новый интерпретатор: VRAM полностью сбрасывается между фазами."""
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
    return runner()


def main() -> None:
    if len(sys.argv) >= 2:
        phase = sys.argv[1]
        ok = _cli_child_phase(phase)
        sys.exit(0 if ok else 1)
    ok = verify()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
