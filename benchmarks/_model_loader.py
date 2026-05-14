"""
Загрузчик моделей с поддержкой локального кэша.

Если задана переменная окружения CHATTERBOX_MTL_MODEL_DIR или
CHATTERBOX_TURBO_MODEL_DIR — модель загружается из локальной директории
через from_local(), минуя Hugging Face.
Иначе — через from_pretrained() (требует сетевого доступа к HF).

Переменные окружения (можно задать в .env в корне репозитория):
    CHATTERBOX_MTL_MODEL_DIR    — путь к директории весов MTL-модели
                                  (ResembleAI/chatterbox, загружается через download_models.py)
    CHATTERBOX_TURBO_MODEL_DIR  — путь к директории весов Turbo-модели
                                  (ResembleAI/chatterbox-turbo)

ВАЖНО: этот модуль импортируется только из кода форка.
Воркеры _baseline_*_worker.py запускаются под upstream-venv и не используют этот хелпер —
они принимают путь к директории через аргументы командной строки.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Загружает переменные из .env в корне проекта, если файл существует.

    Не перезаписывает уже установленные переменные окружения — явный export в
    шелле всегда имеет приоритет. Поддерживает кавычки и комментарии (#).

    Относительные пути разрешаются относительно директории .env файла,
    чтобы значение оставалось корректным независимо от рабочей директории
    при запуске скрипта.
    """
    env_path = (Path(__file__).parent.parent / ".env").resolve()
    if not env_path.exists():
        return
    env_dir = env_path.parent
    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key or key in os.environ:
                continue
            # Превращаем относительные пути в абсолютные относительно директории .env
            p = Path(value)
            if not p.is_absolute():
                value = str((env_dir / p).resolve())
            os.environ[key] = value


_load_dotenv()


def load_mtl_model(
    device: str,
    use_cuda_graph: bool = False,
    t3_model: str | None = None,
):
    """Загружает ChatterboxMultilingualTTS из локального каталога или HuggingFace."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    local_dir = os.environ.get("CHATTERBOX_MTL_MODEL_DIR", "").strip()
    if local_dir:
        print(f"  [loader] MTL: локальная директория: {local_dir}")
        return ChatterboxMultilingualTTS.from_local(
            local_dir, device, t3_model=t3_model, use_cuda_graph=use_cuda_graph
        )

    print("  [loader] MTL: загрузка с HuggingFace...")
    return ChatterboxMultilingualTTS.from_pretrained(
        device=device, t3_model=t3_model, use_cuda_graph=use_cuda_graph
    )


def load_turbo_model(
    device: str,
    use_cuda_graph: bool = False,
):
    """Загружает ChatterboxTurboTTS из локального каталога или HuggingFace."""
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    local_dir = os.environ.get("CHATTERBOX_TURBO_MODEL_DIR", "").strip()
    if local_dir:
        print(f"  [loader] Turbo: локальная директория: {local_dir}")
        return ChatterboxTurboTTS.from_local(local_dir, device, use_cuda_graph=use_cuda_graph)

    print("  [loader] Turbo: загрузка с HuggingFace...")
    return ChatterboxTurboTTS.from_pretrained(device=device, use_cuda_graph=use_cuda_graph)
