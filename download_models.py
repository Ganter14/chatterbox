"""
Скрипт однократной загрузки весов моделей Chatterbox в локальную директорию.

После загрузки бенчмарки и регрессионные тесты работают без обращения к Hugging Face,
что устраняет проблемы с медленным или заблокированным доступом.

Запуск:
    uv run python download_models.py
    uv run python download_models.py --output-dir /data/models
    uv run python download_models.py --model mtl     # только MTL
    uv run python download_models.py --model turbo   # только Turbo

После загрузки добавьте в ~/.bashrc / ~/.zshrc (команды выводятся скриптом):
    export CHATTERBOX_MTL_MODEL_DIR=...
    export CHATTERBOX_TURBO_MODEL_DIR=...
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_OUTPUT_DIR = Path.home() / ".local" / "share" / "chatterbox-models"

MTL_REPO = "ResembleAI/chatterbox"
TURBO_REPO = "ResembleAI/chatterbox-turbo"

MTL_PATTERNS = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "t3_mtl23ls_v3.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]
TURBO_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]


def download_mtl(output_dir: Path, token: str | None) -> Path:
    local_dir = output_dir / "chatterbox"
    print(f"\n[MTL] Загрузка {MTL_REPO} → {local_dir}")
    snapshot_download(
        repo_id=MTL_REPO,
        repo_type="model",
        local_dir=str(local_dir),
        allow_patterns=MTL_PATTERNS,
        token=token,
    )
    print("[MTL] Готово.")
    return local_dir


def download_turbo(output_dir: Path, token: str | None) -> Path:
    local_dir = output_dir / "chatterbox-turbo"
    print(f"\n[Turbo] Загрузка {TURBO_REPO} → {local_dir}")
    snapshot_download(
        repo_id=TURBO_REPO,
        repo_type="model",
        local_dir=str(local_dir),
        allow_patterns=TURBO_PATTERNS,
        token=token,
    )
    print("[Turbo] Готово.")
    return local_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Загрузка весов моделей Chatterbox для офлайн-использования."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Корневая директория для моделей (по умолчанию: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        choices=["mtl", "turbo", "all"],
        default="all",
        help="Какую модель загружать (по умолчанию: all)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace токен (по умолчанию: $HF_TOKEN)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    token: str | None = args.token or None

    mtl_dir: Path | None = None
    turbo_dir: Path | None = None

    if args.model in ("mtl", "all"):
        mtl_dir = download_mtl(output_dir, token)

    if args.model in ("turbo", "all"):
        turbo_dir = download_turbo(output_dir, token)

    print("\n" + "=" * 60)
    print("Загрузка завершена. Добавьте в ~/.bashrc / ~/.zshrc:")
    print()
    if mtl_dir:
        print(f"  export CHATTERBOX_MTL_MODEL_DIR={mtl_dir}")
    if turbo_dir:
        print(f"  export CHATTERBOX_TURBO_MODEL_DIR={turbo_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
