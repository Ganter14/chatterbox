"""
Персистентность результатов бенчмарков.

Каждый прогон сохраняется в benchmarks/results/<timestamp>_<sha7>.json
вместе с git-метаданными и информацией о GPU.

Использование:
    from benchmarks._results import save, load_history

    save("rtf", {"rtf_short": 0.82, "ttfc_p50": 210})
    history = load_history()
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

_RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Метаданные окружения
# ---------------------------------------------------------------------------

def get_git_meta() -> dict[str, str]:
    """Возвращает {'commit': 'abc1234', 'branch': 'feat/...'}.
    При ошибке (нет git) поля содержат пустую строку.
    """
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return ""

    return {
        "commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": _run(["git", "branch", "--show-current"]),
    }


def get_gpu_info() -> str:
    """Название GPU или 'cpu'."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


# ---------------------------------------------------------------------------
# Сохранение / загрузка
# ---------------------------------------------------------------------------

def save(name: str, data: dict[str, Any]) -> Path:
    """
    Сохраняет результаты в JSON.

    Args:
        name: метка прогона, используется в имени файла (например 'cuda', 'rtf').
        data: словарь с результатами.

    Returns:
        Путь к созданному файлу.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    git = get_git_meta()
    timestamp = datetime.now().isoformat(timespec="seconds")
    sha = git["commit"] or "unknown"

    record: dict[str, Any] = {
        "timestamp": timestamp,
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "gpu": get_gpu_info(),
        "results": data,
    }

    safe_ts = timestamp.replace(":", "-")
    filename = _RESULTS_DIR / f"{safe_ts}_{sha}_{name}.json"
    filename.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return filename


def load_history(name: str | None = None) -> list[dict[str, Any]]:
    """
    Загружает все сохранённые результаты из benchmarks/results/.

    Args:
        name: если задан, фильтрует файлы по суффиксу `_{name}.json`.

    Returns:
        Список записей, отсортированных по timestamp (старые первыми).
    """
    if not _RESULTS_DIR.exists():
        return []

    records: list[dict[str, Any]] = []
    suffix = f"_{name}.json" if name else ".json"

    for path in sorted(_RESULTS_DIR.glob(f"*{suffix}")):
        try:
            records.append(json.loads(path.read_text()))
        except Exception:
            pass

    return records
