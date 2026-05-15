"""
Model loader with local-cache support.

If CHATTERBOX_MTL_MODEL_DIR is set, the MTL model is loaded from the local
directory via from_local(), bypassing Hugging Face. Otherwise, from_pretrained()
is used (requires network access).

Environment variables (can be set in .env at the repo root):
    CHATTERBOX_MTL_MODEL_DIR   — path to the MTL model weights directory
                                  (ResembleAI/chatterbox, downloaded via download_models.py)
    CHATTERBOX_TURBO_MODEL_DIR — path to the Turbo model weights directory
                                  (ResembleAI/chatterbox-turbo; retained for upstream
                                  compatibility only — not used in active fork benchmarks)

NOTE: this module is imported only from fork code.
The _baseline_*_worker.py scripts run under the upstream venv and do not use
this helper — they receive the model directory path via CLI arguments.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load variables from .env at the repo root if the file exists.

    Does not overwrite already-set environment variables — an explicit shell
    export always takes priority. Supports quoted values and comments (#).

    Relative paths are resolved relative to the .env file's directory so that
    values remain correct regardless of the working directory at launch time.
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
    """Load ChatterboxMultilingualTTS from a local directory or HuggingFace."""
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    local_dir = os.environ.get("CHATTERBOX_MTL_MODEL_DIR", "").strip()
    if local_dir:
        print(f"  [loader] MTL: local directory: {local_dir}")
        return ChatterboxMultilingualTTS.from_local(
            local_dir, device, t3_model=t3_model, use_cuda_graph=use_cuda_graph
        )

    print("  [loader] MTL: loading from HuggingFace...")
    return ChatterboxMultilingualTTS.from_pretrained(
        device=device, t3_model=t3_model, use_cuda_graph=use_cuda_graph
    )


def load_turbo_model(
    device: str,
    use_cuda_graph: bool = False,
):
    """Load ChatterboxTurboTTS from a local directory or HuggingFace.

    Retained for upstream-venv compatibility only.
    Not used in active fork benchmarks — use load_mtl_model instead.
    """
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    local_dir = os.environ.get("CHATTERBOX_TURBO_MODEL_DIR", "").strip()
    if local_dir:
        print(f"  [loader] Turbo: local directory: {local_dir}")
        return ChatterboxTurboTTS.from_local(local_dir, device, use_cuda_graph=use_cuda_graph)

    print("  [loader] Turbo: loading from HuggingFace...")
    return ChatterboxTurboTTS.from_pretrained(device=device, use_cuda_graph=use_cuda_graph)
