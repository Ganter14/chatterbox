# benchmarks/

Performance measurement and regression testing infrastructure for the
Chatterbox TTS fork.

## Purpose

> **Scope.** All benchmarks in this fork target `ChatterboxMultilingualTTS`.
> The `benchmarks/cuda.py` Turbo-EN case has been removed; `_model_loader.load_turbo_model`
> is retained only for upstream-venv compatibility.

The module answers two independent questions, with a third regression
check living one level up:

| Script / module | Question |
|---|---|
| `benchmarks/cuda.py` | How much did CUDA Graphs speed generation up? |
| `benchmarks/rtf.py` | Does streaming run in real time (RTF < 1.0)? |
| `verify_regression.py` (repo root) | Did an optimization break the audio? (MSE / SQUIM / WER) |

---

## File layout

```
benchmarks/
  __init__.py
  _utils.py                    — shared primitives: set_seed, free_cuda, Timer,
                                  capture_vram_peak, measure_ttfc
  _results.py                  — JSON persistence: save(), load_history(),
                                  git metadata, GPU info
  _baseline_waveform_worker.py — subprocess worker spawned under the
                                  upstream venv; saves waveform .npy
                                  (consumed by verify_regression.py)
  _baseline_timing_worker.py   — subprocess worker spawned under the
                                  upstream venv; saves timing JSON
                                  (consumed by benchmarks/cuda.py)
  _model_loader.py             — fork-side helpers for loading MTL/Turbo models
  cuda.py                      — benchmark: full-gen time, TTFC, peak VRAM,
                                  optional vs-upstream speedup
  rtf.py                       — benchmark: RTF, TTFC p50/p95, per-chunk decode
  results/                     — saved JSON runs (gitignored)
```

---

## Running

Run from the repository root via `uv run`:

```bash
# CUDA Graphs benchmark (full-gen time, TTFC, peak VRAM)
uv run python benchmark_cuda.py

# Streaming RTF benchmark
uv run python benchmark_rtf.py

# Regression: fork waveform vs upstream baseline
uv run python verify_regression.py

# Single regression phase
uv run python verify_regression.py mtl_ru
uv run python verify_regression.py streaming

# Run all phases in a single process (debug only)
CHATTERBOX_REGRESSION_INPROC=1 uv run python verify_regression.py
```

---

## Environment variables

| Variable | Used by | Purpose |
|---|---|---|
| `CHATTERBOX_MTL_MODEL_DIR` | all benchmarks, `verify_regression.py` | Local directory with weights of the MTL model (`ResembleAI/chatterbox`). When set, Hugging Face is not contacted. |
| `CHATTERBOX_REGRESSION_BASELINE_PYTHON` | `verify_regression.py`, `benchmark_cuda.py` | Path to a `python` from the upstream venv. **Required** for the `upstream_quality` regression phase and for the speedup column of the CUDA benchmark. |
| `CHATTERBOX_REGRESSION_INPROC` | `verify_regression.py` | When `1`, all phases run in a single process (no VRAM reset between phases). Debug only. |
| `CHATTERBOX_BASELINE_VENV` | `setup_baseline.sh` | Path to the baseline venv (default `../chatterbox-baseline-venv`). |
| `CHATTERBOX_UPSTREAM_REF` | `setup_baseline.sh` | Tag / commit / branch of upstream (default `master`). |

### Offline mode: local weights

When Hugging Face access is unstable or blocked, download the weights
once:

```bash
# Download the MTL model (~5–8 GB) into ~/.local/share/chatterbox-models/
uv run python download_models.py --model mtl

# Or into a custom directory
uv run python download_models.py --model mtl --output-dir /data/models

# The script prints the export line — append it to ~/.bashrc / ~/.zshrc
export CHATTERBOX_MTL_MODEL_DIR=~/.local/share/chatterbox-models/chatterbox
```

From that point on, every benchmark and `verify_regression.py`
automatically uses the local weights. `CHATTERBOX_MTL_MODEL_DIR` is also
forwarded into the upstream-baseline subprocess workers.

### Bootstrap the upstream baseline in one command

```bash
bash setup_baseline.sh
export CHATTERBOX_REGRESSION_BASELINE_PYTHON=<path printed by the script>
```

---

## Persisted results

Every run of `benchmark_cuda.py` and `benchmark_rtf.py` writes a JSON
file into `benchmarks/results/`:

```
benchmarks/results/
  2026-05-14T23-58-00_abc1234_cuda.json
  2026-05-14T23-59-00_abc1234_rtf.json
```

File structure:

```json
{
  "timestamp": "2026-05-14T23:58:00",
  "git_commit": "abc1234",
  "git_branch": "feat/cuda-graphs",
  "gpu": "NVIDIA GeForce RTX 3090",
  "results": { ... }
}
```

History is loadable from Python:

```python
from benchmarks._results import load_history

for record in load_history("rtf"):
    print(record["git_commit"], record["results"]["Short"]["rtf_avg"])
```

---

## Architectural constraint on the baseline workers

The files `_baseline_waveform_worker.py` and `_baseline_timing_worker.py`
are spawned **under the upstream-venv interpreter**
(`CHATTERBOX_REGRESSION_BASELINE_PYTHON`). In that environment the
fork's `benchmarks` package physically does not exist.

**Therefore they intentionally do not import anything from the fork**,
including `benchmarks._utils`. The duplication of `set_seed` and
gc-related logic inside these files is a deliberate compromise, not
technical debt. Do not "fix" it.

---

## Adding a new benchmark

1. Create `benchmarks/my_bench.py` and use primitives from `_utils.py`.
2. Persist results via `from benchmarks._results import save; save("my_bench", data)`.
3. Add a two-line root wrapper `benchmark_my_bench.py`.
4. Document it in this README.
