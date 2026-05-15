# AGENTS.md — Fork Working Context

This file is the **process / policy** contract for working in this repo.
For architecture, dataflow, lifecycles of caches and graphs, known bugs,
and non-obvious design decisions, read [`ARCHITECTURE.md`](ARCHITECTURE.md).

The two documents do not duplicate each other. AGENTS.md tells you
**how to work**; ARCHITECTURE.md tells you **how the code is shaped and why**.

---

## Project

This repository is a fork of **Chatterbox TTS** by Resemble AI, focused on
high-performance NVIDIA GPU inference for low-latency real-time agents:
CUDA Graphs for the T3 autoregressive decode loop, a CUDA Graph for the
S3Gen flow-matching estimator, and a streaming pipeline with stateful
vocoder and prompt caching.

**Active development target: `ChatterboxMultilingualTTS` only.**
`ChatterboxTTS` and `ChatterboxTurboTTS` remain in the codebase as upstream
code but are not actively developed, optimized, or tested in this fork.

---

## Environment

- **GPU**: NVIDIA, Ampere or newer recommended for stable CUDA Graphs.
  Minimum 8 GB VRAM (12+ GB for batches > 2).
- **CUDA**: 12.8 strictly (hard-pinned by `nvidia-*` deps in
  [`pyproject.toml`](pyproject.toml)).
- **Python**: 3.12+.
- **Package manager**: [`uv`](https://github.com/astral-sh/uv), not pip.

Bootstrap from `chatterbox/`:

```bash
uv sync --frozen
```

`--frozen` installs strictly from `uv.lock` without any version drift.
All scripts are launched as `uv run python ...`.

---

## Public entry points (asymmetric!)

| Class | Module | T3 backbone | `use_cuda_graph` | `generate_streaming` | Status |
|---|---|---|---|---|---|
| `ChatterboxTTS` (EN) | [`src/chatterbox/tts.py`](src/chatterbox/tts.py) | GPT-2 medium + learned speech pos-emb | **NO** (not wired) | synchronous | upstream only, not maintained |
| `ChatterboxTurboTTS` (EN) | [`src/chatterbox/tts_turbo.py`](src/chatterbox/tts_turbo.py) | GPT-2 medium (Turbo config) | yes, `T3Graph(batch_size=1)` | synchronous | upstream only, not maintained |
| `ChatterboxMultilingualTTS` (23 languages incl. RU) | [`src/chatterbox/mtl_tts.py`](src/chatterbox/mtl_tts.py) | Llama_520M | yes, `T3Graph(batch_size=2)` + `S3GenGraph` | threaded | **active target** |

Notes:

- `ChatterboxTurboTTS` is **not** re-exported from
  [`src/chatterbox/__init__.py`](src/chatterbox/__init__.py); import as
  `from chatterbox.tts_turbo import ChatterboxTurboTTS`.
- The three classes are intentionally **not** symmetric. Do not assume
  features present in one class are present in the others (most notably:
  `ChatterboxTTS` has no CUDA Graph wiring and no `skip_watermark`).

---

## Read before editing

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) **before** touching any of:

- [`src/chatterbox/models/t3/t3_graph.py`](src/chatterbox/models/t3/t3_graph.py) — T3 CUDA Graph
- [`src/chatterbox/models/s3gen/s3gen_graph.py`](src/chatterbox/models/s3gen/s3gen_graph.py) — S3Gen estimator CUDA Graph
- [`src/chatterbox/models/s3gen/s3gen_streamer.py`](src/chatterbox/models/s3gen/s3gen_streamer.py) — streaming with prompt cache + stateful vocoder
- [`src/chatterbox/utils/streaming_utils.py`](src/chatterbox/utils/streaming_utils.py) — `ThreadedS3GenStreamer` (MTL streaming)
- `generate_streaming` of [`src/chatterbox/mtl_tts.py`](src/chatterbox/mtl_tts.py)
- [`src/chatterbox/models/s3gen/flow_matching.py`](src/chatterbox/models/s3gen/flow_matching.py) — CFM ODE start (`z = zeros_like(mu)` is intentional)
- [`src/chatterbox/models/t3/t3.py`](src/chatterbox/models/t3/t3.py) inference loops (decode hot path)

These files encode invariants and historical bug fixes that are not
inferrable from the code alone.

---

## Hard rules (do not violate)

1. **Do not change version pins** in `pyproject.toml` without an explicit
   user request:
   - `torch==2.10.0`, `torchaudio==2.10.0`
   - `transformers==4.57.3`
   - `nvidia-cuda-runtime-cu12==12.8.90`, `nvidia-cublas-cu12==12.8.4.1`
   - `requires-python>=3.12`

   The `[[tool.uv.index]] pytorch-cu128` block and the corresponding
   `[tool.uv.sources]` mapping for `torch` / `torchaudio` must stay.

2. **No CPU/GPU sync inside the decode hot loop**: no `.item()`, no
   `.tolist()`, no `print`, no Python-side conditionals on tensor values
   between `graph.replay()` calls. Any of these voids CUDA Graph replay.

3. **`StaticCache` discipline**: single allocation up to `max_seq_len`
   (default 2048), `.reset()` before every new prefill. Raising
   `max_seq_len` grows VRAM linearly.

4. **CFG (multilingual)** must be served by a single `T3Graph` with
   `batch_size=2` (conditional + unconditional in one batch). Do not
   split into two separate graphs.

5. **`ChatterboxMultilingualTTS.generate_streaming` API is public**. New
   parameters must have defaults that preserve current behaviour.

6. **CFM ODE start**: the fork uses
   `z = torch.zeros_like(mu)` (see
   [`src/chatterbox/models/s3gen/flow_matching.py:212`](src/chatterbox/models/s3gen/flow_matching.py)).
   This is deterministic and CUDA-Graph-friendly. Do not revert to
   `randn`. It breaks bit-parity with upstream by design — that is why
   `verify_regression.py` uses *relative* metrics, not bit-equality.

7. **CFM Euler steps default = 2** (set in
   [`src/chatterbox/models/s3gen/s3gen.py`](src/chatterbox/models/s3gen/s3gen.py)
   `S3Token2Wav.flow_inference`). This is the fork-wide default for both
   regular CFM and meanflow paths. Do not raise it back to 10 without
   re-running the empirical perceptual / Whisper-medium check that
   justified the drop. If you need to experiment with another value,
   pass `n_cfm_timesteps=` explicitly at the call site instead of
   editing the dispatch line.

8. **Whisper model = `medium`** for any RU WER measurement
   (`verify_regression.py`, future quality benchmarks). `whisper-small`
   was found to hallucinate end-of-utterance words on this codec
   (e.g. inventing «синтезоритии») and **must not** be used as a
   quality gate. `medium` is the minimum reliable transcriber for RU.

9. **Add dependencies via `pyproject.toml`** and the existing uv policy.
   Do not suggest replacing uv with pip, and do not bypass the explicit
   PyTorch CUDA 12.8 index.

---

## Verification

Run from the `chatterbox/` directory via `uv run`.

### 1. `verify_regression.py` — three independent phases

Each phase runs in its own subprocess so VRAM is fully released between
phases.

| Phase | What it checks | Pass criterion |
|---|---|---|
| `mtl_ru` | Fork-internal parity: `use_cuda_graph=False` vs `True` | MSE < 1e-3 |
| `upstream_quality` | Quality vs upstream `resemble-ai/chatterbox` | SQUIM-STOI ≥ 0.55 **and** ≥ baseline − 0.1; WER fork ≤ max(35%, WER_base + 15 pp); MCD informational |
| `streaming` | Streaming quality vs full generation | SQUIM-STOI ≥ 0.55; length within ±5% of full generation; MSE informational |

- `upstream_quality` is **SKIPPED** if `CHATTERBOX_REGRESSION_BASELINE_PYTHON`
  is unset. Prepare the baseline venv once:
  ```bash
  bash setup_baseline.sh
  export CHATTERBOX_REGRESSION_BASELINE_PYTHON=<path printed by the script>
  ```
- Debug a single phase in one process:
  ```bash
  CHATTERBOX_REGRESSION_INPROC=1 uv run python verify_regression.py
  ```
- Run a single phase:
  ```bash
  uv run python verify_regression.py mtl_ru
  uv run python verify_regression.py upstream_quality
  uv run python verify_regression.py streaming
  ```

What the "informational" metrics mean:

- **Streaming MSE** ≈ 0.026 is **normal**, not a regression.
  `S3GenStreamer` uses prompt caching and a stateful vocoder (`last_s`);
  bit-parity with `generate()` is structurally impossible. SQUIM-STOI is
  the real quality gate.
- **MCD** without DTW alignment, between two systems with different ODE
  starts, is always 20–40 dB regardless of audio quality. Useful as a
  smoke test only.
- **WER** threshold is relative because Whisper makes the same kinds of
  errors on hard Russian words for both the fork and the baseline.

### 2. `benchmark_cuda.py` — full-generation time, TTFC, peak VRAM

Optional speedup-vs-upstream column when
`CHATTERBOX_REGRESSION_BASELINE_PYTHON` is set.

### 3. `benchmark_rtf.py` — streaming RTF, TTFC p50/p95, per-chunk decode

Must stay below 1.0 for `ChatterboxMultilingualTTS` on the documented
text fixtures.

### Benchmark results

All benchmark runs persist JSON in `benchmarks/results/` tagged with git
commit and GPU model. Browse history with
`benchmarks._results.load_history(...)`. See
[`benchmarks/README.md`](benchmarks/README.md) for the full surface.

### Offline weights (no Hugging Face at runtime)

```bash
uv run python download_models.py --model mtl
export CHATTERBOX_MTL_MODEL_DIR=~/.local/share/chatterbox-models/chatterbox
```

---

## Pre-PR checklist

- [ ] No edits to version pins in `pyproject.toml`.
- [ ] If T3 was touched, `T3Graph` static buffer shapes still match the
      model: `input_buf` uses `hidden_size`, `logits_buf` uses
      `speech_tokens_dict_size`.
- [ ] If the S3Gen estimator was touched, `S3GenGraph` static buffer
      shapes (`B=2`, `max_frames`) still match the new estimator I/O.
- [ ] The decode hot path has no `.item()` / `.tolist()` / `print` /
      Python branch on tensor values.
- [ ] `verify_regression.py` passes: `mtl_ru` (MSE < 1e-3) and `streaming`
      green; `upstream_quality` either green or SKIPPED.
- [ ] `StaticCache.reset()` is called before every new prefill.
- [ ] No new `randn` introduced into the CFM ODE start.
- [ ] Default `n_cfm_timesteps` in
      [`s3gen.py`](src/chatterbox/models/s3gen/s3gen.py) is still `2`.
- [ ] Any new RU WER measurement uses `whisper.load_model("medium")`
      (never `"small"`).

---

## Troubleshooting

- **"Illegal memory access" after editing T3** — the size of `input_buf` or
  `logits_buf` in `T3Graph` no longer matches the model's tensors.
- **Performance regressed to baseline** — `T3Graph.capture()` is probably
  being called per step. Capture must happen once, at model init.
- **`uv sync` fails on index** — the `[[tool.uv.index]] pytorch-cu128`
  block was removed or corrupted. PyTorch CUDA 12.8 wheels come from
  that index only.
- **OOM** — check `max_seq_len` in `T3Graph` / `StaticCache`. With
  `batch_size=2` (MTL) the cache cost is doubled.
