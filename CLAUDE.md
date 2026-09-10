# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

WhisperX: time-accurate ASR. Wraps `faster-whisper` (CTranslate2 backend) with VAD-based
batching, wav2vec2 forced alignment for word-level timestamps, and optional `pyannote.audio`
speaker diarization. Pure Python package (`whisperx/`), no build step, `uv`-managed.

## Commands

```bash
uv sync --all-extras --dev          # dev environment (needs ffmpeg on PATH)
uv run pytest tests/ -v             # full test suite
uv run pytest tests/test_word_timestamp_interpolation.py::TestAlignWithWildcards::test_known_chars_get_timestamps -v   # single test
uv run whisperx path/to/audio.wav --compute_type int8 --device cpu   # run the CLI locally
uv run python -c "import whisperx"  # import smoke test (CI gate)
uv lock --check                     # CI fails if uv.lock is stale; run `uv lock` after any dependency edit
uv build                            # build wheel/sdist (release only)
```

No linter/formatter is configured. CI (`.github/workflows/`) runs the test suite and an
`import whisperx` check across Python 3.10–3.13, a `zizmor` GitHub-Actions security lint,
and on GitHub release publish builds + pushes to PyPI.

## Dependency resolution is platform-conditional

`pyproject.toml` `[tool.uv.sources]` selects the torch build by platform: CPU wheels on
macOS and non-x86 Linux, CUDA 12.8 (`cu128` index) on x86_64/AMD64 Linux + Windows.
`triton` installs only on x86_64 Linux; `torchcodec` is excluded on Linux aarch64.
torch is pinned `~=2.8.0`. `requires-python = ">=3.10, <3.14"`. Editing any dependency
means regenerating `uv.lock` or CI's `uv lock --check` step fails.

## Pipeline architecture

Three stages, orchestrated by `transcribe.py:transcribe_task` (called from
`__main__.py:cli`, where every CLI flag is defined):

1. **VAD + batched ASR** (`asr.py`). `load_model()` returns a `FasterWhisperPipeline`
   (subclass of HF `transformers.Pipeline`) wrapping a `WhisperModel` (subclass of
   `faster_whisper.WhisperModel`) plus a VAD. VAD speech regions are merged into
   ~`chunk_size` (30s) windows by `vads/vad.py:Vad.merge_chunks`, then decoded in
   batches by `generate_segment_batched`. ASR runs `without_timestamps` (one forward
   pass per batch item) and `condition_on_previous_text=False` — both deliberate, and
   both cause divergence from vanilla OpenAI Whisper output.
2. **Forced alignment** (`alignment.py`). `load_align_model()` picks a wav2vec2 model by
   language code from `DEFAULT_ALIGN_MODELS_TORCH` / `DEFAULT_ALIGN_MODELS_HF`. `align()`
   does Viterbi forced alignment (`get_trellis` → `backtrack` → `merge_repeats` →
   `merge_words`) to attach per-word and per-char timestamps. Characters absent from the
   model dictionary (digits, currency) get no direct timing and are filled via
   `utils.py:interpolate_nans` per `--interpolate_method`. `--task translate` disables
   alignment.
3. **Diarization** (`diarize.py`, optional, `--diarize`). `DiarizationPipeline` wraps the
   gated `pyannote/speaker-diarization-community-1` model (needs `--hf_token`).
   `assign_word_speakers` maps speaker turns onto words/segments via an `IntervalTree`.

Each stage loads its model, runs over all inputs, then `del model; gc.collect();
torch.cuda.empty_cache()` before the next stage — this staged flushing is what keeps peak
GPU memory low, so preserve it when editing `transcribe_task`.

## Key modules

- `schema.py` — `TypedDict` contracts (`SingleSegment`, `TranscriptionResult`,
  `AlignedTranscriptionResult`, `SingleWordSegment`, …). This is the data handed between
  stages; change it and all three stages plus the writers are affected.
- `__init__.py` — public API via `_lazy_import`; keeps `import whisperx` cheap by deferring
  torch-heavy submodule imports. Add new public entry points the same lazy way.
- `vads/` — `Vad` base + `Pyannote` / `Silero` implementations, chosen by `--vad_method`.
- `utils.py` — output writers (`get_writer` → srt/vtt/txt/tsv/json/aud, `ResultWriter`
  subclasses) and argparse helpers (`str2bool`, `optional_int`, `optional_float`).
- `SubtitlesProcessor.py` + `conjunctions.py` — sentence/subtitle segmentation with
  per-language conjunction lists.
- `audio.py` — `load_audio` shells out to ffmpeg (16 kHz mono) and mel-spectrogram helpers.
- `log_utils.py` — `setup_logging` / `get_logger`; `--log-level` overrides `--verbose`.

## Notes

- `~/.claude/CLAUDE.md` and `~/CLAUDE.md` on this machine contain unrelated "RuFlo/claude-flow"
  boilerplate that does not apply to this repository — ignore its swarm/agent directives here.
- Models download to `~/.cache` (Whisper/HF) unless `--model_dir` is set; `--model_cache_only`
  forces offline use.
- GPU cuDNN load errors: see `CUDNN_TROUBLESHOOTING.md`.
