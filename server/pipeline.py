from __future__ import annotations

import gc
import os
import tempfile
from dataclasses import replace
from pathlib import Path

import whisperx

from server.config import Settings
from server.errors import AudioDecodeError, DiarizationError, InferenceError
from server.models import Job, PipelineResult

_SAMPLE_RATE = 16000


class WhisperXPipeline:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._asr = whisperx.load_model(
            settings.whisperx_model,
            settings.device,
            compute_type=settings.compute_type,
            language=settings.language,
            vad_method=settings.whisperx_vad_method,
            download_root=settings.whisperx_model_dir,
            threads=4,
        )
        self._align_cache: tuple[str, object, dict] | None = None

    def close(self) -> None:
        self._asr = None
        self._align_cache = None
        _empty_cache()

    def _get_align_model(self, lang: str):
        if self._align_cache and self._align_cache[0] == lang:
            return self._align_cache[1], self._align_cache[2]
        model_a, meta = whisperx.load_align_model(
            lang, self._settings.device, model_dir=self._settings.whisperx_model_dir
        )
        if self._settings.align_cache:
            self._align_cache = (lang, model_a, meta)
        return model_a, meta

    def run(self, job: Job) -> PipelineResult:
        suffix = Path(job.filename or "audio").suffix or ".wav"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(job.audio_bytes)
            return self._run_on_path(job, tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _run_on_path(self, job: Job, path: str) -> PipelineResult:
        import torch

        try:
            audio = whisperx.load_audio(path)
        except Exception as exc:  # noqa: BLE001 — any decode failure is a client error
            raise AudioDecodeError("could not decode audio file", param="file") from exc

        duration = len(audio) / _SAMPLE_RATE
        s = self._settings

        try:
            # Apply the OpenAI `prompt` / `temperature` fields (carried on the Job) to
            # the resident model's TranscriptionOptions on every call — so they are also
            # reset when a job omits them. `temperatures` (plural) is faster-whisper's
            # fallback temperature schedule; a single-element list pins the decode
            # temperature. Mirrors asr.py's own `self.options = replace(self.options, ...)`
            # pattern; the inference worker is serial, so there is no race on the
            # shared options object.
            self._asr.options = replace(
                self._asr.options,
                initial_prompt=job.initial_prompt,
                temperatures=[job.temperature],
            )
            result = self._asr.transcribe(
                audio, batch_size=s.whisperx_batch_size, task=job.task, language=job.language
            )
            lang = result.get("language", job.language or "en")

            if job.task != "translate":
                model_a, meta = self._get_align_model(lang)
                try:
                    result = whisperx.align(
                        result["segments"], model_a, meta, audio, s.device,
                        return_char_alignments=False,
                    )
                finally:
                    if not s.align_cache:
                        del model_a
                        gc.collect()
                        _empty_cache()
                lang = result.get("language", lang)

            if job.diarize:
                result = self._diarize(job, audio, result)

        except torch.cuda.OutOfMemoryError as exc:
            _empty_cache()
            raise InferenceError(
                "CUDA out of memory — try a smaller WHISPERX_MODEL or WHISPERX_BATCH_SIZE"
            ) from exc

        segments = [{**seg, "words": seg.get("words", [])} for seg in result["segments"]]
        words: list[dict] = []
        for seg in segments:
            words.extend(seg["words"])

        return PipelineResult(
            task=job.task, language=lang, duration=duration,
            temperature=job.temperature, segments=segments, words=words,
        )

    def _diarize(self, job: Job, audio, result: dict) -> dict:
        import torch
        import whisperx.diarize  # ensure the submodule/attribute is available at runtime

        s = self._settings
        dp = None
        try:
            dp = whisperx.diarize.DiarizationPipeline(
                model_name=s.whisperx_diarize_model, token=s.hf_token,
                device=s.device, cache_dir=s.whisperx_model_dir,
            )
            diar = dp(audio, min_speakers=job.min_speakers, max_speakers=job.max_speakers)
        except torch.cuda.OutOfMemoryError:
            # Let the caller map this to the 500 OOM message rather than a 502
            # "diarization model unavailable".
            raise
        except Exception as exc:  # noqa: BLE001
            raise DiarizationError(
                "diarization model unavailable — check HF token and model-gate acceptance"
            ) from exc
        finally:
            if dp is not None:
                del dp
            gc.collect()
            _empty_cache()
        return whisperx.assign_word_speakers(diar, result)


def _empty_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
