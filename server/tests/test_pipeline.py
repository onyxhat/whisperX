import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class _FakeOptions:
    """Stand-in for faster-whisper's TranscriptionOptions (only the fields we touch)."""

    initial_prompt: str | None = None
    temperatures: list = None


@pytest.fixture
def fake_whisperx(monkeypatch):
    """Install a fake `whisperx` module tree and return a call recorder."""
    calls = {"load_align_model": 0, "align": 0, "assign_word_speakers": 0, "diarize_ctor": 0}

    wx = types.ModuleType("whisperx")
    wx_diarize = types.ModuleType("whisperx.diarize")

    class _ASR:
        def __init__(self):
            self.options = _FakeOptions()

        def transcribe(self, audio, batch_size, task, language):
            calls["transcribe"] = {
                "batch_size": batch_size, "task": task, "language": language,
                "initial_prompt": self.options.initial_prompt,
                "temperatures": self.options.temperatures,
            }
            return {"language": language or "en",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "hi",
                                  "words": [{"word": "hi", "start": 0.0, "end": 1.0}]}]}

    def load_model(name, device, compute_type, language, vad_method, download_root, threads):
        calls["load_model"] = {"name": name, "device": device, "compute_type": compute_type}
        return _ASR()

    def load_audio(path):
        calls["load_audio"] = path
        return np.zeros(16000, dtype=np.float32)

    def load_align_model(lang, device, model_dir=None):
        calls["load_align_model"] += 1
        return object(), {"language": lang}

    def align(segments, model_a, meta, audio, device, return_char_alignments):
        calls["align"] += 1
        return {"language": meta["language"], "segments": [{**s} for s in segments]}

    def assign_word_speakers(diar, result):
        calls["assign_word_speakers"] += 1
        for s in result["segments"]:
            s["speaker"] = "SPEAKER_00"
            for w in s.get("words", []):
                w["speaker"] = "SPEAKER_00"
        return result

    class DiarizationPipeline:
        def __init__(self, model_name, token, device, cache_dir):
            calls["diarize_ctor"] += 1
            if token is None:
                raise RuntimeError("no token")
        def __call__(self, audio, min_speakers=None, max_speakers=None):
            return "DIAR_DF"

    wx.load_model = load_model
    wx.load_audio = load_audio
    wx.load_align_model = load_align_model
    wx.align = align
    wx.assign_word_speakers = assign_word_speakers
    wx_diarize.DiarizationPipeline = DiarizationPipeline
    wx.diarize = wx_diarize

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None, is_available=lambda: False)
    class _OOM(Exception): ...
    fake_torch.cuda.OutOfMemoryError = _OOM

    # `server.pipeline` binds `whisperx` at its first import (module top-level
    # `import whisperx`). If any earlier test imported it, that name is already
    # bound to the real module and a plain `sys.modules` swap here would not take.
    # Patch the attribute on the (possibly already-imported) module directly so
    # the fixture is order-independent; keep the `sys.modules` entries too so the
    # tests' own `import whisperx` resolves to the very same fake object.
    import server.pipeline as _pipeline_mod

    monkeypatch.setattr(_pipeline_mod, "whisperx", wx)
    monkeypatch.setitem(sys.modules, "whisperx", wx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", wx_diarize)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return calls


def _settings(**over):
    from server.config import Settings
    return Settings(**over)


def _job(**over):
    from server.models import Job
    base = dict(audio_bytes=b"RIFF....", filename="a.wav", task="transcribe",
                language="en", initial_prompt=None, temperature=0.0, diarize=False,
                min_speakers=None, max_speakers=None, want_word_timestamps=True)
    base.update(over)
    return Job(**base)


def test_run_transcribe_then_align(fake_whisperx):
    from server.pipeline import WhisperXPipeline
    p = WhisperXPipeline(_settings())
    res = p.run(_job())
    assert fake_whisperx["align"] == 1
    assert res["task"] == "transcribe"
    assert res["language"] == "en"
    assert res["segments"][0]["text"] == "hi"
    assert res["words"] == [{"word": "hi", "start": 0.0, "end": 1.0}]
    assert res["duration"] == pytest.approx(1.0)


def test_initial_prompt_reaches_asr_options(fake_whisperx):
    from server.pipeline import WhisperXPipeline
    p = WhisperXPipeline(_settings())
    p.run(_job(initial_prompt="Acme Corp", temperature=0.4))
    assert fake_whisperx["transcribe"]["initial_prompt"] == "Acme Corp"
    assert fake_whisperx["transcribe"]["temperatures"] == [0.4]
    # A job without a prompt must reset it (not leak the previous call's value).
    p.run(_job(initial_prompt=None, temperature=0.0))
    assert fake_whisperx["transcribe"]["initial_prompt"] is None
    assert fake_whisperx["transcribe"]["temperatures"] == [0.0]


def test_translate_skips_alignment(fake_whisperx):
    from server.pipeline import WhisperXPipeline
    p = WhisperXPipeline(_settings())
    p.run(_job(task="translate"))
    assert fake_whisperx["align"] == 0


def test_diarize_calls_assign_word_speakers(fake_whisperx):
    from server.pipeline import WhisperXPipeline
    p = WhisperXPipeline(_settings(HF_TOKEN="hf_x"))
    res = p.run(_job(diarize=True))
    assert fake_whisperx["assign_word_speakers"] == 1
    assert res["segments"][0]["speaker"] == "SPEAKER_00"


def test_diarize_without_token_raises_diarization_error(fake_whisperx):
    from server.pipeline import WhisperXPipeline
    from server.errors import DiarizationError
    p = WhisperXPipeline(_settings())  # no HF token
    with pytest.raises(DiarizationError):
        p.run(_job(diarize=True))


def test_diarize_oom_surfaces_as_inference_error(fake_whisperx, monkeypatch):
    import torch
    import whisperx

    from server.errors import DiarizationError, InferenceError
    from server.pipeline import WhisperXPipeline

    class _OomPipe:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            raise torch.cuda.OutOfMemoryError("cuda oom during diarization")

    monkeypatch.setattr(whisperx.diarize, "DiarizationPipeline", _OomPipe)
    p = WhisperXPipeline(_settings(HF_TOKEN="hf_x"))
    with pytest.raises(InferenceError):
        p.run(_job(diarize=True))
    # And specifically NOT the broad 502 mapping.
    try:
        p.run(_job(diarize=True))
    except DiarizationError:  # pragma: no cover - would be the bug
        pytest.fail("OOM was mis-mapped to DiarizationError")
    except InferenceError:
        pass


def test_undecodable_audio_raises_audio_decode_error(fake_whisperx, monkeypatch):
    import whisperx
    from server.pipeline import WhisperXPipeline
    from server.errors import AudioDecodeError

    def boom(path):
        raise RuntimeError("ffmpeg failed")
    monkeypatch.setattr(whisperx, "load_audio", boom)
    p = WhisperXPipeline(_settings())
    with pytest.raises(AudioDecodeError):
        p.run(_job())


def test_align_cache_reuses_model_across_calls(fake_whisperx):
    from server.pipeline import WhisperXPipeline
    p = WhisperXPipeline(_settings(WHISPERX_ALIGN_CACHE="true"))
    p.run(_job())
    p.run(_job())
    assert fake_whisperx["load_align_model"] == 1  # loaded once, reused


def test_tempfile_is_deleted(fake_whisperx, monkeypatch):
    import whisperx
    from server.pipeline import WhisperXPipeline
    seen = {}
    real_load = whisperx.load_audio
    def capture(path):
        seen["path"] = path
        return real_load(path)
    monkeypatch.setattr(whisperx, "load_audio", capture)
    WhisperXPipeline(_settings()).run(_job())
    import os
    assert not os.path.exists(seen["path"])
