from __future__ import annotations

import io

from server.errors import BadRequestError
from server.models import PipelineResult

_TEXT = "text/plain; charset=utf-8"
_JSON = "application/json"


def _full_text(result: PipelineResult) -> str:
    return "".join(s["text"] for s in result["segments"]).strip()


def _round(v, n=3):
    return round(v, n) if isinstance(v, (int, float)) else v


def to_text(result: PipelineResult) -> str:
    return _full_text(result)


def to_json(result: PipelineResult) -> dict:
    return {"text": _full_text(result)}


def _segment(i: int, s: dict, temperature: float, diarized: bool) -> dict:
    out = {
        "id": i,
        "seek": 0,
        "start": _round(s.get("start")),
        "end": _round(s.get("end")),
        "text": s.get("text", ""),
        "tokens": [],
        "temperature": temperature,
        "avg_logprob": s.get("avg_logprob"),
        "compression_ratio": None,
        "no_speech_prob": None,
    }
    if diarized:
        out["speaker"] = s.get("speaker")
    return out


def _word(w: dict, diarized: bool) -> dict:
    out = {"word": w.get("word", ""), "start": _round(w.get("start")), "end": _round(w.get("end"))}
    if diarized:
        out["speaker"] = w.get("speaker")
    return out


def to_verbose_json(result: PipelineResult, want_words: bool, diarized: bool) -> dict:
    out = {
        "task": result["task"],
        "language": result["language"],
        "duration": _round(result["duration"]),
        "text": _full_text(result),
        "segments": [
            _segment(i, s, result["temperature"], diarized)
            for i, s in enumerate(result["segments"])
        ],
    }
    if want_words:
        out["words"] = [_word(w, diarized) for w in result["words"]]
    return out


def _write_subtitles(result: PipelineResult, writer_cls) -> str:
    buf = io.StringIO()
    wx_result = {
        "segments": [{**s, "words": s.get("words", [])} for s in result["segments"]],
        "language": result["language"],
    }
    options = {"highlight_words": False, "max_line_width": None, "max_line_count": None}
    writer_cls(".").write_result(wx_result, buf, options)
    return buf.getvalue().strip() + "\n"


def to_srt(result: PipelineResult) -> str:
    from whisperx.utils import WriteSRT

    return _write_subtitles(result, WriteSRT)


def to_vtt(result: PipelineResult) -> str:
    from whisperx.utils import WriteVTT

    return _write_subtitles(result, WriteVTT)


def render(result: PipelineResult, fmt: str, want_words: bool, diarized: bool):
    if fmt == "text":
        return to_text(result), _TEXT
    if fmt == "json":
        return to_json(result), _JSON
    if fmt == "verbose_json":
        return to_verbose_json(result, want_words, diarized), _JSON
    if fmt == "srt":
        return to_srt(result), _TEXT
    if fmt == "vtt":
        return to_vtt(result), _TEXT
    raise BadRequestError(f"unknown response_format '{fmt}'", param="response_format")
