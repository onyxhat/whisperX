import pytest

from server.errors import BadRequestError
from server.formats import render


def _result(diarized=False):
    seg = {
        "start": 0.0, "end": 1.5, "text": " Hello world.",
        "avg_logprob": -0.21,
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.6, "score": 0.9},
            {"word": "world.", "start": 0.7, "end": 1.5, "score": 0.8},
        ],
    }
    if diarized:
        seg["speaker"] = "SPEAKER_00"
        for w in seg["words"]:
            w["speaker"] = "SPEAKER_00"
    return {
        "task": "transcribe", "language": "en", "duration": 1.53,
        "temperature": 0.0,
        "segments": [seg],
        "words": list(seg["words"]),
    }


def test_text_format_returns_plain_transcript():
    body, media = render(_result(), "text", want_words=False, diarized=False)
    assert body == "Hello world."
    assert media.startswith("text/plain")


def test_json_format_returns_text_object():
    body, media = render(_result(), "json", want_words=False, diarized=False)
    assert body == {"text": "Hello world."}
    assert media == "application/json"


def test_verbose_json_has_openai_keys_and_fillers():
    body, _ = render(_result(), "verbose_json", want_words=False, diarized=False)
    assert body["task"] == "transcribe"
    assert body["language"] == "en"
    assert body["duration"] == 1.53
    assert body["text"] == "Hello world."
    seg = body["segments"][0]
    assert seg["id"] == 0
    assert seg["seek"] == 0
    assert seg["tokens"] == []
    assert seg["temperature"] == 0.0
    assert seg["avg_logprob"] == -0.21
    assert seg["compression_ratio"] is None
    assert seg["no_speech_prob"] is None
    assert "speaker" not in seg
    assert "words" not in body  # want_words False


def test_verbose_json_includes_words_when_requested():
    body, _ = render(_result(), "verbose_json", want_words=True, diarized=False)
    assert [w["word"] for w in body["words"]] == ["Hello", "world."]
    assert body["words"][0]["start"] == 0.0
    assert "speaker" not in body["words"][0]


def test_verbose_json_speaker_only_when_diarized():
    body, _ = render(_result(diarized=True), "verbose_json", want_words=True, diarized=True)
    assert body["segments"][0]["speaker"] == "SPEAKER_00"
    assert body["words"][0]["speaker"] == "SPEAKER_00"


def test_word_without_timing_is_tolerated():
    r = _result()
    r["words"].append({"word": "2014.", "start": None, "end": None})
    body, _ = render(r, "verbose_json", want_words=True, diarized=False)
    assert body["words"][-1]["start"] is None


def test_srt_output_is_numbered_cues():
    body, media = render(_result(), "srt", want_words=False, diarized=False)
    assert body.splitlines()[0] == "1"
    assert "-->" in body
    assert media.startswith("text/plain")


def test_vtt_output_has_header():
    body, _ = render(_result(), "vtt", want_words=False, diarized=False)
    assert body.startswith("WEBVTT")


def test_unknown_format_raises_bad_request():
    with pytest.raises(BadRequestError):
        render(_result(), "flac", want_words=False, diarized=False)
