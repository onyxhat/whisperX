# WhisperX OpenAI-Compatible API Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve WhisperX as a long-lived container exposing an OpenAI-compatible transcription API with optional pyannote diarization.

**Architecture:** A standalone FastAPI app in a new top-level `server/` package imports `whisperx` as a library and calls the same public functions the CLI uses. One background asyncio worker owns the GPU; HTTP handlers enqueue a job and await it synchronously. The ASR model is loaded once at startup and kept resident; alignment and diarization models load per request and are freed after. Ships as one CUDA-based Docker image that also runs CPU-only.

**Tech Stack:** Python 3.12, `uv` for all dependency management, `whisperx` (this repo) + `faster-whisper`/`pyannote-audio`/`torch~=2.8` (cu128), `fastapi` + `uvicorn[standard]` + `python-multipart` + `pydantic-settings` + `anyio`, `pytest` + `fastapi.testclient`.

**Spec:** [docs/superpowers/specs/2026-09-10-whisperx-openai-api-server-design.md](../specs/2026-09-10-whisperx-openai-api-server-design.md)

## Global Constraints

- Python target `3.12`; repo supports `>=3.10, <3.14`. Do not narrow it.
- **`uv` only** for dependency and environment management — never `pip` / `pip install` directly (use `uv pip` when a lockfile-free install is needed).
- **Do not modify** the `whisperx/` package, its CLI, or `pyproject.toml` / `uv.lock`. Reuse `whisperx` only through its public functions.
- Server dependencies live **only** in `server/requirements.txt`, installed into the project venv with `uv pip install -r server/requirements.txt`.
- All error responses use the OpenAI envelope: `{"error": {"message", "type", "param", "code"}}`.
- One uvicorn worker, one FIFO queue. Synchronous responses only — no job API, no streaming.
- `/config` is the container's persistent cache root; `HF_HOME` and `TORCH_HOME` sit under it.
- Keep each source file focused and well under 500 lines.
- TDD: write the failing test first, watch it fail, implement minimally, watch it pass, commit.
- Branch: `feature/api` (already checked out). Commit after every task.
- End every commit message with:
  `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`

## Architecture note — deviation from the spec

The spec places `Job` and `PipelineResult` in `pipeline.py`. This plan puts them (and a `Pipeline` Protocol) in a new stdlib-only `server/models.py` instead, so `formats.py`, `schemas.py`, `worker.py`, and `app.py` stay free of any `torch`/`whisperx` import. This strengthens the spec's stated boundary ("`app.py` never imports torch") and lets each unit be tested without loading CUDA. No behavioural change.

## File structure

```
server/
  __init__.py          # empty; marks the package
  __main__.py          # `python -m server` → uvicorn.run(create_app, factory=True)
  models.py            # Task literal, Job dataclass, PipelineResult TypedDict, Pipeline Protocol  (stdlib only)
  config.py            # Settings (pydantic-settings) + resolved device/compute/language + startup validation
  errors.py            # APIError hierarchy, to_envelope(), register_exception_handlers(app)
  formats.py           # render(result, fmt, want_words, diarized) -> (body, media_type)  (pure)
  auth.py              # require_auth FastAPI dependency (no-op unless API_KEY set)
  pipeline.py          # WhisperXPipeline: resident ASR + per-job align/diarize   (imports whisperx/torch)
  worker.py            # InferenceWorker: asyncio.Queue + single consumer
  schemas.py           # ResponseFormat enum, ModelObject / ModelList response models
  app.py               # create_app() factory, lifespan, /health, /v1/models, /v1/audio/*
  requirements.txt
  entrypoint.sh
  tests/
    conftest.py        # gpu marker, wav_bytes, FakePipeline, make_client factory
    test_config.py
    test_errors.py
    test_formats.py
    test_pipeline.py
    test_worker.py
    test_auth.py
    test_app_health_models.py
    test_transcriptions.py
    test_smoke.py       # @pytest.mark.gpu, skipped unless WHISPERX_SMOKE=1
Dockerfile             # repo root
docker-compose.yml     # repo root
.dockerignore          # repo root
README.md              # edited (fork section)
.github/workflows/tests.yml   # edited (run server/tests)
```

---

### Task 1: Package scaffold, `server/requirements.txt`, `models.py`, `config.py`

**Files:**
- Create: `server/__init__.py` (empty)
- Create: `server/requirements.txt`
- Create: `server/models.py`
- Create: `server/config.py`
- Create: `server/tests/conftest.py` (partial — only what this task needs)
- Test: `server/tests/test_config.py`

**Interfaces:**
- Produces:
  - `server/models.py`:
    - `Task = typing.Literal["transcribe", "translate"]`
    - `@dataclass class Job` with fields: `audio_bytes: bytes`, `filename: str`, `task: Task`, `language: str | None`, `initial_prompt: str | None`, `temperature: float`, `diarize: bool`, `min_speakers: int | None`, `max_speakers: int | None`, `want_word_timestamps: bool`, `abandoned: bool = False`
    - `class PipelineResult(TypedDict)`: `task: str`, `language: str`, `duration: float`, `temperature: float`, `segments: list[dict]`, `words: list[dict]`
    - `class Pipeline(typing.Protocol)`: `def run(self, job: Job) -> PipelineResult: ...` and `def close(self) -> None: ...`
  - `server/config.py`:
    - `class Settings(BaseSettings)` — raw env fields (defaults from the spec's config table): `whisperx_model="small"`, `whisperx_device="auto"`, `whisperx_compute_type="auto"`, `whisperx_batch_size=8`, `whisperx_language="en"`, `whisperx_vad_method="pyannote"`, `whisperx_align_cache=False`, `whisperx_model_dir="/config"`, `whisperx_diarize_model="pyannote/speaker-diarization-community-1"`, `hf_token: str | None = None`, `api_key: str | None = None`, `max_queue=16`, `max_upload_mb=200`, `request_timeout_s=1800`, `host="0.0.0.0"`, `port=8000`, `log_level="info"`
    - properties: `device -> str`, `compute_type -> str`, `language -> str | None` (`None` when `whisperx_language == "auto"`), `align_cache -> bool`
    - `def validate_runtime(self) -> None` — raises `ValueError` for impossible combos
    - `def _cuda_available() -> bool` (module-level, monkeypatchable: `import torch; return torch.cuda.is_available()`)
    - `def load_settings() -> Settings` — `s = Settings(); s.validate_runtime(); return s`

- [ ] **Step 1: Write the failing test**

`server/tests/test_config.py`:

```python
import pytest
from server import config
from server.config import Settings


def _settings(monkeypatch, cuda: bool, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    monkeypatch.setattr(config, "_cuda_available", lambda: cuda)
    return Settings()


def test_defaults_language_is_english(monkeypatch):
    s = _settings(monkeypatch, cuda=True)
    assert s.whisperx_language == "en"
    assert s.language == "en"


def test_language_auto_sentinel_means_detect(monkeypatch):
    s = _settings(monkeypatch, cuda=True, WHISPERX_LANGUAGE="auto")
    assert s.language is None


def test_device_auto_resolves_to_cuda_when_available(monkeypatch):
    s = _settings(monkeypatch, cuda=True)
    assert s.device == "cuda"
    assert s.compute_type == "float16"


def test_device_auto_resolves_to_cpu_without_gpu(monkeypatch):
    s = _settings(monkeypatch, cuda=False)
    assert s.device == "cpu"
    assert s.compute_type == "int8"


def test_explicit_compute_type_wins(monkeypatch):
    s = _settings(monkeypatch, cuda=True, WHISPERX_COMPUTE_TYPE="float32")
    assert s.compute_type == "float32"


def test_model_dir_default_is_config(monkeypatch):
    s = _settings(monkeypatch, cuda=False)
    assert s.whisperx_model_dir == "/config"


def test_validate_runtime_rejects_float16_on_cpu(monkeypatch):
    s = _settings(monkeypatch, cuda=False, WHISPERX_DEVICE="cpu",
                  WHISPERX_COMPUTE_TYPE="float16")
    with pytest.raises(ValueError, match="float16.*cpu"):
        s.validate_runtime()


def test_validate_runtime_rejects_unknown_compute_type(monkeypatch):
    s = _settings(monkeypatch, cuda=True, WHISPERX_COMPUTE_TYPE="bogus")
    with pytest.raises(ValueError, match="compute_type"):
        s.validate_runtime()


def test_api_key_and_hf_token_read_from_env(monkeypatch):
    s = _settings(monkeypatch, cuda=True, API_KEY="secret", HF_TOKEN="hf_x")
    assert s.api_key == "secret"
    assert s.hf_token == "hf_x"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest server/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.config'` (and `pydantic_settings` not yet installed).

- [ ] **Step 3: Add server dependencies and install**

`server/requirements.txt`:
```
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
pydantic-settings>=2.4
anyio>=4
```

Run: `uv pip install -r server/requirements.txt`
(Do **not** add these to `pyproject.toml`.)

- [ ] **Step 4: Write minimal implementation**

`server/__init__.py`: empty file.

`server/models.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

Task = Literal["transcribe", "translate"]


@dataclass
class Job:
    audio_bytes: bytes
    filename: str
    task: Task
    language: str | None
    initial_prompt: str | None
    temperature: float
    diarize: bool
    min_speakers: int | None
    max_speakers: int | None
    want_word_timestamps: bool
    abandoned: bool = False


class PipelineResult(TypedDict):
    task: str
    language: str
    duration: float
    temperature: float
    segments: list[dict]
    words: list[dict]


class Pipeline(Protocol):
    def run(self, job: Job) -> PipelineResult: ...
    def close(self) -> None: ...
```

`server/config.py`:
```python
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

_VALID_COMPUTE = {"auto", "float16", "float32", "int8"}


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    whisperx_model: str = "small"
    whisperx_device: str = "auto"
    whisperx_compute_type: str = "auto"
    whisperx_batch_size: int = 8
    whisperx_language: str = "en"
    whisperx_vad_method: str = "pyannote"
    whisperx_align_cache: bool = False
    whisperx_model_dir: str = "/config"
    whisperx_diarize_model: str = "pyannote/speaker-diarization-community-1"
    hf_token: str | None = None
    api_key: str | None = None
    max_queue: int = 16
    max_upload_mb: int = 200
    request_timeout_s: int = 1800
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    @property
    def device(self) -> str:
        if self.whisperx_device != "auto":
            return self.whisperx_device
        return "cuda" if _cuda_available() else "cpu"

    @property
    def compute_type(self) -> str:
        if self.whisperx_compute_type != "auto":
            return self.whisperx_compute_type
        return "float16" if self.device == "cuda" else "int8"

    @property
    def language(self) -> str | None:
        return None if self.whisperx_language == "auto" else self.whisperx_language

    @property
    def align_cache(self) -> bool:
        return self.whisperx_align_cache

    def validate_runtime(self) -> None:
        if self.whisperx_compute_type not in _VALID_COMPUTE:
            raise ValueError(
                f"WHISPERX_COMPUTE_TYPE '{self.whisperx_compute_type}' invalid; "
                f"choose from {sorted(_VALID_COMPUTE)}"
            )
        if self.device == "cpu" and self.compute_type == "float16":
            raise ValueError("compute_type float16 is not supported on cpu; use int8 or float32")
        if self.whisperx_vad_method not in {"pyannote", "silero"}:
            raise ValueError("WHISPERX_VAD_METHOD must be 'pyannote' or 'silero'")


def load_settings() -> Settings:
    s = Settings()
    s.validate_runtime()
    return s
```

`server/tests/conftest.py` (create with just the marker for now; fixtures added in Task 7):
```python
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: requires a real GPU and model downloads (set WHISPERX_SMOKE=1)"
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest server/tests/test_config.py -v`
Expected: PASS (10 passed).

- [ ] **Step 6: Confirm the core suite is untouched**

Run: `uv run pytest tests/ -v`
Expected: PASS (unchanged from baseline).

- [ ] **Step 7: Commit**

```bash
git add server/__init__.py server/requirements.txt server/models.py server/config.py server/tests/conftest.py server/tests/test_config.py
git commit -m "feat(server): package scaffold, settings, and shared models

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `server/errors.py` — error hierarchy and OpenAI envelope

**Files:**
- Create: `server/errors.py`
- Test: `server/tests/test_errors.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class APIError(Exception)` — `__init__(self, message: str, *, param: str | None = None, code: str | None = None)`; class attrs `status: int` and `err_type: str`.
  - Subclasses (with `status` / `err_type` set): `AuthError` (401, `invalid_request_error`), `BadRequestError` (400, `invalid_request_error`), `PayloadTooLargeError` (413, `invalid_request_error`), `AudioDecodeError` (400, `invalid_request_error`), `QueueFullError` (429, `rate_limit_error`), `RequestTimeoutError` (504, `timeout_error`), `DiarizationError` (502, `api_error`), `NotReadyError` (503, `api_error`), `InferenceError` (500, `api_error`).
  - `def to_envelope(exc: APIError) -> dict` → `{"error": {"message", "type", "param", "code"}}`.
  - `def register_exception_handlers(app) -> None` — registers a handler for `APIError` (uses its `status`/`to_envelope`) and one for bare `Exception` (500, `api_error`, generic message "internal error", logs the traceback via `logging.getLogger("server").exception`).

- [ ] **Step 1: Write the failing test**

`server/tests/test_errors.py`:

```python
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import errors
from server.errors import (
    APIError, AuthError, BadRequestError, DiarizationError, InferenceError,
    NotReadyError, PayloadTooLargeError, QueueFullError, RequestTimeoutError,
    to_envelope,
)


@pytest.mark.parametrize("exc_cls,status,err_type", [
    (AuthError, 401, "invalid_request_error"),
    (BadRequestError, 400, "invalid_request_error"),
    (PayloadTooLargeError, 413, "invalid_request_error"),
    (QueueFullError, 429, "rate_limit_error"),
    (RequestTimeoutError, 504, "timeout_error"),
    (DiarizationError, 502, "api_error"),
    (NotReadyError, 503, "api_error"),
    (InferenceError, 500, "api_error"),
])
def test_subclass_status_and_type(exc_cls, status, err_type):
    e = exc_cls("boom")
    assert isinstance(e, APIError)
    assert e.status == status
    assert e.err_type == err_type


def test_envelope_shape_includes_param_and_code():
    e = BadRequestError("bad file", param="file", code="missing_file")
    assert to_envelope(e) == {
        "error": {
            "message": "bad file",
            "type": "invalid_request_error",
            "param": "file",
            "code": "missing_file",
        }
    }


def test_envelope_param_and_code_default_to_none():
    assert to_envelope(AuthError("nope"))["error"]["param"] is None
    assert to_envelope(AuthError("nope"))["error"]["code"] is None


def test_handlers_render_apierror_and_generic():
    app = FastAPI()
    errors.register_exception_handlers(app)

    @app.get("/known")
    def known():
        raise QueueFullError("queue full")

    @app.get("/boom")
    def boom():
        raise RuntimeError("unexpected")

    client = TestClient(app, raise_server_exceptions=False)

    r1 = client.get("/known")
    assert r1.status_code == 429
    assert r1.json()["error"]["type"] == "rate_limit_error"

    r2 = client.get("/boom")
    assert r2.status_code == 500
    body = r2.json()
    assert body["error"]["type"] == "api_error"
    assert "unexpected" not in body["error"]["message"]  # internal detail not leaked
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest server/tests/test_errors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.errors'`.

- [ ] **Step 3: Write minimal implementation**

`server/errors.py`:
```python
from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("server")


class APIError(Exception):
    status: int = 500
    err_type: str = "api_error"

    def __init__(self, message: str, *, param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code


class AuthError(APIError):
    status, err_type = 401, "invalid_request_error"


class BadRequestError(APIError):
    status, err_type = 400, "invalid_request_error"


class PayloadTooLargeError(APIError):
    status, err_type = 413, "invalid_request_error"


class AudioDecodeError(APIError):
    status, err_type = 400, "invalid_request_error"


class QueueFullError(APIError):
    status, err_type = 429, "rate_limit_error"


class RequestTimeoutError(APIError):
    status, err_type = 504, "timeout_error"


class DiarizationError(APIError):
    status, err_type = 502, "api_error"


class NotReadyError(APIError):
    status, err_type = 503, "api_error"


class InferenceError(APIError):
    status, err_type = 500, "api_error"


def to_envelope(exc: APIError) -> dict:
    return {
        "error": {
            "message": exc.message,
            "type": exc.err_type,
            "param": exc.param,
            "code": exc.code,
        }
    }


def register_exception_handlers(app) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError):
        if exc.status >= 500:
            logger.error("APIError %s: %s", exc.status, exc.message)
        else:
            logger.warning("APIError %s: %s", exc.status, exc.message)
        return JSONResponse(status_code=exc.status, content=to_envelope(exc))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "internal error", "type": "api_error",
                               "param": None, "code": None}},
        )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest server/tests/test_errors.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/errors.py server/tests/test_errors.py
git commit -m "feat(server): error hierarchy and OpenAI error envelope

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: `server/formats.py` — result → OpenAI response bodies

**Files:**
- Create: `server/formats.py`
- Test: `server/tests/test_formats.py`

**Interfaces:**
- Consumes: `server.models.PipelineResult`; `whisperx.utils.WriteSRT`, `whisperx.utils.WriteVTT`.
- Produces:
  - `def render(result: PipelineResult, fmt: str, want_words: bool, diarized: bool) -> tuple[str | dict, str]` — returns `(body, media_type)` where `media_type` is `"application/json"` for `json`/`verbose_json` (body is a `dict`) and `"text/plain; charset=utf-8"` for `text`/`srt`/`vtt` (body is a `str`). Unknown `fmt` raises `server.errors.BadRequestError(param="response_format")`.
  - Module-internal: `to_text`, `to_json`, `to_verbose_json`, `to_srt`, `to_vtt`.

**Notes for the implementer:**
- `WriteSRT(".").write_result(whisperx_result, string_io, options)` — `options` needs keys `highlight_words` (False), `max_line_width` (None), `max_line_count` (None). `whisperx_result` is `{"segments": result["segments"], "language": result["language"]}`. Each segment must carry a `"words"` key (list; may be empty). Verify the exact key set `WriteSRT`/`SubtitlesWriter.iterate_result` touches in `whisperx/utils.py` and adjust the options dict if it reads more keys.
- Word entries may have `start`/`end` == `None` (unalignable tokens); round only when not `None`.

- [ ] **Step 1: Write the failing test**

`server/tests/test_formats.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest server/tests/test_formats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.formats'`.

- [ ] **Step 3: Write minimal implementation**

`server/formats.py`:
```python
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
    from whisperx.utils import WriteSRT, WriteVTT  # noqa: F401 (import check)

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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest server/tests/test_formats.py -v`
Expected: PASS. If `test_srt_output_is_numbered_cues` / `test_vtt_output_has_header` fail on an options KeyError, inspect `whisperx/utils.py:SubtitlesWriter.iterate_result` and add the missing option keys, then re-run.

- [ ] **Step 5: Commit**

```bash
git add server/formats.py server/tests/test_formats.py
git commit -m "feat(server): render pipeline results to OpenAI response formats

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: `server/pipeline.py` — `WhisperXPipeline`

**Files:**
- Create: `server/pipeline.py`
- Test: `server/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `server.config.Settings`; `server.models.Job`, `PipelineResult`; `server.errors.AudioDecodeError`, `DiarizationError`, `InferenceError`; `whisperx.load_model`, `whisperx.load_audio`, `whisperx.load_align_model`, `whisperx.align`, `whisperx.assign_word_speakers`, `whisperx.diarize.DiarizationPipeline`.
- Produces:
  - `class WhisperXPipeline`:
    - `__init__(self, settings: Settings)` — calls `whisperx.load_model(settings.whisperx_model, settings.device, compute_type=settings.compute_type, language=settings.language, vad_method=settings.whisperx_vad_method, download_root=settings.whisperx_model_dir, threads=4)`; stores it as `self._asr`; `self._align_cache: tuple[str, object, dict] | None = None`.
    - `run(self, job: Job) -> PipelineResult`
    - `close(self) -> None`
  - Satisfies `server.models.Pipeline`.

**Behaviour `run()` must implement (mirrors `whisperx/transcribe.py:transcribe_task`, minus argparse/file IO):**
1. Write `job.audio_bytes` to a `NamedTemporaryFile` (suffix from `job.filename`); always `os.unlink` it in `finally`.
2. `audio = whisperx.load_audio(tmp_path)`. Any `Exception` → `AudioDecodeError("could not decode audio file", param="file")`.
3. `duration = len(audio) / 16000`.
4. `result = self._asr.transcribe(audio, batch_size=self._settings.whisperx_batch_size, task=job.task, language=job.language)`.
5. If `job.task != "translate"`: get align model for `result["language"]` (reuse `self._align_cache` when `settings.align_cache` and language matches, else `whisperx.load_align_model(lang, device, model_dir=settings.whisperx_model_dir)`); `result = whisperx.align(result["segments"], model_a, meta, audio, device, return_char_alignments=False)`; if not `settings.align_cache`: `del model_a`, `gc.collect()`, `torch.cuda.empty_cache()`, clear cache slot.
6. If `job.diarize`: `dp = whisperx.diarize.DiarizationPipeline(model_name=settings.whisperx_diarize_model, token=settings.hf_token, device=device, cache_dir=settings.whisperx_model_dir)` — construction/`__call__` failure → `DiarizationError("diarization model unavailable — check HF token and model gate acceptance")`; `diar = dp(audio, min_speakers=job.min_speakers, max_speakers=job.max_speakers)`; `result = whisperx.assign_word_speakers(diar, result)`; `del dp; gc.collect(); torch.cuda.empty_cache()`.
7. Normalise: ensure every segment dict has a `"words"` key (`[]` if absent); build flat `words` = concat of segment `words`.
8. `torch.cuda.OutOfMemoryError` anywhere → `InferenceError("CUDA out of memory — try a smaller WHISPERX_MODEL or WHISPERX_BATCH_SIZE")` after `torch.cuda.empty_cache()`.
9. Return `PipelineResult(task=job.task, language=result.get("language", job.language or "en"), duration=duration, temperature=job.temperature, segments=result["segments"], words=flat_words)`.

- [ ] **Step 1: Write the failing test**

`server/tests/test_pipeline.py`:

```python
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def fake_whisperx(monkeypatch):
    """Install a fake `whisperx` module tree and return a call recorder."""
    calls = {"load_align_model": 0, "align": 0, "assign_word_speakers": 0, "diarize_ctor": 0}

    wx = types.ModuleType("whisperx")
    wx_diarize = types.ModuleType("whisperx.diarize")

    class _ASR:
        def transcribe(self, audio, batch_size, task, language):
            calls["transcribe"] = {"batch_size": batch_size, "task": task, "language": language}
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


def test_undecodable_audio_raises_audio_decode_error(fake_whisperx, monkeypatch):
    import whisperx
    from server.pipeline import WhisperXPipeline
    from server.errors import AudioDecodeError

    def boom(path):
        raise RuntimeError("ffmpeg failed")//
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
```

> Implementer: delete the stray `//` typo on the `boom` line above when transcribing; it is not valid Python.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest server/tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.pipeline'`.

- [ ] **Step 3: Write minimal implementation**

`server/pipeline.py`:
```python
from __future__ import annotations

import gc
import os
import tempfile
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
        s = self._settings
        try:
            dp = whisperx.diarize.DiarizationPipeline(
                model_name=s.whisperx_diarize_model, token=s.hf_token,
                device=s.device, cache_dir=s.whisperx_model_dir,
            )
            diar = dp(audio, min_speakers=job.min_speakers, max_speakers=job.max_speakers)
        except Exception as exc:  # noqa: BLE001
            raise DiarizationError(
                "diarization model unavailable — check HF token and model-gate acceptance"
            ) from exc
        result = whisperx.assign_word_speakers(diar, result)
        del dp
        gc.collect()
        _empty_cache()
        return result


def _empty_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest server/tests/test_pipeline.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add server/pipeline.py server/tests/test_pipeline.py
git commit -m "feat(server): WhisperXPipeline wrapping the whisperx library

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: `server/worker.py` — single-consumer inference queue

**Files:**
- Create: `server/worker.py`
- Test: `server/tests/test_worker.py`

**Interfaces:**
- Consumes: `server.models.Pipeline`, `server.models.Job`; `server.errors.QueueFullError`, `NotReadyError`.
- Produces:
  - `class InferenceWorker`:
    - `__init__(self, pipeline: Pipeline, max_queue: int)`
    - `def start(self) -> None` — creates the consumer task on the running loop
    - `async def stop(self) -> None` — cancels the task; resolves any queued futures with `NotReadyError("server shutting down")`
    - `def submit(self, job: Job) -> asyncio.Future` — `put_nowait`; on `asyncio.QueueFull` raise `QueueFullError("inference queue is full", code="queue_full")`
    - `def depth(self) -> int` — `queue.qsize() + (1 if self._busy else 0)`
    - property `busy -> bool`

- [ ] **Step 1: Write the failing test**

`server/tests/test_worker.py`:

```python
import asyncio

import pytest

from server.errors import QueueFullError
from server.models import Job
from server.worker import InferenceWorker


def _job():
    return Job(audio_bytes=b"x", filename="a.wav", task="transcribe", language="en",
               initial_prompt=None, temperature=0.0, diarize=False, min_speakers=None,
               max_speakers=None, want_word_timestamps=False)


class FakePipeline:
    def __init__(self, delay=0.0, raises=None, result=None):
        self.delay, self.raises, self.result = delay, raises, result
        self.ran = 0

    def run(self, job):
        import time
        self.ran += 1
        time.sleep(self.delay)
        if self.raises:
            raise self.raises
        return self.result or {"task": job.task, "language": "en", "duration": 0.0,
                               "temperature": 0.0, "segments": [], "words": []}

    def close(self): ...


@pytest.mark.anyio
async def test_submit_resolves_future_with_result():
    w = InferenceWorker(FakePipeline(), max_queue=4)
    w.start()
    try:
        res = await w.submit(_job())
        assert res["task"] == "transcribe"
    finally:
        await w.stop()


@pytest.mark.anyio
async def test_exception_propagates_and_worker_survives():
    pipe = FakePipeline(raises=RuntimeError("boom"))
    w = InferenceWorker(pipe, max_queue=4)
    w.start()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await w.submit(_job())
        pipe.raises = None
        res = await w.submit(_job())          # next job still processed
        assert res["language"] == "en"
        assert pipe.ran == 2
    finally:
        await w.stop()


@pytest.mark.anyio
async def test_queue_full_raises():
    w = InferenceWorker(FakePipeline(delay=0.2), max_queue=1)
    w.start()
    try:
        f1 = asyncio.ensure_future(w.submit(_job()))  # taken by consumer
        await asyncio.sleep(0.05)
        f2 = asyncio.ensure_future(w.submit(_job()))  # fills queue (size 1)
        await asyncio.sleep(0.01)
        with pytest.raises(QueueFullError):
            await w.submit(_job())                    # over capacity
        await asyncio.gather(f1, f2)
    finally:
        await w.stop()


@pytest.mark.anyio
async def test_depth_reflects_busy_and_queued():
    w = InferenceWorker(FakePipeline(delay=0.2), max_queue=8)
    w.start()
    try:
        assert w.depth() == 0
        fut = asyncio.ensure_future(w.submit(_job()))
        await asyncio.sleep(0.05)
        assert w.depth() == 1                         # one in flight
        await fut
    finally:
        await w.stop()


@pytest.fixture
def anyio_backend():
    return "asyncio"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest server/tests/test_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.worker'`.

- [ ] **Step 3: Write minimal implementation**

`server/worker.py`:
```python
from __future__ import annotations

import asyncio

import anyio

from server.errors import NotReadyError, QueueFullError
from server.models import Job, Pipeline


class InferenceWorker:
    def __init__(self, pipeline: Pipeline, max_queue: int) -> None:
        self._pipeline = pipeline
        self._queue: asyncio.Queue[tuple[Job, asyncio.Future]] = asyncio.Queue(max_queue)
        self._task: asyncio.Task | None = None
        self._busy = False

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        while not self._queue.empty():
            _job, fut = self._queue.get_nowait()
            if not fut.done():
                fut.set_exception(NotReadyError("server shutting down"))

    @property
    def busy(self) -> bool:
        return self._busy

    def depth(self) -> int:
        return self._queue.qsize() + (1 if self._busy else 0)

    def submit(self, job: Job) -> asyncio.Future:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait((job, fut))
        except asyncio.QueueFull as exc:
            raise QueueFullError("inference queue is full", code="queue_full") from exc
        return fut

    async def _run(self) -> None:
        while True:
            job, fut = await self._queue.get()
            if job.abandoned or fut.cancelled():
                continue
            self._busy = True
            try:
                result = await anyio.to_thread.run_sync(self._pipeline.run, job)
                if not fut.done():
                    fut.set_result(result)
            except asyncio.CancelledError:
                if not fut.done():
                    fut.set_exception(NotReadyError("server shutting down"))
                raise
            except BaseException as exc:  # noqa: BLE001 — isolate the consumer loop
                if not fut.done():
                    fut.set_exception(exc)
                _empty_cache()
            finally:
                self._busy = False


def _empty_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
```

> Note: `submit` returns a `Future` (awaitable); tests `await w.submit(...)`. That is intentional — awaiting a Future yields its result/exception.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest server/tests/test_worker.py -v`
Expected: PASS (4 passed). If `anyio` marker is unknown, confirm `anyio` is installed (Task 1) — `pytest-anyio` ships with `anyio`.

- [ ] **Step 5: Commit**

```bash
git add server/worker.py server/tests/test_worker.py
git commit -m "feat(server): single-consumer inference worker with bounded queue

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: `server/auth.py` — optional bearer-token dependency

**Files:**
- Create: `server/auth.py`
- Test: `server/tests/test_auth.py`

**Interfaces:**
- Consumes: `server.config.Settings` (via `request.app.state.settings`); `server.errors.AuthError`.
- Produces:
  - `async def require_auth(request: Request) -> None` — if `settings.api_key` is falsy, return (no auth). Else require header `Authorization: Bearer <token>` with `secrets.compare_digest(token, settings.api_key)`; otherwise raise `AuthError("missing or invalid API key", code="invalid_api_key")`.

- [ ] **Step 1: Write the failing test**

`server/tests/test_auth.py`:

```python
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from server.auth import require_auth
from server.errors import register_exception_handlers


def _app(api_key):
    app = FastAPI()
    register_exception_handlers(app)
    app.state.settings = type("S", (), {"api_key": api_key})()

    @app.get("/x", dependencies=[Depends(require_auth)])
    def x():
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


def test_no_key_configured_allows_all():
    assert _app(None).get("/x").status_code == 200


def test_key_configured_rejects_missing_header():
    r = _app("secret").get("/x")
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_key_configured_rejects_wrong_token():
    r = _app("secret").get("/x", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_key_configured_accepts_correct_token():
    r = _app("secret").get("/x", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest server/tests/test_auth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.auth'`.

- [ ] **Step 3: Write minimal implementation**

`server/auth.py`:
```python
from __future__ import annotations

import secrets

from fastapi import Request

from server.errors import AuthError


async def require_auth(request: Request) -> None:
    settings = request.app.state.settings
    expected = getattr(settings, "api_key", None)
    if not expected:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token or not secrets.compare_digest(token, expected):
        raise AuthError("missing or invalid API key", code="invalid_api_key")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest server/tests/test_auth.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add server/auth.py server/tests/test_auth.py
git commit -m "feat(server): optional bearer-token auth dependency

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: `server/app.py` core — factory, lifespan, `/health`, `/v1/models`; conftest fixtures

**Files:**
- Create: `server/schemas.py`
- Create: `server/app.py`
- Create: `server/__main__.py`
- Modify: `server/tests/conftest.py` (add fixtures)
- Test: `server/tests/test_app_health_models.py`

**Interfaces:**
- Consumes: `server.config` (`Settings`, `load_settings`), `server.errors.register_exception_handlers`, `server.worker.InferenceWorker`, `server.pipeline.WhisperXPipeline`, `server.models.Pipeline`.
- Produces:
  - `server/schemas.py`:
    - `RESPONSE_FORMATS = ("json", "verbose_json", "text", "srt", "vtt")`
    - `class ModelObject(BaseModel)`: `id: str`, `object: str = "model"`, `created: int = 0`, `owned_by: str = "whisperx"`
    - `class ModelList(BaseModel)`: `object: str = "list"`, `data: list[ModelObject]`
  - `server/app.py`:
    - `def create_app(settings: Settings | None = None, pipeline_factory: Callable[[Settings], Pipeline] | None = None) -> FastAPI`
    - `app.state.settings`, `app.state.worker`, `app.state.ready: bool`
    - lifespan: build pipeline via `pipeline_factory` (default `WhisperXPipeline`) inside `anyio.to_thread.run_sync`; `worker = InferenceWorker(pipeline, settings.max_queue)`; `worker.start()`; `app.state.ready = True`. Shutdown: `await worker.stop()`; `pipeline.close()`.
    - `GET /health` → 200 `{"status":"ok","model":<str>,"device":<str>,"queue_depth":<int>,"diarization":<bool>}` once ready; else 503 `{"status":"starting"}`. `diarization` is `bool(settings.hf_token)`.
    - `GET /v1/models` → `ModelList` with `settings.whisperx_model` then `whisper-1`.
  - `server/__main__.py`: `uvicorn.run("server.app:create_app", factory=True, host=..., port=..., log_level=..., workers=1)` using `load_settings()`.
  - `conftest.py` fixtures: `anyio_backend` (→ `"asyncio"`), `wav_bytes`, `FakePipeline`, `make_client`.

- [ ] **Step 1: Add conftest fixtures**

Append to `server/tests/conftest.py`:

```python
import io
import struct
import wave

import pytest

from server.config import Settings
from server.models import Job, PipelineResult


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<" + "h" * 4800, *([0] * 4800)))  # 0.3 s silence
    return buf.getvalue()


class FakePipeline:
    """Implements server.models.Pipeline without touching torch/whisperx."""

    def __init__(self, *, delay: float = 0.0, raises: BaseException | None = None):
        self.delay = delay
        self.raises = raises
        self.jobs: list[Job] = []
        self.aligned = False

    def run(self, job: Job) -> PipelineResult:
        import time

        self.jobs.append(job)
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        self.aligned = job.task != "translate"
        seg = {
            "start": 0.0, "end": 1.0, "text": " hello world.",
            "avg_logprob": -0.1,
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world.", "start": 0.6, "end": 1.0},
            ],
        }
        if job.diarize:
            seg["speaker"] = "SPEAKER_00"
            for wd in seg["words"]:
                wd["speaker"] = "SPEAKER_00"
        return PipelineResult(
            task=job.task,
            language="en" if job.task == "translate" else (job.language or "en"),
            duration=0.3, temperature=job.temperature,
            segments=[seg], words=list(seg["words"]),
        )

    def close(self) -> None:  # noqa: D401
        pass


@pytest.fixture
def make_client():
    from fastapi.testclient import TestClient

    from server.app import create_app

    created: list = []

    def _make(pipeline: FakePipeline | None = None, **env) -> TestClient:
        pipe = pipeline or FakePipeline()
        settings = Settings(**env)
        app = create_app(settings=settings, pipeline_factory=lambda _s: pipe)
        client = TestClient(app, raise_server_exceptions=False)
        client._fake_pipeline = pipe  # for assertions
        created.append(client)
        return client

    yield _make
    for c in created:
        c.close()
```

- [ ] **Step 2: Write the failing test**

`server/tests/test_app_health_models.py`:

```python
def test_health_reports_ok_after_startup(make_client):
    with make_client(WHISPERX_MODEL="large-v2") as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"] == "large-v2"
        assert body["queue_depth"] == 0
        assert body["diarization"] is False


def test_health_reports_diarization_true_when_token_present(make_client):
    with make_client(HF_TOKEN="hf_x") as client:
        assert client.get("/health").json()["diarization"] is True


def test_models_lists_configured_model_and_whisper_1(make_client):
    with make_client(WHISPERX_MODEL="medium") as client:
        r = client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        assert ids == ["medium", "whisper-1"]
        assert body["data"][0]["owned_by"] == "whisperx"


def test_unhandled_route_still_uses_error_envelope(make_client):
    with make_client() as client:
        r = client.post("/v1/audio/transcriptions")  # not implemented until Task 8
        assert r.status_code in (404, 422, 401)
```

> `test_unhandled_route_still_uses_error_envelope` is a placeholder guard; it is replaced by real handler tests in Task 8. Keep it lenient here.

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest server/tests/test_app_health_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.app'`.

- [ ] **Step 4: Write minimal implementation**

`server/schemas.py`:
```python
from __future__ import annotations

from pydantic import BaseModel

RESPONSE_FORMATS = ("json", "verbose_json", "text", "srt", "vtt")


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "whisperx"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelObject]
```

`server/app.py` (core only — the transcription routes are added in Task 8):
```python
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

import anyio
from fastapi import FastAPI

from server.config import Settings, load_settings
from server.errors import register_exception_handlers
from server.models import Pipeline
from server.schemas import ModelList, ModelObject
from server.worker import InferenceWorker


def _default_pipeline_factory(settings: Settings) -> Pipeline:
    from server.pipeline import WhisperXPipeline

    return WhisperXPipeline(settings)


def create_app(
    settings: Settings | None = None,
    pipeline_factory: Callable[[Settings], Pipeline] | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    factory = pipeline_factory or _default_pipeline_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.ready = False
        pipeline = await anyio.to_thread.run_sync(factory, settings)
        worker = InferenceWorker(pipeline, settings.max_queue)
        worker.start()
        app.state.pipeline = pipeline
        app.state.worker = worker
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await worker.stop()
            pipeline.close()

    app = FastAPI(title="WhisperX API", lifespan=lifespan)
    app.state.settings = settings
    app.state.ready = False
    register_exception_handlers(app)

    @app.get("/health")
    async def health():
        if not app.state.ready:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=503, content={"status": "starting"})
        return {
            "status": "ok",
            "model": settings.whisperx_model,
            "device": settings.device,
            "queue_depth": app.state.worker.depth(),
            "diarization": bool(settings.hf_token),
        }

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        return ModelList(data=[
            ModelObject(id=settings.whisperx_model),
            ModelObject(id="whisper-1"),
        ])

    from server.routes_transcription import register_transcription_routes  # noqa: E402

    register_transcription_routes(app)  # no-op import until Task 8 creates it
    return app
```

> Task 8 creates `server/routes_transcription.py`. To keep Task 7 self-contained, for **this task** create a stub:
>
> `server/routes_transcription.py`:
> ```python
> def register_transcription_routes(app):  # replaced in Task 8
>     pass
> ```

`server/__main__.py`:
```python
from __future__ import annotations

import uvicorn

from server.config import load_settings

if __name__ == "__main__":
    s = load_settings()
    uvicorn.run(
        "server.app:create_app",
        factory=True,
        host=s.host,
        port=s.port,
        log_level=s.log_level,
        workers=1,
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest server/tests/test_app_health_models.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Run the whole server suite**

Run: `uv run pytest server/tests/ -v`
Expected: PASS (all tasks 1–7).

- [ ] **Step 7: Commit**

```bash
git add server/schemas.py server/app.py server/__main__.py server/routes_transcription.py server/tests/conftest.py server/tests/test_app_health_models.py
git commit -m "feat(server): app factory, lifespan, /health and /v1/models

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 8: transcription & translation endpoints

**Files:**
- Modify: `server/routes_transcription.py` (replace stub with real routes)
- Test: `server/tests/test_transcriptions.py`

**Interfaces:**
- Consumes: `server.auth.require_auth`, `server.schemas.RESPONSE_FORMATS`, `server.formats.render`, `server.models.Job`, `server.errors` (`BadRequestError`, `PayloadTooLargeError`, `RequestTimeoutError`, `NotReadyError`), `app.state.worker`, `app.state.settings`, `app.state.ready`.
- Produces:
  - `def register_transcription_routes(app: FastAPI) -> None` adding:
    - `POST /v1/audio/transcriptions` → `_handle(request, task="transcribe", ...)`
    - `POST /v1/audio/translations` → `_handle(request, task="translate", ...)`
  - Both use `Depends(require_auth)` and accept `multipart/form-data`: `file: UploadFile` (required), `model: str = Form("whisper-1")`, `language: str | None = Form(None)`, `prompt: str | None = Form(None)`, `temperature: float = Form(0.0)`, `response_format: str = Form("json")`, `timestamp_granularities: list[str] = Form(default_factory=list, alias="timestamp_granularities[]")`, `diarize: bool = Form(False)`, `min_speakers: int | None = Form(None)`, `max_speakers: int | None = Form(None)`.

**Handler `_handle` logic:**
1. If not `app.state.ready` → `NotReadyError("model is still loading")`.
2. `response_format` not in `RESPONSE_FORMATS` → `BadRequestError(param="response_format")`.
3. Read `file`. Enforce `len(data) <= settings.max_upload_mb * 1024 * 1024` else `PayloadTooLargeError(param="file")`. Empty → `BadRequestError("file is empty", param="file")`.
4. `diarize` and not `settings.hf_token` → `BadRequestError("diarize=true requires HF_TOKEN to be configured on the server", param="diarize", code="hf_token_missing")`.
5. Language reconciliation (transcribe only): let `configured = settings.language` (None if `auto`). If `task == "translate"`: `job_language = None`. Elif `language` given and `configured` is not None and `language.lower() != configured.lower()` → `BadRequestError("server is pinned to language '{configured}'; set WHISPERX_LANGUAGE=auto to allow per-request language", param="language")`. Else `job_language = (language or configured)` lowercased-or-None.
6. `want_words = "word" in timestamp_granularities`.
7. Build `Job(...)`. `submit`. `try: result = await asyncio.wait_for(fut, settings.request_timeout_s)` — `asyncio.TimeoutError` → set `job.abandoned = True`, raise `RequestTimeoutError("processing exceeded {n}s", code="timeout")`.
8. `body, media = render(result, response_format, want_words, diarize)`.
9. If `response_format in {"json","verbose_json"}` → `JSONResponse(body)`; else `Response(body, media_type=media)`.

- [ ] **Step 1: Write the failing test**

`server/tests/test_transcriptions.py`:

```python
import asyncio

import pytest

from server.errors import InferenceError

pytestmark = pytest.mark.usefixtures("make_client")


def _post(client, **form):
    files = {"file": ("a.wav", form.pop("_bytes"), "audio/wav")}
    return client.post("/v1/audio/transcriptions", files=files, data=form)


def test_json_default_returns_text_object(make_client, wav_bytes):
    with make_client() as c:
        r = _post(c, _bytes=wav_bytes)
        assert r.status_code == 200
        assert r.json() == {"text": "hello world."}


def test_text_format_plain_body(make_client, wav_bytes):
    with make_client() as c:
        r = _post(c, _bytes=wav_bytes, response_format="text")
        assert r.headers["content-type"].startswith("text/plain")
        assert r.text == "hello world."


def test_verbose_json_shapes_and_filler_fields(make_client, wav_bytes):
    with make_client() as c:
        r = _post(c, _bytes=wav_bytes, response_format="verbose_json")
        body = r.json()
        assert body["task"] == "transcribe"
        assert body["language"] == "en"
        seg = body["segments"][0]
        assert seg["tokens"] == [] and seg["no_speech_prob"] is None
        assert "words" not in body


def test_word_granularity_adds_words(make_client, wav_bytes):
    with make_client() as c:
        r = c.post("/v1/audio/transcriptions",
                   files={"file": ("a.wav", wav_bytes, "audio/wav")},
                   data=[("response_format", "verbose_json"),
                         ("timestamp_granularities[]", "word")])
        assert [w["word"] for w in r.json()["words"]] == ["hello", "world."]


def test_params_reach_the_job(make_client, wav_bytes):
    with make_client() as c:
        _post(c, _bytes=wav_bytes, prompt="Acme Corp", temperature="0.4")
        job = c._fake_pipeline.jobs[-1]
        assert job.initial_prompt == "Acme Corp"
        assert job.temperature == 0.4


def test_oversized_upload_rejected(make_client, wav_bytes):
    with make_client(MAX_UPLOAD_MB="0") as c:  # 0 MiB → anything non-empty is too large
        r = _post(c, _bytes=wav_bytes)
        assert r.status_code == 413
        assert r.json()["error"]["param"] == "file"


def test_empty_file_rejected(make_client):
    with make_client() as c:
        r = _post(c, _bytes=b"")
        assert r.status_code == 400


def test_unknown_response_format_rejected(make_client, wav_bytes):
    with make_client() as c:
        r = _post(c, _bytes=wav_bytes, response_format="mp3")
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "response_format"


def test_diarize_without_server_token_rejected(make_client, wav_bytes):
    with make_client() as c:  # no HF_TOKEN
        r = _post(c, _bytes=wav_bytes, diarize="true")
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "hf_token_missing"


def test_diarize_with_token_sets_speaker(make_client, wav_bytes):
    with make_client(HF_TOKEN="hf_x") as c:
        r = _post(c, _bytes=wav_bytes, diarize="true", response_format="verbose_json")
        assert r.json()["segments"][0]["speaker"] == "SPEAKER_00"


def test_language_conflict_with_pinned_server_rejected(make_client, wav_bytes):
    with make_client(WHISPERX_LANGUAGE="en") as c:
        r = _post(c, _bytes=wav_bytes, language="fr")
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "language"


def test_language_allowed_when_server_is_auto(make_client, wav_bytes):
    with make_client(WHISPERX_LANGUAGE="auto") as c:
        _post(c, _bytes=wav_bytes, language="fr")
        assert c._fake_pipeline.jobs[-1].language == "fr"


def test_translations_route_sets_translate_task_and_skips_align(make_client, wav_bytes):
    with make_client(WHISPERX_LANGUAGE="en") as c:
        r = c.post("/v1/audio/translations",
                   files={"file": ("a.wav", wav_bytes, "audio/wav")},
                   data={"response_format": "verbose_json"})
        assert r.status_code == 200
        assert r.json()["language"] == "en"
        job = c._fake_pipeline.jobs[-1]
        assert job.task == "translate"
        assert c._fake_pipeline.aligned is False


def test_auth_enforced_when_configured(make_client, wav_bytes):
    with make_client(API_KEY="k") as c:
        assert _post(c, _bytes=wav_bytes).status_code == 401
        r = c.post("/v1/audio/transcriptions",
                   files={"file": ("a.wav", wav_bytes, "audio/wav")},
                   headers={"Authorization": "Bearer k"})
        assert r.status_code == 200


def test_queue_full_returns_429(make_client, wav_bytes):
    from server.tests.conftest import FakePipeline
    slow = FakePipeline(delay=0.5)
    with make_client(slow, MAX_QUEUE="1") as c:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_post, c, _bytes=wav_bytes) for _ in range(4)]
            codes = sorted(f.result().status_code for f in futs)
        assert 429 in codes


def test_timeout_returns_504_and_worker_survives(make_client, wav_bytes):
    from server.tests.conftest import FakePipeline
    slow = FakePipeline(delay=0.4)
    with make_client(slow, REQUEST_TIMEOUT_S="0") as c:  # 0s ceiling
        r = _post(c, _bytes=wav_bytes)
        assert r.status_code == 504
    # a fresh client (fast pipeline) still works
    with make_client() as c2:
        assert _post(c2, _bytes=wav_bytes).status_code == 200


def test_pipeline_exception_returns_500_envelope(make_client, wav_bytes):
    from server.tests.conftest import FakePipeline
    boom = FakePipeline(raises=InferenceError("CUDA out of memory — try a smaller model"))
    with make_client(boom) as c:
        r = _post(c, _bytes=wav_bytes)
        assert r.status_code == 500
        assert r.json()["error"]["type"] == "api_error"
```

> `REQUEST_TIMEOUT_S=0`: implement `wait_for` so that a `0` ceiling means "fail if not already done" — `asyncio.wait_for(fut, 0)` raises `TimeoutError` when the result is not immediately ready, which is the intended behaviour here.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest server/tests/test_transcriptions.py -v`
Expected: FAIL — routes return 404 (stub `register_transcription_routes` does nothing).

- [ ] **Step 3: Write minimal implementation**

Replace `server/routes_transcription.py`:
```python
from __future__ import annotations

import asyncio

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from server.auth import require_auth
from server.errors import (
    BadRequestError, NotReadyError, PayloadTooLargeError, RequestTimeoutError,
)
from server.formats import render
from server.models import Job
from server.schemas import RESPONSE_FORMATS


def register_transcription_routes(app: FastAPI) -> None:
    async def _handle(
        request: Request,
        task: str,
        file: UploadFile,
        model: str,
        language: str | None,
        prompt: str | None,
        temperature: float,
        response_format: str,
        timestamp_granularities: list[str],
        diarize: bool,
        min_speakers: int | None,
        max_speakers: int | None,
    ):
        settings = request.app.state.settings
        if not request.app.state.ready:
            raise NotReadyError("model is still loading")

        if response_format not in RESPONSE_FORMATS:
            raise BadRequestError(
                f"unknown response_format '{response_format}'", param="response_format"
            )

        data = await file.read()
        limit = settings.max_upload_mb * 1024 * 1024
        if len(data) > limit:
            raise PayloadTooLargeError(
                f"file exceeds MAX_UPLOAD_MB ({settings.max_upload_mb} MiB)", param="file"
            )
        if not data:
            raise BadRequestError("file is empty", param="file")

        if diarize and not settings.hf_token:
            raise BadRequestError(
                "diarize=true requires HF_TOKEN to be configured on the server",
                param="diarize", code="hf_token_missing",
            )

        configured = settings.language  # None when WHISPERX_LANGUAGE=auto
        if task == "translate":
            job_language: str | None = None
        elif language and configured and language.lower() != configured.lower():
            raise BadRequestError(
                f"server is pinned to language '{configured}'; set WHISPERX_LANGUAGE=auto "
                "to allow a per-request language",
                param="language",
            )
        else:
            chosen = language or configured
            job_language = chosen.lower() if chosen else None

        want_words = "word" in timestamp_granularities
        job = Job(
            audio_bytes=data, filename=file.filename or "audio.wav", task=task,  # type: ignore[arg-type]
            language=job_language, initial_prompt=prompt, temperature=temperature,
            diarize=diarize, min_speakers=min_speakers, max_speakers=max_speakers,
            want_word_timestamps=want_words,
        )

        fut = request.app.state.worker.submit(job)
        try:
            result = await asyncio.wait_for(fut, timeout=settings.request_timeout_s)
        except asyncio.TimeoutError as exc:
            job.abandoned = True
            raise RequestTimeoutError(
                f"processing exceeded {settings.request_timeout_s}s", code="timeout"
            ) from exc

        body, media = render(result, response_format, want_words, diarize)
        if response_format in ("json", "verbose_json"):
            return JSONResponse(content=body)
        return Response(content=body, media_type=media)

    def _form_params(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        temperature: float = Form(0.0),
        response_format: str = Form("json"),
        timestamp_granularities: list[str] = Form(default=[], alias="timestamp_granularities[]"),
        diarize: bool = Form(False),
        min_speakers: int | None = Form(None),
        max_speakers: int | None = Form(None),
    ) -> dict:
        return dict(
            file=file, model=model, language=language, prompt=prompt,
            temperature=temperature, response_format=response_format,
            timestamp_granularities=timestamp_granularities, diarize=diarize,
            min_speakers=min_speakers, max_speakers=max_speakers,
        )

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(require_auth)])
    async def transcriptions(request: Request, params: dict = Depends(_form_params)):
        return await _handle(request, task="transcribe", **params)

    @app.post("/v1/audio/translations", dependencies=[Depends(require_auth)])
    async def translations(request: Request, params: dict = Depends(_form_params)):
        return await _handle(request, task="translate", **params)
```

Also remove the stale comment in `server/app.py` (`# no-op import until Task 8 creates it`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest server/tests/test_transcriptions.py -v`
Expected: PASS. If FastAPI rejects the `list[str]` `Form` alias `timestamp_granularities[]`, fall back to reading it from `await request.form()` directly (`request.form().getlist("timestamp_granularities[]")`) and drop that field from `_form_params`.

- [ ] **Step 5: Full server suite**

Run: `uv run pytest server/tests/ -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add server/routes_transcription.py server/app.py server/tests/test_transcriptions.py
git commit -m "feat(server): OpenAI-compatible transcription and translation endpoints

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: Docker image, entrypoint, compose

**Files:**
- Create: `Dockerfile` (repo root)
- Create: `server/entrypoint.sh`
- Create: `.dockerignore` (repo root)
- Create: `docker-compose.yml` (repo root)
- Test: `server/tests/test_packaging.py`

**Interfaces:**
- Consumes: everything in `server/`; `pyproject.toml` + `uv.lock` (read-only).
- Produces: a buildable image whose container serves `/health` on `:8000`, runs as `PUID:PGID`, caches models under `/config`.

- [ ] **Step 1: Write the failing test**

`server/tests/test_packaging.py`:

```python
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_entrypoint_is_posix_sh_valid():
    ep = ROOT / "server" / "entrypoint.sh"
    assert ep.exists()
    r = subprocess.run(["sh", "-n", str(ep)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_entrypoint_drops_privileges_and_sets_cache_dirs():
    text = (ROOT / "server" / "entrypoint.sh").read_text()
    assert "gosu" in text
    assert "PUID" in text and "PGID" in text
    assert "/config" in text


def test_dockerfile_uses_uv_not_pip():
    df = (ROOT / "Dockerfile").read_text()
    assert "astral-sh/uv" in df
    assert "uv sync --frozen" in df
    assert "uv pip install -r server/requirements.txt" in df
    assert "pip install " not in df.replace("uv pip install ", "")
    assert "nvidia/cuda:12.8" in df
    assert 'CMD ["uv", "run", "--no-sync", "python", "-m", "server"]' in df or \
           'CMD ["uv","run","--no-sync","python","-m","server"]' in df


def test_dockerignore_keeps_lock_and_pyproject():
    di = (ROOT / ".dockerignore").read_text().splitlines()
    assert "uv.lock" not in di and "pyproject.toml" not in di
    assert any(line.strip() in {".git", ".git/"} for line in di)


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_compose_config_is_valid():
    r = subprocess.run(["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "config"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "whisperx-api" in r.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest server/tests/test_packaging.py -v`
Expected: FAIL — files absent.

- [ ] **Step 3: Write the files**

`server/entrypoint.sh`:
```sh
#!/bin/sh
set -e

PUID=${PUID:-1000}
PGID=${PGID:-1000}

groupmod -o -g "$PGID" whisperx 2>/dev/null || groupadd -o -g "$PGID" whisperx
usermod -o -u "$PUID" -g "$PGID" whisperx 2>/dev/null || \
    useradd -o -u "$PUID" -g "$PGID" -M -d /config whisperx

mkdir -p /config/huggingface /config/torch
chown -R "$PUID:$PGID" /config

exec gosu whisperx "$@"
```

`Dockerfile`:
```dockerfile
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg gosu passwd ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 whisperx \
    && useradd -u 1000 -g 1000 -M -d /config whisperx

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    WHISPERX_MODEL_DIR=/config \
    HF_HOME=/config/huggingface \
    TORCH_HOME=/config/torch \
    PORT=8000

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r server/requirements.txt

RUN cp server/entrypoint.sh /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

VOLUME ["/config"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uv", "run", "--no-sync", "python", "-m", "server"]
```

`.dockerignore`:
```
.git
.venv
tests
server/tests
figures
docs
.zvec-grep
graft
.github
__pycache__
*.pyc
.ignore
.python-version
```

`docker-compose.yml`:
```yaml
services:
  whisperx-api:
    build: .
    image: whisperx-api:local
    ports:
      - "8000:8000"
    environment:
      WHISPERX_MODEL: large-v2
      WHISPERX_LANGUAGE: en
      HF_TOKEN: ${HF_TOKEN:-}
      API_KEY: ${API_KEY:-}
      PUID: 1000
      PGID: 1000
    volumes:
      - whisperx-config:/config
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: ["gpu"]
    restart: unless-stopped

volumes:
  whisperx-config:
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest server/tests/test_packaging.py -v`
Expected: PASS (the `docker compose config` test is skipped if Docker is absent).

- [ ] **Step 5: Manual build + smoke (documented; run where Docker + network are available)**

```bash
docker build -t whisperx-api:local .
docker run --rm -e WHISPERX_MODEL=tiny -e WHISPERX_LANGUAGE=en -p 8000:8000 \
  -v whisperx-config:/config whisperx-api:local &
# wait for first-run model download, then:
curl -s localhost:8000/health
curl -s -F file=@tests/data/sample.wav -F response_format=verbose_json \
  localhost:8000/v1/audio/transcriptions
```
Expected: `/health` → `{"status":"ok",...}`; transcription returns segments. Record the result in the PR description.

- [ ] **Step 6: Commit**

```bash
git add Dockerfile server/entrypoint.sh .dockerignore docker-compose.yml server/tests/test_packaging.py
git commit -m "feat(server): CUDA Docker image, uv build, PUID/PGID entrypoint, compose

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 10: README fork section

**Files:**
- Modify: `README.md`

**Interfaces:** none (documentation).

- [ ] **Step 1: Add the section**

Insert after the feature bullet list (the `- 🗣️ VAD preprocessing…` line) and before `**Whisper** is an ASR model…`, a new section:

```markdown
<h2 align="left" id="api-server">API server (this fork) 🛰️</h2>

This fork adds a Dockerised, **OpenAI-compatible transcription API** on top of the
WhisperX pipeline, with **optional `pyannote` speaker diarization exposed over the
API** — the main way this fork differs from upstream and from a plain
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) server
([linuxserver image](https://docs.linuxserver.io/images/docker-faster-whisper/)).

### Quick start

```bash
export HF_TOKEN=hf_...        # only needed for diarize=true
docker compose up -d          # GPU host: needs nvidia-container-toolkit
curl -s localhost:8000/health
curl -s localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word'
```

CPU-only: delete the `deploy.resources` block from `docker-compose.yml`.

### Endpoints

| Method & path | Purpose |
|---|---|
| `POST /v1/audio/transcriptions` | OpenAI-compatible transcription (`json`, `verbose_json`, `text`, `srt`, `vtt`) |
| `POST /v1/audio/translations` | X→English; alignment is skipped |
| `GET /v1/models` | lists the configured model and `whisper-1` |
| `GET /health` | readiness + queue depth (container `HEALTHCHECK`) |

Extension form fields (beyond OpenAI): `diarize` (bool), `min_speakers`,
`max_speakers`. When `diarize=true`, `verbose_json` segments and words gain a
`speaker` field. `diarize=true` requires `HF_TOKEN` set on the server and
acceptance of the
[pyannote model gate](https://huggingface.co/pyannote/speaker-diarization-community-1).

### Configuration

All via environment (see
[the design doc](docs/superpowers/specs/2026-09-10-whisperx-openai-api-server-design.md)
for the full table). Common ones: `WHISPERX_MODEL` (default `small`),
`WHISPERX_LANGUAGE` (default `en`; `auto` to detect per request),
`WHISPERX_MODEL_DIR` (default `/config`), `API_KEY` (optional bearer token),
`HF_TOKEN`, `PUID` / `PGID`, `MAX_QUEUE`, `REQUEST_TIMEOUT_S`.
```

- [ ] **Step 2: Verify links and commands**

Run: `grep -n "api-server\|/v1/audio\|WHISPERX_MODEL_DIR" README.md`
Expected: the new anchor and references present. Manually confirm the design-doc relative path resolves from repo root and the fenced blocks render.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe the OpenAI-compatible API server in the README

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 11: CI wiring + GPU smoke test

**Files:**
- Modify: `.github/workflows/tests.yml`
- Create: `server/tests/test_smoke.py`

**Interfaces:**
- Consumes: the full `server/` app; a real `whisperx` model download (only when `WHISPERX_SMOKE=1`).
- Produces: CI runs `server/tests/` on every push/PR; a `@pytest.mark.gpu` smoke test that is skipped by default.

- [ ] **Step 1: Write the smoke test (skipped by default)**

`server/tests/test_smoke.py`:
```python
import io
import os
import struct
import wave

import pytest

pytestmark = pytest.mark.gpu

SMOKE = os.environ.get("WHISPERX_SMOKE") == "1"


@pytest.mark.skipif(not SMOKE, reason="set WHISPERX_SMOKE=1 to run the real-model smoke test")
def test_real_tiny_transcription(tmp_path):
    from fastapi.testclient import TestClient

    from server.app import create_app
    from server.config import Settings

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

    app = create_app(Settings(whisperx_model="tiny", whisperx_model_dir=str(tmp_path),
                              whisperx_language="en"))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/v1/audio/transcriptions",
                        files={"file": ("s.wav", buf.getvalue(), "audio/wav")},
                        data={"response_format": "verbose_json"})
        assert r.status_code == 200
        assert "segments" in r.json()
```

- [ ] **Step 2: Verify it is collected but skipped**

Run: `uv run pytest server/tests/test_smoke.py -v`
Expected: `1 skipped` (reason: `WHISPERX_SMOKE=1` not set). No "unknown mark" warning (the `gpu` marker is registered in `conftest.py` from Task 1).

- [ ] **Step 3: Extend the CI workflow**

In `.github/workflows/tests.yml`, in the `Run tests` step, replace the run line so both suites execute:

```yaml
      - name: Run tests
        run: |
          uv sync --all-extras
          uv pip install -r server/requirements.txt
          uv run pytest tests/ server/tests/ -v
```

(Leave `python-compatibility.yml` and the other workflows unchanged.)

- [ ] **Step 4: Validate the workflow YAML**

Run: `uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/tests.yml')); print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Full local run**

Run: `uv run pytest tests/ server/tests/ -v`
Expected: PASS (core suite unchanged; all server tests pass; `test_smoke.py` skipped).

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/tests.yml server/tests/test_smoke.py
git commit -m "ci: run server test suite; add opt-in GPU smoke test

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-review

**1. Spec coverage**

| Spec section | Task(s) |
|---|---|
| Decisions 1 (API shape), API reference | Task 8 (routes), Task 3 (formats), Task 7 (`/v1/models`, `/health`) |
| Decision 2 (preload ASR, per-req align/diarize free) | Task 4 (`WhisperXPipeline`) |
| Decision 3 (GPU primary, CPU fallback) | Task 1 (`device`/`compute_type` auto), Task 9 (CUDA base image) |
| Decision 4 (optional bearer auth) | Task 6 (`auth.py`), Task 8 (wired + tested) |
| Decision 5 (one worker + FIFO queue, 429) | Task 5 (`InferenceWorker`), Task 8 (429 path) |
| Decision 6 (synchronous only) | Task 8 (`await wait_for`) |
| Decision 7 (standalone `server/`, own requirements) | Task 1 (`server/requirements.txt`, not in `pyproject.toml`) |
| Decision 8 (always align; translate never) | Task 4 (`run()` skips align for `translate`), Task 8 (`/v1/audio/translations`) |
| `WHISPERX_LANGUAGE=en` default + `auto` sentinel | Task 1 (`Settings.language`), Task 8 (reconciliation + tests) |
| `WHISPERX_MODEL_DIR=/config` + `HF_HOME`/`TORCH_HOME` | Task 1 (default), Task 9 (Dockerfile env + entrypoint) |
| `PUID`/`PGID` privilege drop | Task 9 (`entrypoint.sh`) |
| Config table (all env vars) | Task 1 (`Settings`) |
| Error handling table (401/400/413/429/504/502/500/503, envelope) | Task 2 (`errors.py`), Tasks 4/5/8 (raise sites) |
| Worker survives job errors; `empty_cache` after | Task 5 (`_run` isolation), Task 4 (OOM mapping) |
| Temp file always deleted | Task 4 (`run()` `finally`) |
| `verbose_json` filler fields (`seek`,`tokens`,`compression_ratio`,`no_speech_prob`) | Task 3 (`_segment`) |
| `srt`/`vtt` via `whisperx.utils` writers | Task 3 (`_write_subtitles`) |
| Docker image (uv, CUDA, healthcheck, volume) | Task 9 |
| `docker-compose.yml` | Task 9 |
| README fork section | Task 10 |
| Testing strategy (fake pipeline, no GPU) | Tasks 1–8 tests, Task 7 `conftest.py` |
| GPU smoke test, `gpu` marker | Task 1 (marker), Task 11 (`test_smoke.py`) |
| Startup config validation aborts container | Task 1 (`validate_runtime`), Task 7 (`load_settings` in lifespan/`__main__`) |

No gaps identified. CI wiring (Task 11) is an addition beyond the spec's "documented manual smoke test" note, kept minimal and confined to `tests.yml`.

**2. Placeholder scan**

- The two acknowledged test typos (`//` in Task 4, the lenient placeholder test in Task 7) are called out inline with removal/replacement instructions — not silent placeholders.
- No "TBD"/"handle edge cases"/"similar to Task N". Each code step carries full code.

**3. Type consistency**

- `Job` fields identical across Tasks 1, 4, 5, 8 (`audio_bytes, filename, task, language, initial_prompt, temperature, diarize, min_speakers, max_speakers, want_word_timestamps, abandoned`).
- `PipelineResult` keys identical across Tasks 1, 3, 4, 7 (`task, language, duration, temperature, segments, words`).
- `render(result, fmt, want_words, diarized) -> (body, media_type)` — same signature in Task 3 definition and Task 8 call site.
- `InferenceWorker(pipeline, max_queue)` / `.submit() -> Future` / `.depth()` / `.start()` / `async .stop()` — consistent Tasks 5, 7, 8.
- `create_app(settings=None, pipeline_factory=None)` — consistent Tasks 7, 8 (via `conftest.make_client`), 11.
- `APIError.status` / `.err_type` / `.message` / `.param` / `.code` — consistent Tasks 2, 4, 5, 6, 8.
- `require_auth` reads `request.app.state.settings.api_key` — set in Task 7 lifespan and `create_app`, exercised in Tasks 6, 8.
- `register_transcription_routes(app)` — stub in Task 7, real in Task 8, same name/signature.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-10-whisperx-openai-api-server.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
