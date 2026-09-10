# WhisperX OpenAI-Compatible API Server — Design

**Status:** Approved for planning
**Date:** 2026-09-10
**Author:** brainstorming session (Isaac Springer + Claude)

## Goal

Run WhisperX as a long-lived HTTP service in a Docker container on a private
network, exposing an OpenAI-compatible audio-transcription API, with optional
`pyannote.audio` speaker diarization surfaced through extension fields.

## Architecture

A standalone FastAPI application in a new top-level `server/` directory imports
`whisperx` as a library and calls the same public functions the CLI uses
(`load_model`, `load_audio`, `load_align_model`, `align`, `assign_word_speakers`).
The core `whisperx` package is **not modified**. A single background worker task
owns the GPU: HTTP handlers validate a request, enqueue a job, and await its
result synchronously. The ASR model is loaded once at startup and stays resident;
alignment and diarization models are loaded per job and freed afterward, matching
WhisperX's existing memory-flush pattern. The service ships as one CUDA-based
Docker image that also runs on CPU when no GPU is present.

## Tech stack

- Python 3.12 (repo supports `>=3.10,<3.14`), `uv` for all dependency management
- `whisperx` (this repo) + its deps: `faster-whisper`, `pyannote-audio`, `torch~=2.8` (cu128)
- `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic-settings`
- Base image `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`; `gosu` for privilege drop
- `pytest` + `fastapi.testclient` for tests (no GPU, no model downloads)

## Reference projects

- SYSTRAN/faster-whisper — the ASR backend WhisperX already wraps
- docs.linuxserver.io faster-whisper image — container conventions (`/config`, `PUID`/`PGID`)
- pyannote/pyannote-audio — diarization models (gated, needs a Hugging Face token)

**How this fork differs:** it exposes optional diarization through the API.

---

## Decisions (locked during brainstorming)

| # | Decision |
|---|---|
| 1 | **API shape:** OpenAI endpoints (`/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/models`) with extension form fields; speaker labels are non-standard fields inside `verbose_json`. |
| 2 | **Model lifecycle:** preload one ASR model, keep resident; load align + diarize models per request and free them after. |
| 3 | **Runtime target:** NVIDIA GPU primary (CUDA 12.8), automatic CPU fallback. |
| 4 | **Auth:** optional bearer token — enforced only when `API_KEY` env is set. |
| 5 | **Concurrency:** one uvicorn worker + one internal FIFO queue; queue full → `429`. |
| 6 | **Processing:** synchronous only — the client holds the connection until the transcript is ready. No job API. |
| 7 | **Code layout:** standalone top-level `server/` dir, own `requirements.txt`, not shipped to PyPI, not added to `pyproject.toml`. |
| 8 | **Alignment:** always run the full VAD→ASR→align pipeline (except `translate`, which WhisperX forbids aligning). |

### Refinements agreed after the decisions

- `WHISPERX_LANGUAGE` defaults to `en`; the sentinel `auto` enables per-request detection.
- `WHISPERX_MODEL_DIR` defaults to `/config`; the entrypoint also points `HF_HOME`
  and `TORCH_HOME` under `/config` so align + diarization weights share the volume.
- `PUID`/`PGID` (default `1000`/`1000`) set the UID/GID the server process runs as;
  the entrypoint chowns `/config` and drops privileges via `gosu`.
- Per-request `language` handling: if `WHISPERX_LANGUAGE` is a fixed code, a
  differing per-request `language` returns `400`; a matching one is accepted. If
  `WHISPERX_LANGUAGE=auto`, the per-request `language` is used when supplied, else
  the model detects. This avoids re-tokenizing a resident model mid-flight.

## Non-goals (YAGNI)

- No async/job-polling API, no websockets, no streaming responses.
- No per-request model selection; no multi-model LRU cache.
- No multi-worker / multi-GPU scaling.
- No changes to the `whisperx` package, CLI, or `pyproject.toml`.
- No `/v1/audio/speech` (TTS) or any non-transcription OpenAI route.
- No request persistence, result retention, or usage metering.
- No HTTPS termination (front with a reverse proxy if needed).

---

## Component design

All files live under `server/` unless noted. Every file has one responsibility;
`app.py` never imports torch, `pipeline.py` never imports fastapi, `formats.py` is
pure functions.

### `server/config.py`

`Settings(BaseSettings)` from `pydantic-settings`, read once at startup, stored on
`app.state.settings`.

| Env var | Default | Meaning |
|---|---|---|
| `WHISPERX_MODEL` | `small` | faster-whisper model name or local path |
| `WHISPERX_DEVICE` | `auto` | `auto` → cuda if available else cpu; or `cuda`/`cpu` |
| `WHISPERX_COMPUTE_TYPE` | `auto` | `auto` → `float16` on cuda / `int8` on cpu; or `float16`/`float32`/`int8` |
| `WHISPERX_BATCH_SIZE` | `8` | ASR batch size |
| `WHISPERX_LANGUAGE` | `en` | pinned language; `auto` = detect per request |
| `WHISPERX_VAD_METHOD` | `pyannote` | `pyannote` or `silero` |
| `WHISPERX_ALIGN_CACHE` | `0` | `1` keeps the last-used language's align model resident |
| `WHISPERX_MODEL_DIR` | `/config` | persistent cache root (also `HF_HOME`/`TORCH_HOME` via entrypoint) |
| `WHISPERX_DIARIZE_MODEL` | `pyannote/speaker-diarization-community-1` | diarization pipeline id |
| `HF_TOKEN` | *(unset)* | Hugging Face token; required for `diarize=true`, else that request → `400` |
| `API_KEY` | *(unset)* | if set, require `Authorization: Bearer <API_KEY>` |
| `MAX_QUEUE` | `16` | queued jobs before `429` (in-flight job excluded) |
| `MAX_UPLOAD_MB` | `200` | larger uploads → `413` |
| `REQUEST_TIMEOUT_S` | `1800` | ceiling on one job (queue wait + processing) → `504` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | uvicorn bind |
| `LOG_LEVEL` | `info` | app + uvicorn log level |
| `PUID` / `PGID` | `1000` / `1000` | UID/GID the server runs as (handled in entrypoint) |
| `WHISPERX_SMOKE` | *(unset)* | `1` enables the GPU-marked smoke test |

Startup validates combinations (e.g. `float16` + `cpu`, unknown compute type,
missing model dir) and exits non-zero with a one-line message on failure — no
half-up server.

**Resolved fields:** `device: str`, `compute_type: str`, `language: str | None`
(`None` when `auto`), `model_dir: str`.

### `server/pipeline.py`

```python
class Job:
    audio_bytes: bytes
    filename: str
    task: Literal["transcribe", "translate"]
    language: str | None          # already reconciled against settings
    initial_prompt: str | None
    temperature: float
    diarize: bool
    min_speakers: int | None
    max_speakers: int | None
    want_word_timestamps: bool
    abandoned: bool = False       # set by the handler on timeout

class PipelineResult(TypedDict):
    task: str
    language: str
    duration: float
    segments: list[dict]          # start,end,text,avg_logprob|None,words[],speaker?
    words: list[dict]             # word,start,end,speaker?  (flattened)

class WhisperXPipeline:
    def __init__(self, settings: Settings) -> None: ...
        # whisperx.load_model(settings.model, settings.device,
        #   compute_type=settings.compute_type, language=settings.language,
        #   vad_method=settings.vad_method, download_root=settings.model_dir,
        #   threads=4)
        # self.asr_model stays resident. self._align_cache: tuple[str, Any] | None

    def run(self, job: Job) -> PipelineResult: ...
        # 1. write job.audio_bytes to NamedTemporaryFile(suffix=Path(filename).suffix)
        # 2. audio = whisperx.load_audio(tmp_path)            # ffmpeg, 16 kHz mono
        #    duration = len(audio) / 16000
        # 3. result = self.asr_model.transcribe(audio,
        #        batch_size=settings.batch_size, task=job.task,
        #        language=job.language)
        # 4. if job.task != "translate":
        #        model_a, meta = self._get_align_model(result["language"])
        #        result = whisperx.align(result["segments"], model_a, meta, audio,
        #            settings.device, return_char_alignments=False)
        #        if not settings.align_cache: free model_a
        # 5. if job.diarize:
        #        dp = whisperx.diarize.DiarizationPipeline(
        #            model_name=settings.diarize_model, token=settings.hf_token,
        #            device=settings.device, cache_dir=settings.model_dir)
        #        diar = dp(audio, min_speakers=job.min_speakers,
        #                  max_speakers=job.max_speakers)
        #        result = whisperx.assign_word_speakers(diar, result)
        #        del dp; gc.collect(); torch.cuda.empty_cache()
        # 6. finally: os.unlink(tmp_path)
        # 7. build PipelineResult (flatten words from segments; carry `speaker`)

    def close(self) -> None: ...   # del models, empty_cache
```

`load_audio` / `align` raising `RuntimeError`/`ValueError` for an undecodable file
is caught by the worker and mapped (see Error handling). OOM
(`torch.cuda.OutOfMemoryError`) is caught specifically.

### `server/worker.py`

```python
class InferenceWorker:
    def __init__(self, pipeline: WhisperXPipeline, max_queue: int) -> None:
        self._queue: asyncio.Queue[tuple[Job, asyncio.Future]] = asyncio.Queue(max_queue)
        self._busy = False

    def submit(self, job: Job) -> asyncio.Future: ...
        # self._queue.put_nowait((job, fut))  -> asyncio.QueueFull propagates

    def depth(self) -> int:            # qsize() + (1 if self._busy else 0)

    async def _run(self) -> None: ...
        # while True:
        #   job, fut = await self._queue.get()
        #   if job.abandoned or fut.cancelled(): continue
        #   self._busy = True
        #   try:
        #     res = await anyio.to_thread.run_sync(self._pipeline.run, job)
        #     if not fut.done(): fut.set_result(res)
        #   except BaseException as e:
        #     if not fut.done(): fut.set_exception(e)
        #     gc.collect(); torch.cuda.empty_cache()
        #   finally:
        #     self._busy = False
```

The loop never exits on a job exception; the consumer survives for the next job.
Started/stopped by the FastAPI lifespan.

### `server/auth.py`

`require_auth` FastAPI dependency: if `settings.api_key` is set, compare the
`Authorization: Bearer` header with `secrets.compare_digest`; mismatch/absent →
`AuthError` (→ 401). No-op when `api_key` is unset.

### `server/schemas.py`

Pydantic models: `TranscriptionForm` (parsed from `Form`/`File`), `ModelObject`,
`ModelList`, `VerboseJSONResponse`, `ErrorEnvelope`. `response_format` is an enum
`{json, verbose_json, text, srt, vtt}`; `timestamp_granularities[]` an optional
list of `{word, segment}`.

### `server/formats.py`

Pure functions, `PipelineResult -> str | dict`:

- `to_text(result) -> str` — concatenated segment text.
- `to_json(result) -> dict` — `{"text": ...}`.
- `to_verbose_json(result, want_words, diarized) -> dict` — OpenAI keys `task`,
  `language`, `duration`, `text`, `segments[]`, `words[]` (only when `want_words`).
  Per-segment: `id`, `seek=0`, `start`, `end`, `text`, `tokens=[]`, `temperature`,
  `avg_logprob` (from WhisperX or `null`), `compression_ratio=null`,
  `no_speech_prob=null`. Adds `"speaker"` to each segment and word **only when
  `diarized`**.
- `to_srt(result) -> str` / `to_vtt(result) -> str` — reuse
  `whisperx.utils.WriteSRT` / `WriteVTT` by calling their `write_result` with a
  `io.StringIO` in place of a file handle and a minimal options dict
  (`highlight_words=False`, `max_line_width=None`, `max_line_count=None`).
- `render(result, fmt, want_words, diarized) -> tuple[str|dict, media_type]` — dispatch.

### `server/errors.py`

Exception classes (`AuthError`, `BadRequestError`, `PayloadTooLargeError`,
`AudioDecodeError`, `QueueFullError`, `RequestTimeoutError`, `DiarizationError`,
`NotReadyError`, `InferenceError`) each carrying `status`, `type`, `param`, `code`.
One handler registered on the app renders `ErrorEnvelope`:
`{"error": {"message", "type", "param", "code"}}` and logs (4xx at warning, 5xx at
error with traceback).

### `server/app.py`

- `create_app(settings: Settings | None = None) -> FastAPI` — factory so tests can
  inject settings.
- **Lifespan:** resolve settings → construct `WhisperXPipeline` (blocking, in a
  thread) → construct + start `InferenceWorker` → set `app.state.ready = True`.
  Shutdown: stop worker (pending futures → `NotReadyError`/`503`), `pipeline.close()`.
- **Routes:**
  - `POST /v1/audio/transcriptions` → `_handle(task="transcribe")`
  - `POST /v1/audio/translations` → `_handle(task="translate")`
  - `GET /v1/models` → `ModelList` with `settings.model` + `whisper-1`
  - `GET /health` → `{status, model, device, queue_depth, diarization}`; `503`
    with `{status: "starting"}` until `app.state.ready`
- `_handle`: `require_auth` → validate form (size, format, diarize/token,
  language reconciliation) → build `Job` → `worker.submit` (`QueueFull` → 429) →
  `await asyncio.wait_for(fut, settings.request_timeout_s)` (`TimeoutError` → mark
  `job.abandoned`, 504) → `formats.render` → `Response`/`JSONResponse`.

### `server/__main__.py`

`uvicorn.run("server.app:create_app", factory=True, host=settings.host,
port=settings.port, log_level=settings.log_level, workers=1)`.

### `server/requirements.txt`

```
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
pydantic-settings>=2.4
anyio>=4
```

### `server/entrypoint.sh`

```sh
#!/bin/sh
set -e
PUID=${PUID:-1000}; PGID=${PGID:-1000}
groupmod -o -g "$PGID" whisperx 2>/dev/null || groupadd -o -g "$PGID" whisperx
usermod  -o -u "$PUID" -g "$PGID" whisperx 2>/dev/null || useradd -o -u "$PUID" -g "$PGID" -M -d /config whisperx
mkdir -p /config/huggingface /config/torch
chown -R "$PUID:$PGID" /config
exec gosu whisperx "$@"
```

---

## API reference

### `POST /v1/audio/transcriptions` · `POST /v1/audio/translations`

`multipart/form-data`.

| Field | Type | Notes |
|---|---|---|
| `file` | file (**required**) | any ffmpeg-decodable container; saved to temp, deleted after |
| `model` | string | accepted, not enforced; maps to the resident model |
| `language` | string | ISO code; reconciled against `WHISPERX_LANGUAGE` (see refinements). Ignored for `translations` |
| `prompt` | string | → `initial_prompt` |
| `temperature` | float | default `0` |
| `response_format` | enum | `json` (default) · `verbose_json` · `text` · `srt` · `vtt` |
| `timestamp_granularities[]` | list | `word` populates `words[]` in `verbose_json`; alignment always runs regardless |
| `diarize` | bool | **extension**; default `false`; `true` without `HF_TOKEN` → `400` |
| `min_speakers` / `max_speakers` | int | **extension**; only used when `diarize=true` |

**Responses**

- `text` → `text/plain`, transcript only
- `json` → `{"text": "..."}`
- `verbose_json` → OpenAI object (`task`, `language`, `duration`, `text`,
  `segments[]`, `words[]` when word granularity). Extension: `speaker` on segments
  and words when `diarize=true`. Absent-in-WhisperX fields are `0`/`[]`/`null`.
- `srt` / `vtt` → `text/plain` via WhisperX writers
- `translations` always reports `language: "en"` and never aligns (WhisperX
  forbids aligning `translate`)

### `GET /v1/models`

```json
{"object":"list","data":[
  {"id":"large-v2","object":"model","created":0,"owned_by":"whisperx"},
  {"id":"whisper-1","object":"model","created":0,"owned_by":"whisperx"}]}
```

### `GET /health`

```json
{"status":"ok","model":"large-v2","device":"cuda","queue_depth":0,"diarization":true}
```
`200` once the ASR model has loaded; `503 {"status":"starting"}` before that.

### Example

```bash
curl -s http://whisperx.lan:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@meeting.m4a \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word' \
  -F diarize=true -F min_speakers=2 -F max_speakers=4
```

---

## Execution & concurrency model

1. Handler validates, builds `Job`, calls `worker.submit(job)` → `asyncio.Future`.
   `asyncio.QueueFull` → `429`.
2. Handler `await asyncio.wait_for(fut, REQUEST_TIMEOUT_S)`. Timeout → `job.abandoned = True`, `504`.
3. Worker consumes one job at a time (FIFO), runs `pipeline.run` in a thread via
   `anyio.to_thread.run_sync` so the event loop keeps serving `/health`.
4. `queue_depth` = `qsize() + (1 if worker busy)`.
5. On any inference exception the worker resolves the future with that exception,
   runs `gc.collect()` + `torch.cuda.empty_cache()`, and continues.
6. Shutdown cancels the worker; queued futures resolve to `503`; models freed.

Per-job model flow (`pipeline.run`): temp file → `load_audio` → resident
`asr_model.transcribe` → `load_align_model` + `align` + free (unless
`WHISPERX_ALIGN_CACHE=1`) → optional `DiarizationPipeline` + `assign_word_speakers`
+ free → delete temp file in `finally`.

---

## Error handling

| Condition | Status | `type` | Origin |
|---|---|---|---|
| `API_KEY` set, header missing/wrong | 401 | `invalid_request_error` | `auth.py` |
| Missing `file` / bad `response_format` / bad `temperature` / `diarize` w/o token / conflicting `language` | 400 | `invalid_request_error` | handler validation (`param` set) |
| Upload > `MAX_UPLOAD_MB` | 413 | `invalid_request_error` | size check before temp write |
| ffmpeg cannot decode audio | 400 | `invalid_request_error` | `AudioDecodeError` from `pipeline.run` |
| Queue at `MAX_QUEUE` | 429 | `rate_limit_error` | `QueueFull` |
| Job exceeds `REQUEST_TIMEOUT_S` | 504 | `timeout_error` | `asyncio.wait_for`; job abandoned |
| Diarization model download/auth fails | 502 | `api_error` | `DiarizationError` (message names the HF gate) |
| CUDA OOM | 500 | `api_error` | caught specifically; suggests smaller model/batch; cache emptied |
| Any other inference error | 500 | `api_error` | worker → `future.set_exception`; traceback logged, generic message returned |
| Request before ASR model ready | 503 | `api_error` | `NotReadyError` |
| Invalid startup config | *(exit ≠ 0)* | — | `config.py` validation |

Cross-cutting: temp files removed in `finally` even on abandonment; the worker
loop never dies on a job error; startup failures abort the container.

---

## Docker

### `Dockerfile` (repo root)

- Base `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`.
- `COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/` (matches CI's uv version).
- `apt-get install -y --no-install-recommends ffmpeg gosu passwd ca-certificates`,
  lists cleaned in the same layer. `passwd` provides `usermod`/`groupadd`.
- Build-time: `groupadd -g 1000 whisperx && useradd -u 1000 -g 1000 -M -d /config whisperx`.
- Workdir `/app`. Layered install, each `RUN` with `--mount=type=cache,target=/root/.cache/uv`:
  1. `COPY pyproject.toml uv.lock README.md ./` → `uv sync --frozen --no-dev --no-install-project`
  2. `COPY . .`
  3. `uv sync --frozen --no-dev`
  4. `uv pip install -r server/requirements.txt`
- `COPY server/entrypoint.sh /usr/local/bin/entrypoint.sh` (+x).
- Env: `UV_COMPILE_BYTECODE=1`, `UV_LINK_MODE=copy`, `UV_PYTHON_DOWNLOADS=never`,
  `WHISPERX_MODEL_DIR=/config`, `HF_HOME=/config/huggingface`,
  `TORCH_HOME=/config/torch`, `PYTHONUNBUFFERED=1`, `PORT=8000`.
- `VOLUME ["/config"]`, `EXPOSE 8000`.
- `HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 CMD`
  `/app/.venv/bin/python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"`.
- `ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]`, `CMD ["uv","run","--no-sync","python","-m","server"]`.

### `docker-compose.yml` (repo root)

```yaml
services:
  whisperx-api:
    build: .
    image: whisperx-api:local
    ports: ["8000:8000"]
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
          devices: [{ capabilities: ["gpu"] }]
    restart: unless-stopped
volumes:
  whisperx-config:
```

### `.dockerignore` (repo root)

Keep `pyproject.toml`, `uv.lock`, `README.md`, `whisperx/`, `server/` (minus
tests). Exclude `.git`, `.venv`, `tests/`, `server/tests/`, `figures/`, `docs/`,
`.zvec-grep/`, `graft/`, `__pycache__`, `*.pyc`, `.ignore`.

---

## README update

Edit the existing `README.md` (keep all upstream content):

- Add a short **"WhisperX API server (this fork)"** section after the feature
  bullets / before **Setup**, stating: this fork adds a Dockerized,
  OpenAI-compatible transcription API with **optional `pyannote` diarization
  exposed over the API**, referencing faster-whisper and the linuxserver image
  for lineage.
- Quick start:
  ```bash
  export HF_TOKEN=hf_...      # only needed for --diarize / diarize=true
  docker compose up -d
  curl -s localhost:8000/v1/audio/transcriptions -F file=@audio.wav -F response_format=verbose_json
  ```
- Endpoint table (`/v1/audio/transcriptions`, `/v1/audio/translations`,
  `/v1/models`, `/health`) and the extension fields (`diarize`, `min_speakers`,
  `max_speakers`).
- Link to this design doc and to the env-var table (or inline the table).
- Note: GPU needs `nvidia-container-toolkit`; CPU-only works by omitting the
  `deploy.resources` block.

---

## Testing strategy

`server/tests/`, `pytest` + `fastapi.testclient.TestClient`. **No GPU, no model
downloads.** `conftest.py` overrides the pipeline with a fake.

**Fixtures**

- `fake_pipeline` — a `WhisperXPipeline` stand-in whose `run(job)` returns a canned
  `PipelineResult` derived from the job (segments, words, `speaker` when
  `job.diarize`), with configurable delay and optional raise; records whether
  alignment "ran" (for the translate test).
- `client(**env)` — builds the app via `create_app(Settings(**env))` with the fake
  pipeline injected; parametrises `API_KEY`, `HF_TOKEN`, `MAX_QUEUE`,
  `REQUEST_TIMEOUT_S`, `WHISPERX_LANGUAGE`.
- `wav_bytes` — a few hundred ms of silence (numpy → WAV) for real multipart bytes.

**Coverage**

| File | Tests |
|---|---|
| `test_transcriptions.py` | each `response_format` → correct content-type & shape; `verbose_json` has OpenAI keys + filler `null`/`[]`; `timestamp_granularities[]=word` → `words[]` present, absent otherwise; `language`/`prompt`/`temperature` reach the `Job`; `> MAX_UPLOAD_MB` → 413; undecodable bytes → 400 |
| `test_translations.py` | `task=translate` on the job; response `language == "en"`; fake pipeline flag shows alignment was **not** invoked |
| `test_auth.py` | no `API_KEY` → open; `API_KEY` set → 401 without / with wrong bearer, 200 with correct |
| `test_models.py` | `/v1/models` lists configured id + `whisper-1`; OpenAI shape |
| `test_queue.py` | fake pipeline blocks; `MAX_QUEUE=1` → 3rd concurrent request → 429; `/health` `queue_depth` tracks state; short `REQUEST_TIMEOUT_S` → 504 and the next request still succeeds (worker survived) |
| `test_formats.py` | pure `formats.py`: result → srt/vtt golden strings; `speaker` present only when diarized; `words[]` flattened from segment `words` |
| `test_errors.py` | fake pipeline raises → 500 envelope + worker alive; `diarize=true` without `HF_TOKEN` → 400; conflicting `language` vs fixed `WHISPERX_LANGUAGE` → 400; bad compute/device combo → raises at `create_app` |
| `test_health.py` | `503 {"status":"starting"}` before lifespan ready; `200` after; fields present |

**Marked, skipped by default:** `test_smoke.py` — `@pytest.mark.gpu`, runs only
when `WHISPERX_SMOKE=1`; does one real `tiny` transcription end to end.

---

## Risks & open questions

- **First-run latency:** `large-v2` + wav2vec2 + pyannote downloads can take
  minutes on first request; `HEALTHCHECK --start-period=5m` and docs cover it, but
  a very slow link may need a longer start period.
- **Resident model + per-request `language`:** the reconciliation rule (400 on
  conflict) is the simple choice; if a real client library always sends
  `language`, operators must set `WHISPERX_LANGUAGE=auto` or match it. Documented.
- **`avg_logprob` fidelity:** WhisperX's batched path may not populate
  `avg_logprob` on every segment; `verbose_json` emits `null` when missing. Some
  strict OpenAI consumers may dislike `null` — acceptable for this fork.
- **`torchcodec`/pyannote 4 on the CUDA base:** `pyproject.toml` already scopes
  `torchcodec` to linux/x86_64; the image is x86_64, so `uv sync --frozen`
  resolves it. ARM hosts are out of scope.
- **CPU fallback speed:** functional but slow; not load-tested here.
