from __future__ import annotations

import asyncio

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from server.auth import require_auth
from server.errors import (
    BadRequestError,
    NotReadyError,
    PayloadTooLargeError,
    RequestTimeoutError,
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
                param="diarize",
                code="hf_token_missing",
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
            audio_bytes=data,
            filename=file.filename or "audio.wav",
            task=task,  # type: ignore[arg-type]
            language=job_language,
            initial_prompt=prompt,
            temperature=temperature,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
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

    async def _form_params(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        temperature: float = Form(0.0),
        response_format: str = Form("json"),
        diarize: bool = Form(False),
        min_speakers: int | None = Form(None),
        max_speakers: int | None = Form(None),
    ) -> dict:
        form = await request.form()
        timestamp_granularities = list(form.getlist("timestamp_granularities[]"))
        return dict(
            file=file,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    @app.post("/v1/audio/transcriptions", dependencies=[Depends(require_auth)])
    async def transcriptions(request: Request, params: dict = Depends(_form_params)):
        return await _handle(request, task="transcribe", **params)

    @app.post("/v1/audio/translations", dependencies=[Depends(require_auth)])
    async def translations(request: Request, params: dict = Depends(_form_params)):
        return await _handle(request, task="translate", **params)
