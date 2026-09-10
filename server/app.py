from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

import anyio
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from server.config import Settings, load_settings
from server.errors import register_exception_handlers
from server.models import Pipeline
from server.routes_transcription import register_transcription_routes
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
        return ModelList(
            data=[
                ModelObject(id=settings.whisperx_model),
                ModelObject(id="whisper-1"),
            ]
        )

    register_transcription_routes(app)  # no-op until Task 8 fills in the body
    return app
