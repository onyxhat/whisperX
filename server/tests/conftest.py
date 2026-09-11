import io
import struct
import wave

import pytest

from server.config import Settings
from server.models import Job, PipelineResult


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: requires a real GPU and model downloads (set WHISPERX_SMOKE=1)"
    )


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
        }
        # A `translate` job skips alignment, so server.pipeline produces segments
        # with no word-level timing (top-level `words` is []). Mirror that here so
        # the suite exercises the words-less subtitle path.
        if job.task == "translate":
            words: list[dict] = []
        else:
            words = [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world.", "start": 0.6, "end": 1.0},
            ]
            seg["words"] = words
        if job.diarize:
            seg["speaker"] = "SPEAKER_00"
            for wd in words:
                wd["speaker"] = "SPEAKER_00"
        return PipelineResult(
            task=job.task,
            language="en" if job.task == "translate" else (job.language or "en"),
            duration=0.3, temperature=job.temperature,
            segments=[seg], words=list(words),
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
