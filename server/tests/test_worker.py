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
