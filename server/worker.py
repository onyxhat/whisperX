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
