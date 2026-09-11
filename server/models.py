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
