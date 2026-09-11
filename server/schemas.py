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
