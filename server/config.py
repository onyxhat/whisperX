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
                f"compute_type '{self.whisperx_compute_type}' invalid; "
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
