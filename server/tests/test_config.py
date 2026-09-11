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
