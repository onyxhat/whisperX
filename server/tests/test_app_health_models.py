def test_health_reports_ok_after_startup(make_client):
    with make_client(WHISPERX_MODEL="large-v2") as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model"] == "large-v2"
        assert body["queue_depth"] == 0
        assert body["diarization"] is False


def test_health_reports_diarization_true_when_token_present(make_client):
    with make_client(HF_TOKEN="hf_x") as client:
        assert client.get("/health").json()["diarization"] is True


def test_models_lists_configured_model_and_whisper_1(make_client):
    with make_client(WHISPERX_MODEL="medium") as client:
        r = client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        assert ids == ["medium", "whisper-1"]
        assert body["data"][0]["owned_by"] == "whisperx"


def test_unhandled_route_still_uses_error_envelope(make_client):
    with make_client() as client:
        r = client.post("/v1/audio/transcriptions")  # missing required `file`
        assert r.status_code == 400
        body = r.json()
        assert list(body) == ["error"]
        assert body["error"]["message"]
        assert body["error"]["type"] == "invalid_request_error"


def test_missing_file_returns_envelope_with_param(make_client):
    with make_client() as client:
        r = client.post("/v1/audio/transcriptions")
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "file"


def test_bad_temperature_returns_400_envelope(make_client):
    import io
    with make_client() as client:
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", io.BytesIO(b"RIFF...."), "audio/wav")},
            data={"temperature": "abc"},
        )
        assert r.status_code == 400
        body = r.json()
        assert list(body) == ["error"]
        assert body["error"]["type"] == "invalid_request_error"


def test_404_uses_error_envelope(make_client):
    with make_client() as client:
        r = client.get("/no/such/route")
        assert r.status_code == 404
        body = r.json()
        assert list(body) == ["error"]
        assert body["error"]["message"]
        assert body["error"]["type"] == "invalid_request_error"


def test_models_requires_auth_when_api_key_set(make_client):
    with make_client(WHISPERX_MODEL="medium", API_KEY="k") as client:
        assert client.get("/v1/models").status_code == 401
        r = client.get("/v1/models", headers={"Authorization": "Bearer k"})
        assert r.status_code == 200
        assert [m["id"] for m in r.json()["data"]] == ["medium", "whisper-1"]


def test_health_stays_open_when_api_key_set(make_client):
    with make_client(API_KEY="k") as client:
        assert client.get("/health").status_code == 200


def test_create_app_validates_injected_settings():
    import pytest

    from server.app import create_app
    from server.config import Settings

    with pytest.raises(ValueError):
        create_app(Settings(whisperx_device="cpu", whisperx_compute_type="float16"))
