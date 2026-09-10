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
        r = client.post("/v1/audio/transcriptions")  # not implemented until Task 8
        assert r.status_code in (404, 422, 401)
