import pytest

from server.errors import InferenceError

pytestmark = pytest.mark.usefixtures("make_client")


def _post(client, **form):
    files = {"file": ("a.wav", form.pop("_bytes"), "audio/wav")}
    return client.post("/v1/audio/transcriptions", files=files, data=form)


def test_json_default_returns_text_object(make_client, wav_bytes):
    with make_client() as c:
        r = _post(c, _bytes=wav_bytes)
        assert r.status_code == 200
        assert r.json() == {"text": "hello world."}


def test_text_format_plain_body(make_client, wav_bytes):
    with make_client() as c:
        r = _post(c, _bytes=wav_bytes, response_format="text")
        assert r.headers["content-type"].startswith("text/plain")
        assert r.text == "hello world."


def test_verbose_json_shapes_and_filler_fields(make_client, wav_bytes):
    with make_client() as c:
        r = _post(c, _bytes=wav_bytes, response_format="verbose_json")
        body = r.json()
        assert body["task"] == "transcribe"
        assert body["language"] == "en"
        seg = body["segments"][0]
        assert seg["tokens"] == [] and seg["no_speech_prob"] is None
        assert "words" not in body


def test_word_granularity_adds_words(make_client, wav_bytes):
    with make_client() as c:
        r = c.post("/v1/audio/transcriptions",
                   files={"file": ("a.wav", wav_bytes, "audio/wav")},
                   data={"response_format": "verbose_json",
                         "timestamp_granularities[]": "word"})
        assert [w["word"] for w in r.json()["words"]] == ["hello", "world."]


def test_bare_timestamp_granularities_key_adds_words(make_client, wav_bytes):
    with make_client() as c:
        r = c.post("/v1/audio/transcriptions",
                   files={"file": ("a.wav", wav_bytes, "audio/wav")},
                   data={"response_format": "verbose_json",
                         "timestamp_granularities": "word"})
        assert [w["word"] for w in r.json()["words"]] == ["hello", "world."]


def test_params_reach_the_job(make_client, wav_bytes):
    with make_client() as c:
        _post(c, _bytes=wav_bytes, prompt="Acme Corp", temperature="0.4")
        job = c._fake_pipeline.jobs[-1]
        assert job.initial_prompt == "Acme Corp"
        assert job.temperature == 0.4


def test_oversized_upload_rejected(make_client, wav_bytes):
    with make_client(MAX_UPLOAD_MB="0") as c:  # 0 MiB → anything non-empty is too large
        r = _post(c, _bytes=wav_bytes)
        assert r.status_code == 413
        assert r.json()["error"]["param"] == "file"


def test_empty_file_rejected(make_client):
    with make_client() as c:
        r = _post(c, _bytes=b"")
        assert r.status_code == 400


def test_unknown_response_format_rejected(make_client, wav_bytes):
    with make_client() as c:
        r = _post(c, _bytes=wav_bytes, response_format="mp3")
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "response_format"


def test_diarize_without_server_token_rejected(make_client, wav_bytes):
    with make_client() as c:  # no HF_TOKEN
        r = _post(c, _bytes=wav_bytes, diarize="true")
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "hf_token_missing"


def test_diarize_with_token_sets_speaker(make_client, wav_bytes):
    with make_client(HF_TOKEN="hf_x") as c:
        r = _post(c, _bytes=wav_bytes, diarize="true", response_format="verbose_json")
        assert r.json()["segments"][0]["speaker"] == "SPEAKER_00"


def test_language_conflict_with_pinned_server_rejected(make_client, wav_bytes):
    with make_client(WHISPERX_LANGUAGE="en") as c:
        r = _post(c, _bytes=wav_bytes, language="fr")
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "language"


def test_language_allowed_when_server_is_auto(make_client, wav_bytes):
    with make_client(WHISPERX_LANGUAGE="auto") as c:
        _post(c, _bytes=wav_bytes, language="fr")
        assert c._fake_pipeline.jobs[-1].language == "fr"


def test_translations_route_sets_translate_task_and_skips_align(make_client, wav_bytes):
    with make_client(WHISPERX_LANGUAGE="en") as c:
        r = c.post("/v1/audio/translations",
                   files={"file": ("a.wav", wav_bytes, "audio/wav")},
                   data={"response_format": "verbose_json"})
        assert r.status_code == 200
        assert r.json()["language"] == "en"
        job = c._fake_pipeline.jobs[-1]
        assert job.task == "translate"
        assert c._fake_pipeline.aligned is False


def test_translations_srt_is_non_empty_subtitle(make_client, wav_bytes):
    with make_client(WHISPERX_LANGUAGE="en") as c:
        r = c.post("/v1/audio/translations",
                   files={"file": ("a.wav", wav_bytes, "audio/wav")},
                   data={"response_format": "srt"})
        assert r.status_code == 200
        assert "hello world." in r.text
        assert "-->" in r.text


def test_auth_enforced_when_configured(make_client, wav_bytes):
    with make_client(API_KEY="k") as c:
        assert _post(c, _bytes=wav_bytes).status_code == 401
        r = c.post("/v1/audio/transcriptions",
                   files={"file": ("a.wav", wav_bytes, "audio/wav")},
                   headers={"Authorization": "Bearer k"})
        assert r.status_code == 200


def test_queue_full_returns_429(make_client, wav_bytes):
    from server.tests.conftest import FakePipeline
    slow = FakePipeline(delay=0.5)
    with make_client(slow, MAX_QUEUE="1") as c:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_post, c, _bytes=wav_bytes) for _ in range(4)]
            codes = sorted(f.result().status_code for f in futs)
        assert 429 in codes


def test_timeout_returns_504_and_worker_survives(make_client, wav_bytes):
    from server.tests.conftest import FakePipeline
    slow = FakePipeline(delay=0.4)
    with make_client(slow, REQUEST_TIMEOUT_S="0") as c:  # 0s ceiling
        r = _post(c, _bytes=wav_bytes)
        assert r.status_code == 504
    # a fresh client (fast pipeline) still works
    with make_client() as c2:
        assert _post(c2, _bytes=wav_bytes).status_code == 200


def test_pipeline_exception_returns_500_envelope(make_client, wav_bytes):
    from server.tests.conftest import FakePipeline
    boom = FakePipeline(raises=InferenceError("CUDA out of memory — try a smaller model"))
    with make_client(boom) as c:
        r = _post(c, _bytes=wav_bytes)
        assert r.status_code == 500
        assert r.json()["error"]["type"] == "api_error"


def test_not_ready_returns_503_envelope(make_client, wav_bytes):
    # No `with`: the lifespan never runs, so app.state.ready stays False.
    c = make_client()
    r = c.post("/v1/audio/transcriptions",
               files={"file": ("a.wav", wav_bytes, "audio/wav")},
               data={"response_format": "json"})
    assert r.status_code == 503
    body = r.json()
    assert list(body) == ["error"]
    assert body["error"]["type"] == "api_error"
    assert body["error"]["message"]
