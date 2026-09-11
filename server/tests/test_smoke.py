import io
import os
import struct
import wave

import pytest

pytestmark = pytest.mark.gpu

SMOKE = os.environ.get("WHISPERX_SMOKE") == "1"


@pytest.mark.skipif(not SMOKE, reason="set WHISPERX_SMOKE=1 to run the real-model smoke test")
def test_real_tiny_transcription(tmp_path):
    from fastapi.testclient import TestClient

    from server.app import create_app
    from server.config import Settings

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(struct.pack("<" + "h" * 16000, *([0] * 16000)))

    app = create_app(Settings(whisperx_model="tiny", whisperx_model_dir=str(tmp_path),
                              whisperx_language="en"))
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/v1/audio/transcriptions",
                        files={"file": ("s.wav", buf.getvalue(), "audio/wav")},
                        data={"response_format": "verbose_json"})
        assert r.status_code == 200
        assert "segments" in r.json()
