def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: requires a real GPU and model downloads (set WHISPERX_SMOKE=1)"
    )
