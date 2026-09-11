import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_entrypoint_is_posix_sh_valid():
    ep = ROOT / "server" / "entrypoint.sh"
    assert ep.exists()
    r = subprocess.run(["sh", "-n", str(ep)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_entrypoint_drops_privileges_and_sets_cache_dirs():
    text = (ROOT / "server" / "entrypoint.sh").read_text()
    assert "gosu" in text
    assert "PUID" in text and "PGID" in text
    assert "/config" in text


def test_dockerfile_uses_uv_not_pip():
    df = (ROOT / "Dockerfile").read_text()
    assert "astral-sh/uv" in df
    assert "uv sync --frozen" in df
    assert "uv pip install -r server/requirements.lock" in df
    assert "pip install " not in df.replace("uv pip install ", "")
    assert "nvidia/cuda:12.8" in df
    assert 'CMD ["/app/.venv/bin/python", "-m", "server"]' in df or \
           'CMD ["/app/.venv/bin/python","-m","server"]' in df


def test_dockerfile_sets_writable_home_for_dropped_privileges():
    df = (ROOT / "Dockerfile").read_text()
    assert "HOME=/config" in df
    assert "XDG_CACHE_HOME=/config/.cache" in df


def test_entrypoint_creates_xdg_cache_dir():
    text = (ROOT / "server" / "entrypoint.sh").read_text()
    assert "/config/.cache" in text


def test_dockerfile_installs_python():
    df = (ROOT / "Dockerfile").read_text()
    # Rocky's default python3 is 3.9, below requires-python >=3.10 — the image
    # must install an explicit newer interpreter for uv to build the venv from.
    assert "python3.12" in df


def test_dockerignore_keeps_lock_and_pyproject():
    di = (ROOT / ".dockerignore").read_text().splitlines()
    assert "uv.lock" not in di and "pyproject.toml" not in di
    assert any(line.strip() in {".git", ".git/"} for line in di)


def test_dockerignore_excludes_heavy_dirs():
    di = [line.strip() for line in (ROOT / ".dockerignore").read_text().splitlines()]
    for entry in (".claude", ".superpowers", ".pytest_cache", "*.egg-info"):
        assert entry in di, entry


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_compose_config_is_valid():
    r = subprocess.run(["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "config"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "whisperx-api" in r.stdout
