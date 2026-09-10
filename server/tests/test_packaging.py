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
    assert "uv pip install -r server/requirements.txt" in df
    assert "pip install " not in df.replace("uv pip install ", "")
    assert "nvidia/cuda:12.8" in df
    assert 'CMD ["uv", "run", "--no-sync", "python", "-m", "server"]' in df or \
           'CMD ["uv","run","--no-sync","python","-m","server"]' in df


def test_dockerfile_installs_python():
    df = (ROOT / "Dockerfile").read_text()
    apt_line = next(line for line in df.splitlines() if "ffmpeg" in line and "gosu" in line)
    assert "python3" in apt_line


def test_dockerignore_keeps_lock_and_pyproject():
    di = (ROOT / ".dockerignore").read_text().splitlines()
    assert "uv.lock" not in di and "pyproject.toml" not in di
    assert any(line.strip() in {".git", ".git/"} for line in di)


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")
def test_compose_config_is_valid():
    r = subprocess.run(["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "config"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "whisperx-api" in r.stdout
