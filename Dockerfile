# syntax=docker/dockerfile:1
# Plain CUDA runtime (NOT the -cudnn variant): the PyTorch cu128 wheels bundle
# their own cuDNN 9, and a second system cuDNN segfaults in libcudnn_graph.so.9.
FROM nvidia/cuda:12.8.1-runtime-rockylinux9

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

# gosu: static, distro-agnostic binary used by entrypoint.sh for the privilege
# drop (Rocky packages no gosu / su-exec). Add `--checksum=sha256:...` to pin it.
ADD --chmod=755 https://github.com/tianon/gosu/releases/download/1.17/gosu-amd64 \
    /usr/local/bin/gosu

# ffmpeg-free (EPEL): the ffmpeg CLI whisperx.load_audio shells out to — covers
#   wav/mp3/aac/m4a/flac/opus. Exotic codecs would need full ffmpeg via RPM Fusion.
# python3.12: Rocky's default python3 is 3.9, below requires-python >=3.10.
# shadow-utils: useradd/groupadd/usermod for entrypoint.sh (usually already present).
RUN dnf install -y --setopt=install_weak_deps=False epel-release \
 && dnf install -y --setopt=install_weak_deps=False \
        ffmpeg-free python3.12 shadow-utils ca-certificates \
 && dnf clean all && rm -rf /var/cache/dnf

# The `whisperx` user/group is created at container start by entrypoint.sh
# (remapped to $PUID/$PGID); there is no build-time user creation.

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    WHISPERX_MODEL_DIR=/config \
    HF_HOME=/config/huggingface \
    TORCH_HOME=/config/torch \
    PORT=8000

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r server/requirements.lock

# HOME/XDG_CACHE_HOME are runtime concerns for the dropped-privilege `whisperx`
# user; setting them here (not above) keeps uv's build cache at /root/.cache/uv,
# matching the --mount targets so the cache actually hits.
ENV HOME=/config \
    XDG_CACHE_HOME=/config/.cache

RUN cp server/entrypoint.sh /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

VOLUME ["/config"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD /app/.venv/bin/python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health').status==200 else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/app/.venv/bin/python", "-m", "server"]
