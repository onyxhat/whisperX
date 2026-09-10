# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg gosu passwd ca-certificates python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000 whisperx \
    && useradd -u 1000 -g 1000 -M -d /config whisperx

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    WHISPERX_MODEL_DIR=/config \
    HF_HOME=/config/huggingface \
    TORCH_HOME=/config/torch \
    HOME=/config \
    XDG_CACHE_HOME=/config/.cache \
    PORT=8000

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r server/requirements.txt

RUN cp server/entrypoint.sh /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

VOLUME ["/config"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD /app/.venv/bin/python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health').status==200 else 1)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/app/.venv/bin/python", "-m", "server"]
