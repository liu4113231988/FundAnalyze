# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.13-slim-bookworm
ARG UV_VERSION=0.12.17

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${PYTHON_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Cache dependencies independently of application and static-file changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python /usr/local/bin/python --locked --no-dev --no-install-project

COPY src/ ./src/
COPY compat/ ./compat/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python /usr/local/bin/python --locked --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data/output/data /data/output/figures /data/output/results /data/.cache/matplotlib \
    && chown -R app:app /data

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    FIL_HOME=/data \
    FIL_DATA_DIR=/data/output/data \
    FIL_HOST=0.0.0.0 \
    FIL_PORT=8877 \
    FIL_AUTO_SERVICES=1 \
    FIL_SYNC_ENABLED=1 \
    FIL_SYNC_TIME=18:30 \
    MPLCONFIGDIR=/data/.cache/matplotlib

USER app:app
VOLUME ["/data"]
EXPOSE 8877
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('FIL_PORT', '8877') + '/api/health', timeout=3).close()"]

# The API starts the existing singleton worker and registered-asset scheduler.
CMD ["market-research"]
