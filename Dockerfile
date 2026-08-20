# syntax=docker/dockerfile:1.7

# =============================================================================
# builder — has uv, a compiler toolchain and the lockfile. None of it ships.
# =============================================================================
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # never reach for an interpreter at build time — the base image is the one we want
    UV_PYTHON_DOWNLOADS=never \
    # build the venv outside the source tree so the runtime stage copies one path
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src

# Dependencies resolve from the lockfile alone, so this layer is cached until
# pyproject.toml or uv.lock changes and editing application code reinstalls
# nothing. --locked fails loudly on a stale lockfile rather than silently
# resolving something other than what was tested.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Wheels ship with debug symbols the runtime never reads; uvloop and asyncpg
# alone carry several megabytes of them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends binutils \
 && find /opt/venv -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null || true

# =============================================================================
# runtime — base image, one venv, the app. No uv, no toolchain, no apt cache.
# =============================================================================
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # /code on the path so `app.main` resolves regardless of entrypoint
    PYTHONPATH=/code \
    # the venv is on PATH, so `uvicorn`/`celery`/`alembic` need no `uv run`
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv

# Non-root. Nothing in the container writes to disk, so the app owns nothing.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

WORKDIR /code

COPY --from=builder /opt/venv /opt/venv
COPY alembic.ini ./
COPY app ./app

USER appuser
EXPOSE 8000

# Python rather than curl, so the runtime image needs no extra package.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

# --loop uvloop rather than the default "auto": auto silently falls back to
# the stdlib selector loop if uvloop is missing, which is a performance
# regression nothing reports. Naming it makes that a startup error instead.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop"]

# =============================================================================
# dev — runtime plus uv and the test group, for running the suite in a container.
#   docker compose --profile test run --rm tests
# =============================================================================
FROM runtime AS dev

USER root
COPY --from=ghcr.io/astral-sh/uv:0.6.12 /uv /bin/uv
COPY pyproject.toml uv.lock .python-version ./
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    # /code is root-owned and appuser writes nothing, so pytest gets no cache dir
    PYTEST_ADDOPTS="-p no:cacheprovider"
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project
USER appuser
CMD ["pytest", "-q"]
