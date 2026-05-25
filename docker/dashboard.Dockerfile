# syntax=docker/dockerfile:1

FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS python-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VLA_LENS_FRONTEND_DIST=/app/frontend/dist \
    VLA_LENS_TRACE_ROOT=/data/vla-lens/runs/vla_lens_demo \
    VLA_LENS_HOST=0.0.0.0 \
    VLA_LENS_PORT=8080 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

COPY --from=python-builder /app/.venv ./.venv
COPY --from=python-builder /app/src ./src
COPY --from=python-builder /app/scripts ./scripts
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

RUN chmod +x scripts/docker_dashboard_entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["scripts/docker_dashboard_entrypoint.sh"]
