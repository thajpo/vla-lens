# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PI05_BACKEND=rocm \
	    PI05_VENV=/opt/vla-lens/pi05 \
	    PI05_STRICT_DEVICE_CHECK=0 \
	    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
	    LIBERO_CONFIG_PATH=/root/.libero \
	    LIBERO_DATASETS_PATH=/root/.cache/libero/datasets \
	    VLA_LENS_CAPTURE_PYTHON=/opt/vla-lens/pi05/bin/python \
    VLA_LENS_CAPTURE_PYTHONPATH=/app/src \
    VLA_LENS_CAPTURE_DEVICE=cuda \
    VLA_LENS_CAPTURE_DTYPE=bfloat16 \
    PATH=/opt/vla-lens/pi05/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
	      build-essential \
	      cmake \
	      git \
	      libegl-dev \
	      libegl1 \
	      libgl-dev \
	      libgl1 \
	      libglib2.0-0 \
	      libosmesa6 \
	      libx11-dev \
	      libxext-dev \
	      libxext6 \
	      libxrender1 \
	      linux-libc-dev \
	      ninja-build \
	      pkg-config \
	    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

RUN --mount=type=cache,target=/root/.cache/uv scripts/setup_pi05_env.sh --backend rocm

ENTRYPOINT ["scripts/docker_pi05_entrypoint.sh"]
CMD ["vla-pi05-batch-capture", "--help"]
