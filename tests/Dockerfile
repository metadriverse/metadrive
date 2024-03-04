ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-bullseye

RUN apt update && \
    apt install -y --no-install-recommends xvfb ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,target=/metadrive,rw,source=. \
    --mount=type=cache,target=/root/.cache          \
    /metadrive/tests/setup.sh