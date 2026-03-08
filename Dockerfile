FROM ubuntu:24.04

# Ubuntu 24.04 (Noble) ships ffmpeg 6.1.1, matching the host.
# Debian Bookworm (python:3.12-slim default) ships ffmpeg 7.x which has a
# breaking change in its V4L2 demuxer: VIDIOC_G_INPUT failure (ENOTTY) is
# now treated as fatal. The Raysentek/InfiRay P2 camera does not support
# VIDIOC_G_INPUT; ffmpeg 6.x silently warns and continues, 7.x aborts.
#
# ffmpeg:         required by PyAV for V4L2 YUYV demux
# libgl1:         required by opencv-python-headless (libGL.so.1)
# libglib2.0-0:   required by opencv-python-headless (libgthread)
# python3.12:     from ubuntu noble default repos
# python3-pip:    for installing uv

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        python3.12 \
        python3.12-venv \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv: fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies in a separate layer so rebuilds after code changes are fast.
# opencv-python-headless: no X11/Qt dependencies -- correct for a headless server.
COPY requirements.txt .
RUN uv venv /app/.venv --python python3.12 \
    && uv pip install --python /app/.venv/bin/python --no-cache -r requirements.txt

ENV PATH="/app/.venv/bin:$PATH"

COPY thermal_server.py .

EXPOSE 7700

# THERMAL_DEVICE: camera device path inside the container.
#   Set to match whatever the host passes via --device (e.g. /dev/thermal_cam).
#   If unset, find_camera() will attempt sysfs scanning (may not work in containers).
# SAVE_DIR: directory for snapshots and recordings.
#   Mount a host directory here so files persist after container restart.
# THERMAL_PORT: port the server listens on inside the container (default 7700).
#   Change via docker-compose ports + this env var together if you need a different port.
ENV THERMAL_DEVICE=/dev/thermal_cam \
    SAVE_DIR=/data \
    THERMAL_PORT=7700

CMD ["/app/.venv/bin/python3", "thermal_server.py"]
