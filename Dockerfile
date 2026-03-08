FROM python:3.12-slim

# ffmpeg:         required by PyAV for V4L2 YUYV demux
# libgl1:         required by opencv-python-headless (libGL.so.1)
# libglib2.0-0:   required by opencv-python-headless (libgthread)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv: fast Python package installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies in a separate layer so rebuilds after code changes are fast.
# opencv-python-headless: no X11/Qt dependencies -- correct for a headless server.
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY thermal_server.py .

EXPOSE 7700

# THERMAL_DEVICE: camera device path inside the container.
#   Set to match whatever the host passes via --device (e.g. /dev/thermal_cam).
#   If unset, find_camera() will attempt sysfs scanning (may not work in containers).
# SAVE_DIR: directory for snapshots and recordings.
#   Mount a host directory here so files persist after container restart.
# PORT: port the server listens on inside the container (default 7700).
#   Change via docker-compose ports + this env var together if you need a different port.
ENV THERMAL_DEVICE=/dev/thermal_cam \
    SAVE_DIR=/data \
    PORT=7700

CMD ["python3", "thermal_server.py"]
