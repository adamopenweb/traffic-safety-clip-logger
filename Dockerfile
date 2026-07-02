# CPU / portable image — runs on the Ubuntu mini-PC (no CUDA assumed).
# Python 3.12 (not 3.14) because the CV wheels (opencv, numpy) ship for it and
# it satisfies the project's requires-python >=3.11.
FROM python:3.12-slim

# System deps: ffmpeg for recording/clip export, libgl/libglib for OpenCV,
# v4l-utils for the camera probe.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching, then the source.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .[analyze]

# Config and data are provided at runtime via volume mounts (see compose).
# No ENTRYPOINT: compose passes the full ["traffic-log", ...] command verbatim.
CMD ["traffic-log", "--help"]
