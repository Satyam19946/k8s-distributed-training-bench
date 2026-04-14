# Layer 1 — base OS + Python runtime
# python:3.11-slim is Debian bookworm with Python 3.11, stripped of docs/tests.
# ~130MB vs ~900MB for the full image. We don't need the extras.
FROM python:3.11-slim

# Layer 2 — system dependencies
# gcc is needed to compile some torch C extensions at install time.
# No apt cache is kept (/var/cache/apt) — reduces image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Layer 3 — Python dependencies
# Copy requirements first, before the script.
# If only train.py changes on rebuild, this layer is served from cache —
# Docker doesn't reinstall torch just because your script changed.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Layer 4 — application code
# This layer changes most frequently — keep it last.
COPY train.py /app/train.py

# Set working directory
WORKDIR /app

# What runs when the container starts
ENTRYPOINT ["python", "train.py"]
