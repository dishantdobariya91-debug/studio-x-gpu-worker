# Dockerfile for Studio X ZeroScope RunPod worker

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy worker code
COPY . .

# Run the RunPod serverless worker
CMD ["python3", "-u", "handler.py"]
