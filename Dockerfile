FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy worker code
COPY . .

# IMPORTANT: start RunPod serverless worker, not main.py
CMD ["python3", "-u", "handler.py"]
