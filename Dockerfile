FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Basic deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Copy code
COPY . .

# RunPod serverless entrypoint
CMD ["python3", "handler.py"]
