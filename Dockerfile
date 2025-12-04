FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git git-lfs libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Worker code
COPY . .

ENV PYTHONUNBUFFERED=1

# IMPORTANT: start RunPod worker (not FastAPI)
CMD ["python3", "handler.py"]
