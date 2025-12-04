FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg libsm6 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "handler.py"]

