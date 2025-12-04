FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

# IMPORTANT: Start serverless handler, NOT main.py
CMD ["python3", "handler.py"]
