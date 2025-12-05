FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

