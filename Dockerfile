# Dockerfile - CUDA-enabled container with PyTorch and necessary libs
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install ffmpeg for video encoding
RUN apt-get update && apt-get install -y ffmpeg

# Copy application code
COPY handler.py model_loader.py ./

# Launch the handler on container start
CMD ["python", "handler.py"]
