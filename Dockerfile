# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies (e.g., ffmpeg)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy code
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port for FastAPI (if using main.py)
EXPOSE 8000

# Command to run FastAPI server; for RunPod worker you might use handler mode instead
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
