# Use a base image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (ffmpeg is required for video saving)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- PRE-DOWNLOAD MODEL CACHE ---
# We run a small python script during build to download the model into the image.
# This ensures fast cold-starts.
RUN python -c "from diffusers import TextToVideoSDPipeline; \
    TextToVideoSDPipeline.from_pretrained('cerspense/zeroscope_v2_576w')"

# Copy the handler code
COPY handler.py .

# Start the RunPod handler
CMD [ "python", "-u", "handler.py" ]
