FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
RUN apt-get update && apt-get install -y git ffmpeg
RUN pip install --no-cache-dir fastapi uvicorn
# Install ML libraries
RUN pip install --no-cache-dir diffusers transformers accelerate torch safetensors SwissArmyTransformer imageio-ffmpeg
# Copy app code and set entrypoint (example FastAPI)
WORKDIR /app
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
