import runpod
import torch
import base64
import os
import io
import logging
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model ID - using the 576w version which is the base Text-to-Video model
# (The XL model is typically an upscaler, using 576w ensures proper generation from text)
MODEL_ID = "cerspense/zeroscope_v2_576w"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = None

def init_model():
    """Load the model into memory only once when the worker starts."""
    global pipe
    if pipe is None:
        logger.info(f"[Studio X] Loading ZeroScope v2 model: {MODEL_ID}...")
        try:
            pipe = TextToVideoSDPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16
            )
            pipe.to(device)
            # Optimizations for speed and memory
            pipe.enable_model_cpu_offload() 
            pipe.enable_vae_slicing()
            logger.info("[Studio X] Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

def handler(job):
    """
    RunPod Handler Function
    """
    global pipe
    job_input = job.get("input", {})
    
    # 1. Extract inputs
    prompt = job_input.get("prompt", "A futuristic cyberpunk city with neon lights")
    num_frames = job_input.get("num_frames", 24) # Default 24 frames (approx 3-4 seconds depending on fps)
    fps = job_input.get("fps", 8)
    
    logger.info(f"[Studio X] Generating video for prompt: {prompt}")

    # 2. Generate Video
    # Ensure model is initialized
    if pipe is None:
        init_model()
        
    try:
        video_frames = pipe(
            prompt, 
            num_frames=num_frames, 
            height=320, 
            width=576
        ).frames[0]
        
        logger.info("[Studio X] Video frames generated.")

        # 3. Export to video file (temporary path)
        output_path = "/tmp/output_video.mp4"
        export_to_video(video_frames, output_path, fps=fps)
        logger.info(f"[Studio X] Video saved to: {output_path}")

        # 4. Convert to Base64 for immediate return
        # (In production, you would upload 'output_path' to S3 here and return the URL)
        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()
            base64_video = base64.b64encode(video_bytes).decode("utf-8")
        
        # Create a Data URI
        video_url = f"data:video/mp4;base64,{base64_video}"

        return {
            "video_url": video_url,
            "status": "completed",
            "message": "Video generated successfully"
        }

    except Exception as e:
        logger.error(f"Error during generation: {e}")
        return {"error": str(e)}

# Initialize model immediately to cache it if running purely as script
if __name__ == "__main__":
    init_model()
    # Start the handler
    runpod.serverless.start({"handler": handler})
