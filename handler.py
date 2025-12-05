# handler.py - Main RunPod serverless handler using FastAPI-like server
import runpod
import torch
import requests
from io import BytesIO
from PIL import Image
from diffusers.utils import export_to_video
from model_loader import load_models  # helper to load pipelines

# Load both pipelines once at startup
cog_pipe, svd_pipe = load_models()

def handler(job):
    """
    RunPod handler function. Expects job["input"] to have either 'prompt' or 'image_url'.
    Returns {"video_path": <local_mp4_path>} or error.
    """
    job_input = job.get("input", {})
    prompt = job_input.get("prompt")
    image_url = job_input.get("image_url")

    output_path = "/workspace/output.mp4"

    # Text-to-Video with CogVideoX
    if prompt:
        # Generate video frames from prompt (CogVideoX in BF16):contentReference[oaicite:5]{index=5}
        result = cog_pipe(prompt=prompt, num_inference_steps=50)  # use default height/width
        frames = result.frames  # Tensor of shape [batch, frames, C, H, W]
        # Move to CPU and convert to numpy list of frames
        frames = frames[0].cpu().numpy()  # shape: [num_frames, C, H, W]
        # Convert to list of HxWxC uint8 images
        frames = [((frame.transpose(1, 2, 0) * 255).clip(0,255).astype('uint8')) for frame in frames]
        # Export to MP4 (fps=8 for CogVideoX):contentReference[oaicite:6]{index=6}
        export_to_video(frames, output_path, fps=8)
        return {"video_path": output_path}

    # Image-to-Video with Stable Video Diffusion XT
    elif image_url:
        # Download and load the image
        response = requests.get(image_url)
        if response.status_code != 200:
            return {"error": f"Failed to download image from URL: {image_url}"}
        image = Image.open(BytesIO(response.content)).convert("RGB")
        # Resize image to 1024x576 (width x height) as recommended:contentReference[oaicite:7]{index=7}
        image = image.resize((1024, 576))
        # Generate video frames from image (SVD-XT):contentReference[oaicite:8]{index=8}
        result = svd_pipe(image, decode_chunk_size=8)  # use default num_frames=25
        frames = result.frames[0]  # list of PIL Images or numpy frames
        # If frames are tensors, convert to numpy as above
        if isinstance(frames[0], torch.Tensor):
            frames = [((frame.cpu().numpy().transpose(1, 2, 0) * 255).clip(0,255).astype('uint8')) 
                      for frame in frames]
        # Export to MP4 (fps=7 for SVD-XT):contentReference[oaicite:9]{index=9}
        export_to_video(frames, output_path, fps=7)
        return {"video_path": output_path}

    else:
        return {"error": "Please provide either a 'prompt' or an 'image_url' in the input."}

# Start the RunPod serverless worker with this handler
runpod.serverless.start({"handler": handler})
