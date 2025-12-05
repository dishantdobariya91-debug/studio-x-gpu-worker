# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import get_pipeline
from diffusers.utils import export_to_video
from diffusers import StableDiffusionPipeline
import torch

app = FastAPI()

class RenderRequest(BaseModel):
    prompt: str
    model: str  # "CogVideoX" or "SVD-XT"

@app.get("/health")
def health_check():
    """
    Health check endpoint. Returns OK if the service is running.
    """
    return {"status": "ok"}

@app.post("/video/render")
def render_video(req: RenderRequest):
    """
    Accepts JSON {prompt: str, model: str} and generates a video.
    Returns JSON with video path or error message.
    """
    prompt = req.prompt
    model_name = req.model

    try:
        pipeline = get_pipeline(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate video frames similarly to handler.py
    if model_name == "CogVideoX":
        result = pipeline(prompt=prompt, guidance_scale=6.0, num_inference_steps=30)
        frames = result.frames[0]
    else:
        # Generate image for SVD-XT
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to("cuda")
        with torch.autocast("cuda"):
            image = sd_pipe(prompt).images[0]
        result = pipeline(image, decode_chunk_size=8)
        frames = result.frames[0]

    export_to_video(frames, "output.mp4", fps=16)
    return {"video_path": "output.mp4"}

