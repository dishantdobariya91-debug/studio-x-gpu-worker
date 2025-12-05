from fastapi import FastAPI
from pydantic import BaseModel
from model_loader import load_pipeline
from diffusers.utils import load_image
import torch

app = FastAPI()

class GenRequest(BaseModel):
    model: str
    prompt: str = None
    image_url: str = None

@app.post("/generate")
async def generate_video(req: GenRequest):
    pipe = load_pipeline(req.model)
    if req.model == "CogVideoX1.5":
        res = pipe(prompt=req.prompt, num_inference_steps=50, num_frames=81,
                   guidance_scale=6,
                   generator=torch.Generator(device="cuda").manual_seed(42))
        frames = res.frames[0]
    else:
        img = load_image(req.image_url)
        res = pipe(img, num_inference_steps=50, num_frames=25,
                   generator=torch.Generator(device="cuda").manual_seed(42))
        frames = res.frames[0]
    # Convert frames to a video file or base64; omitted here
    return {"status": "done", "num_frames": len(frames)}
