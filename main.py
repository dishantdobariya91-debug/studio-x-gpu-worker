from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import time
from model_loader import video_model

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    duration_sec: int = 10
    resolution: str = "720p"

class GenerateResponse(BaseModel):
    status: str
    video_url: str
    model_version: Optional[str] = None
    duration_sec: int
    resolution: str

@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    video_url = video_model.generate(
        prompt=req.prompt,
        duration_sec=req.duration_sec,
        resolution=req.resolution,
    )

    return GenerateResponse(
        status="completed",
        video_url=video_url,
        model_version="studio-x-gpu",
        duration_sec=req.duration_sec,
        resolution=req.resolution,
    )
