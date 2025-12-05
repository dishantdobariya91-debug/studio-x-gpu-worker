# handler.py
import runpod
from model_loader import get_pipeline
from diffusers.utils import export_to_video
import torch

def handler(job):
    """
    RunPod serverless handler. Expects job["input"] with keys:
      - "prompt": text prompt (string)
      - "model": either "CogVideoX" or "SVD-XT"
    Returns a dict; here we save the video to 'output.mp4' and return its path.
    """
    job_input = job.get("input", {})
    prompt = job_input.get("prompt", "")
    model_name = job_input.get("model", "CogVideoX")
    
    if not prompt:
        return {"error": "No prompt provided."}

    try:
        pipeline = get_pipeline(model_name)
    except ValueError as e:
        return {"error": str(e)}

    # Generate video frames
    if model_name == "CogVideoX":
        # Text-to-video generation
        result = pipeline(
            prompt=prompt,
            num_inference_steps=30,  # adjust for quality/speed trade-off
            guidance_scale=6.0
        )
        frames = result.frames[0]  # CogVideoX returns .frames list
    else:
        # SVD-XT normally requires an image. Generate a conditioning image from text.
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to("cuda")
        with torch.autocast("cuda"):
            image = sd_pipe(prompt).images[0]
        # Run image-to-video pipeline
        result = pipeline(
            image, 
            decode_chunk_size=8
        )
        frames = result.frames[0]

    # Export frames to a video file (MP4, 16 fps)
    export_to_video(frames, "output.mp4", fps=16)
    return {"video_path": "output.mp4"}

if __name__ == "__main__":
    # Start the RunPod serverless handler (listens on /run)
    runpod.serverless.start({"handler": handler})
