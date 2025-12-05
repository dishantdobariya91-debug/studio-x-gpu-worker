# handler.py
import runpod
import torch
from diffusers.utils import load_image
from model_loader import load_pipeline

def handler(job):
    data = job["input"]
    model_name = data.get("model")
    prompt = data.get("prompt", "")
    pipeline = load_pipeline(model_name)
    if model_name == "CogVideoX1.5":
        # Text-to-video
        result = pipeline(prompt=prompt, num_videos_per_prompt=1,
                          num_inference_steps=50, num_frames=81,
                          guidance_scale=6,
                          generator=torch.Generator(device="cuda").manual_seed(42)
                         )
        frames = result.frames[0]
    else:
        # SVD-XT is image->video; assume an image URL is given
        image = load_image(data.get("image_url"))
        result = pipeline(image, num_inference_steps=50, num_frames=25,
                          generator=torch.Generator(device="cuda").manual_seed(42))
        frames = result.frames[0]
    # Here you would encode or save `frames` to return. For brevity, returning a placeholder.
    return {"status": "generated", "frame_count": len(frames)}

# Start the Runpod serverless handler
runpod.serverless.start({"handler": handler})
