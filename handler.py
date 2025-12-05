import runpod
from model_loader import pipeline

def handler(job):
    prompt = job["input"].get("prompt", "")
    video = pipeline(
        prompt=prompt,
        guidance_scale=6.0,
        num_inference_steps=50
    ).frames[0]
    # Return frames (or encode/save as needed)
    return {"frames": [frame for frame in video]}

runpod.serverless.start({"handler": handler})
