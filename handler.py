import runpod
from inference import generate_video

def handler(event):
    try:
        input_data = event.get("input", {})

        prompt = input_data.get("prompt", "A cinematic scene")
        duration = input_data.get("duration_sec", 4)
        resolution = input_data.get("resolution", "576p")
        model_name = input_data.get("model", "cog")  # cog or svd

        video_url = generate_video(
            prompt=prompt,
            duration_sec=duration,
            resolution=resolution,
            model_name=model_name
        )

        return {
            "status": "success",
            "model": model_name,
            "video_url": video_url,
            "duration_sec": duration,
            "resolution": resolution
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
