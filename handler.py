import runpod
from inference import generate_video


def handler(event):
    """
    event = {
        "input": {
            "prompt": "...",
            "duration_sec": 5,
            "resolution": "720p"
        }
    }
    """
    try:
        input_data = event.get("input", {})

        prompt = input_data.get("prompt", "A beautiful cinematic scene")
        duration = input_data.get("duration_sec", 5)
        resolution = input_data.get("resolution", "720p")

        video_url = generate_video(
            prompt=prompt,
            duration_sec=duration,
            resolution=resolution
        )

        return {
            "status": "success",
            "video_url": video_url,
            "duration_sec": duration,
            "resolution": resolution
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


runpod.serverless.start({"handler": handler})
