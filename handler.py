import runpod
from model_loader import video_model


# -------------------------------------------------------------
# RunPod Serverless Handler
# -------------------------------------------------------------
# This function is executed every time a job is sent to:
# POST https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/run
# -------------------------------------------------------------

def generate(event):
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

        # Run your video model
        video_url = video_model.generate(
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


# -------------------------------------------------------------
# Start RunPod Serverless Worker
# -------------------------------------------------------------
runpod.serverless.start({"handler": generate})
