# handler.py
import runpod
from inference import generate_video

def handler(event):
    """
    RunPod serverless handler.
    Expects:
    {
      "input": {
        "prompt": "text...",
        "duration_sec": 5,
        "resolution": "720p"
      }
    }
    """

    input_data = event.get("input", {}) or {}

    prompt = input_data.get("prompt", "a simple test video")
    duration_sec = int(input_data.get("duration_sec", 4))
    resolution = input_data.get("resolution", "576p")

    try:
        video_path = generate_video(
            prompt=prompt,
            duration_sec=duration_sec,
            resolution=resolution
        )

        return {
            "status": "completed",
            "video_path": video_path,  # local path for now
            "prompt": prompt,
            "duration_sec": duration_sec,
            "resolution": resolution
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "prompt": prompt
        }

# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
