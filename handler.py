# handler.py
import time
from model_loader import video_model

def handler(event):
    """RunPod entrypoint"""

    inp = event.get("input", {})

    prompt = inp.get("prompt", "")
    duration = inp.get("duration_sec", 4)
    resolution = inp.get("resolution", "720p")

    # Run the model
    video_url = video_model.generate(
        prompt=prompt,
        duration_sec=duration,
        resolution=resolution
    )

    return {
        "status": "completed",
        "video_url": video_url,
        "duration_sec": duration,
        "resolution": resolution
    }

# REQUIRED
from runpod.serverless import serverless_handle
serverless_handle(handler)
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
