# handler.py
# Minimal RunPod serverless handler to verify worker health.

import time
from typing import Any, Dict

import runpod


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Simple echo handler – NO GPU / model usage yet."""
    job_input = job.get("input", {}) or {}

    prompt = job_input.get("prompt", "test prompt")
    duration_sec = int(job_input.get("duration_sec", 5))
    resolution = job_input.get("resolution", "576p")

    print(f"[Studio X] Received job | prompt='{prompt[:60]}' "
          f"duration={duration_sec}s resolution={resolution}")

    # simulate some work
    time.sleep(1)

    return {
        "status": "ok",
        "prompt": prompt,
        "duration_sec": duration_sec,
        "resolution": resolution,
        "message": "Minimal handler ran successfully."
    }


if __name__ == "__main__":
    print("[Studio X] Starting minimal RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
