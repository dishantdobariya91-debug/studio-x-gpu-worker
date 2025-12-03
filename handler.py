# handler.py
#
# RunPod Serverless worker for Studio X using ZeroScope v2 576w
# Endpoint: /run  (RunPod standard)
#
# Input JSON (from Studio X / curl):
# {
#   "input": {
#     "prompt": "your text here",
#     "duration_sec": 8,
#     "resolution": "576p"
#   }
# }

import os
import tempfile
import time
from typing import Any, Dict

import numpy as np
import runpod
import torch
from diffusers import DiffusionPipeline
import imageio

MODEL_ID = "cerspense/zeroscope_v2_576w"

# --------------------------------------------------------------------
# Global model load (runs once per worker start)
# --------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"[Studio X GPU] Loading ZeroScope model: {MODEL_ID} on {device} ...")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype
)
if device == "cuda":
    pipe.enable_model_cpu_offload()
print("[Studio X GPU] Model loaded.")


# --------------------------------------------------------------------
# Helper: generate video and return local path
# --------------------------------------------------------------------
def generate_video(prompt: str, duration_sec: int, resolution: str) -> str:
    # duration -> frames (simple mapping)
    # ZeroScope works well with 24–48 frames
    num_frames = max(12, min(48, int(duration_sec * 8)))  # 8 fps baseline

    print(f"[Studio X GPU] Generating video: "
          f"prompt='{prompt[:80]}', duration={duration_sec}s, "
          f"frames={num_frames}, resolution={resolution}")

    # Run model
    start = time.time()
    result = pipe(
        prompt,
        num_frames=num_frames
    )
    frames = result.frames  # List[List[PIL.Image]] or List[PIL.Image]

    if isinstance(frames[0], list):
        frames = frames[0]

    frames_np = [np.array(f) for f in frames]

    # Save to temporary mp4
    tmp_dir = tempfile.mkdtemp(prefix="studiox_")
    video_path = os.path.join(tmp_dir, "output.mp4")

    fps = 8  # playback fps
    imageio.mimwrite(
        video_path,
        frames_np,
        fps=fps,
        codec="libx264"
    )

    print(f"[Studio X GPU] Video generated in {time.time() - start:.1f}s "
          f"and saved to {video_path}")

    return video_path


# --------------------------------------------------------------------
# RunPod handler
# --------------------------------------------------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main serverless job handler for RunPod."""
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "Test video from Studio X with ZeroScope")
    duration_sec = int(job_input.get("duration_sec", 8))
    resolution = job_input.get("resolution", "576p")

    try:
        video_path = generate_video(prompt, duration_sec, resolution)

        # For now we return local file path.
        # Later we will upload to S3/R2 and return https URL.
        return {
            "status": "completed",
            "video_path": video_path,
            "model_version": MODEL_ID,
            "duration_sec": duration_sec,
            "resolution": resolution,
        }

    except Exception as e:
        print("[Studio X GPU] Error during generation:", e)
        return {
            "status": "error",
            "error": str(e),
        }


if __name__ == "__main__":
    print("[Studio X GPU] Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
