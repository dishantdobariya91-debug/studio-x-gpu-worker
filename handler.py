# handler.py
#
# RunPod Serverless worker for Studio X using CogVideoX.
# Endpoint: /run
#
# Expected input JSON (from Studio X / curl):
# {
#   "input": {
#       "prompt": "your text here",
#       "duration_sec": 8,
#       "resolution": "720p"
#   }
# }

import os
import tempfile
import time
from typing import Any, Dict

import numpy as np
import imageio
import runpod

from model_loader import video_model


# -------------------------------------------------------------------
# NeuroPause / NeuroChain / NeuroCloud: lightweight metadata hooks
# -------------------------------------------------------------------


def compute_neuropause_metrics(duration_sec: int) -> Dict[str, Any]:
    """
    Simple placeholder metrics so Studio X can display NP/NC/Cloud info.
    These are NOT the full scientific implementation, just sane defaults.
    """
    # Example: assume we always achieve at least 20% jolt reduction baseline
    jr_pct = 20.0
    nmi = 0.08  # NeuroMotion Instability (target <= 0.1)
    car = 0.95  # Conscious Alignment Rhythm index

    return {
        "jr_pct": jr_pct,
        "nmi": nmi,
        "car_index": car,
        "duration_sec": duration_sec,
    }


# -------------------------------------------------------------------
# Helper: convert frames -> MP4 file
# -------------------------------------------------------------------


def frames_to_video(frames, duration_sec: int, resolution: str) -> str:
    """
    Convert a list of PIL images to an MP4 on local disk.
    Returns the local file path.
    """
    tmp_dir = tempfile.mkdtemp(prefix="studiox_")
    out_path = os.path.join(tmp_dir, "output.mp4")

    # Convert PIL -> numpy
    frames_np = [np.array(f) for f in frames]

    # 8 fps baseline, adjusted by duration if possible
    if duration_sec > 0:
        fps = max(4, min(16, int(len(frames_np) / max(1, duration_sec))))
    else:
        fps = 8

    print(f"[Studio X GPU] Writing video {out_path} at {fps} fps, {len(frames_np)} frames")

    imageio.mimwrite(
        out_path,
        frames_np,
        fps=fps,
        codec="libx264",
    )

    return out_path


# -------------------------------------------------------------------
# RunPod handler
# -------------------------------------------------------------------


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main serverless job handler for RunPod."""
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "Test video from Studio X with CogVideoX")
    duration_sec = int(job_input.get("duration_sec", 8))
    resolution = job_input.get("resolution", "720p")

    print(f"[Studio X GPU] Job received with prompt='{prompt[:80]}...'")

    try:
        start = time.time()

        # 1) Generate frames
        frames = video_model.generate(prompt, duration_sec, resolution)

        # 2) Convert to video
        video_path = frames_to_video(frames, duration_sec, resolution)

        elapsed = time.time() - start
        print(f"[Studio X GPU] Job completed in {elapsed:.1f}s, video at {video_path}")

        # 3) NeuroPause / Chain / Cloud summary
        np_metrics = compute_neuropause_metrics(duration_sec)

        # NOTE: For now we return a local path.
        # Later you can upload to S3/Cloudflare and return a public URL.
        return {
            "status": "completed",
            "video_path": video_path,
            "model_id": "THUDM/CogVideoX-2b",
            "duration_sec": duration_sec,
            "resolution": resolution,
            "latency_sec": round(elapsed, 2),
            "neuro": {
                "neuropause": np_metrics,
                "neurochain": {
                    "frame_count": len(frames),
                    "fps_estimate": len(frames) / max(1, duration_sec),
                },
                "neurocloud": {
                    "worker_type": "runpod_serverless",
                    "gpu": os.environ.get("RUNPOD_GPU_NAME", "unknown"),
                },
            },
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

