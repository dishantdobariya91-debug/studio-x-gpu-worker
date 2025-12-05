# inference.py
from model_loader import video_model

def generate_video(prompt: str, duration_sec: int = 4, resolution: str = "576p") -> str:
    """
    Wrapper called by handler.py.
    Returns a local path to the generated video file (e.g., /tmp/output.mp4).
    """
    return video_model.generate(
        prompt=prompt,
        duration_sec=duration_sec,
        resolution=resolution
    )
