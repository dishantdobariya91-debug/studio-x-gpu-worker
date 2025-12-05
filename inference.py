from model_loader import video_model

def generate_video(prompt, duration_sec, resolution):
    """
    Wrapper to call the loaded model.
    """
    return video_model.generate(
        prompt=prompt,
        duration_sec=duration_sec,
        resolution=resolution
    )
