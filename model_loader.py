import time

class DummyVideoModel:
    def __init__(self):
        print("[Studio X GPU] Loading model...")
        time.sleep(1)
        print("[Studio X GPU] Model ready.")

    def generate(self, prompt: str, duration_sec: int, resolution: str) -> str:
        print(f"[Studio X GPU] Generating video for: {prompt}")
        time.sleep(2)
        return "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4"

video_model = DummyVideoModel()
