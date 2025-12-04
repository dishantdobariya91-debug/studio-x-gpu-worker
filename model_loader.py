import torch
from diffusers import CogVideoXPipeline

class CogVideoModel:
    def __init__(self):
        print("Loading CogVideoX-2B (Text-to-Video)…")
        self.pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2B",
            torch_dtype=torch.float16
        ).to("cuda")
        print("Model ready.")

    def generate(self, prompt: str, duration_sec: int, resolution: str):
        print(f"Generating video for: {prompt}")

        output = self.pipe(
            prompt=prompt,
            num_frames=49,         # ≈2 seconds
            guidance_scale=6.0
        )

        video = output.frames[0]

        out_path = "/tmp/output.mp4"
        video.save(out_path)

        return out_path

video_model = CogVideoModel()

