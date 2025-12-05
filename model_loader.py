import torch
from diffusers import CogVideoXPipeline, StableVideoDiffusionPipeline
import time


DEVICE = "cuda"


class CogVideoXWrapper:
    def __init__(self):
        print("[CogVideoX] Loading model...")
        self.pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-1.5",
            torch_dtype=torch.float16
        ).to(DEVICE)
        print("[CogVideoX] Ready.")

    def generate(self, prompt, duration_sec, resolution):
        print("[CogVideoX] Generating video...")
        frames = self.pipe(prompt, num_frames=duration_sec * 8).frames
        # Save MP4 output
        out = "cog_output.mp4"
        self.pipe.save_video(frames, out)
        return out


class SVDXTWrapper:
    def __init__(self):
        print("[SVD-XT] Loading model...")
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16
        ).to(DEVICE)
        print("[SVD-XT] Ready.")

    def generate(self, prompt, duration_sec, resolution):
        print("[SVD-XT] Generating video...")
        frames = self.pipe(prompt, num_frames=duration_sec * 8).frames
        out = "svd_output.mp4"
        self.pipe.save_video(frames, out)
        return out


# Load both models once per worker
cog_model = CogVideoXWrapper()
svd_model = SVDXTWrapper()
