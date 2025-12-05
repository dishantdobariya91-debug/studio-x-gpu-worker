# model_loader.py
import torch
from diffusers import DiffusionPipeline

class TextToVideoModel:
    def __init__(self):
        print("[Studio X GPU] Loading ModelScope Text-to-Video 1.7B...")
        # Model: open-source text-to-video, works on 24GB GPUs
        self.pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()
        print("[Studio X GPU] Model loaded and ready.")

    def generate(self, prompt: str, duration_sec: int = 4, resolution: str = "576p") -> str:
        """
        Generate a short video from text prompt.
        ModelScope usually outputs ~16 frames; we keep it simple and let
        the model decide length; duration_sec is just for your UI.
        """

        print(f"[Studio X GPU] Generating video for prompt: {prompt}")
        # The ModelScope pipeline uses num_frames, we’ll keep default ~16
        output = self.pipe(prompt=prompt)

        frames = output.frames[0]  # list of PIL images

        # Save video to /tmp/output.mp4
        import imageio
        import os

        os.makedirs("/tmp", exist_ok=True)
        output_path = "/tmp/output.mp4"

        # 8 fps → ~2 seconds. You can tune fps.
        imageio.mimsave(output_path, frames, fps=8)

        print(f"[Studio X GPU] Video saved to {output_path}")
        # For now we just return the local path; later you’ll upload to R2/S3
        return output_path


# Global singleton instance
video_model = TextToVideoModel()

