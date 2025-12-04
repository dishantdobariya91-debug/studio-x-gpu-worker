import torch
from diffusers import CogVideoXPipeline


class CogVideoModel:
    """
    Minimal wrapper around CogVideoX for Studio X.
    Returns a list of frames (PIL Images).
    """

    def __init__(self):
        model_id = "THUDM/CogVideoX-2b"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"[Studio X GPU] Loading CogVideoX model {model_id} on {self.device} ...")

        self.pipe = CogVideoXPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
        )

        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload() if self.device == "cuda" else None

        print("[Studio X GPU] CogVideoX model ready.")

    def generate(self, prompt: str, duration_sec: int, resolution: str):
        """
        Generate frames for a prompt.

        Returns:
            List[PIL.Image.Image] frames
        """
        # Simple mapping: ~8 fps, cap at 64 frames
        num_frames = max(16, min(64, duration_sec * 8))

        print(
            f"[Studio X GPU] Generating video: "
            f"prompt='{prompt[:80]}...', duration={duration_sec}s, frames={num_frames}, res={resolution}"
        )

        out = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
        )

        # diffusers CogVideoX: out.frames[0] is list of PIL images
        frames = out.frames[0]
        return frames


# Global singleton model (loaded once per worker)
video_model = CogVideoModel()

