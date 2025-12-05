# model_loader.py - Load CogVideoX 1.5-5B and Stable Video Diffusion XT pipelines
import torch
from diffusers import CogVideoXPipeline, StableVideoDiffusionPipeline

def load_models():
    """Load both pipelines onto CUDA (with optimizations)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CogVideoX 1.5 (5B) in BF16 precision:contentReference[oaicite:10]{index=10}
    cog_pipe = CogVideoXPipeline.from_pretrained(
        "zai-org/CogVideoX1.5-5B", torch_dtype=torch.bfloat16
    )
    cog_pipe.to(device)
    # Offload parts to CPU to save VRAM
    cog_pipe.enable_model_cpu_offload()

    # Load Stable Video Diffusion XT (image-to-video) in FP16 (variant="fp16")
    svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", 
        torch_dtype=torch.float16, variant="fp16"
    )
    svd_pipe.to(device)
    svd_pipe.enable_model_cpu_offload()  # enable CPU offload for memory

    return cog_pipe, svd_pipe
