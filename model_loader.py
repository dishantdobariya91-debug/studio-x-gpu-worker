import torch
from diffusers import CogVideoXPipeline, StableVideoDiffusionPipeline
from diffusers.utils import load_image

# Cache pipelines to avoid reloading
PIPELINES = {}
def load_pipeline(model_name: str):
    if model_name == "CogVideoX1.5":
        if "cogvideo" not in PIPELINES:
            pipe = CogVideoXPipeline.from_pretrained(
                "THUDM/CogVideoX1.5-5B", torch_dtype=torch.bfloat16
            )
            # Enable CPU offloading and VAE optimizations for VRAM savings
            pipe.enable_sequential_cpu_offload()
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
            pipe.to("cuda")
            PIPELINES["cogvideo"] = pipe
        return PIPELINES["cogvideo"]
    elif model_name == "SVD-XT":
        if "svdxt" not in PIPELINES:
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float16, variant="fp16"
            )
            # Enable CPU offload to reduce memory peaks
            pipe.enable_model_cpu_offload()
            pipe.to("cuda")
            PIPELINES["svdxt"] = pipe
        return PIPELINES["svdxt"]
    else:
        raise ValueError(f"Unknown model: {model_name}")
