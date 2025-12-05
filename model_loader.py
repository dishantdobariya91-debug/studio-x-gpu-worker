# model_loader.py
from diffusers import CogVideoXPipeline, StableVideoDiffusionPipeline, StableDiffusionPipeline
import torch

# Dictionary to cache loaded pipelines per model name
_pipelines = {}

def get_pipeline(model_name: str):
    """
    Load and return the specified model pipeline.
    Models:
      - "CogVideoX": CogVideoX 1.5 text-to-video model
      - "SVD-XT": Stable Video Diffusion XT image-to-video model
    The pipelines are cached to ensure one-time initialization per worker.
    """
    if model_name == "CogVideoX":
        if "CogVideoX" not in _pipelines:
            # Load CogVideoX 1.5 text-to-video pipeline (5B parameters)
            _pipelines["CogVideoX"] = CogVideoXPipeline.from_pretrained(
                "THUDM/CogVideoX1.5-5B",
                torch_dtype=torch.bfloat16
            )
            _pipelines["CogVideoX"].to("cuda")  # Move to GPU
        return _pipelines["CogVideoX"]

    elif model_name == "SVD-XT":
        if "SVD-XT" not in _pipelines:
            # Load Stable Video Diffusion XT (image-to-video) pipeline
            _pipelines["SVD-XT"] = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            _pipelines["SVD-XT"].to("cuda")
        return _pipelines["SVD-XT"]

    else:
        raise ValueError(f"Unsupported model '{model_name}'. Choose 'CogVideoX' or 'SVD-XT'.")
