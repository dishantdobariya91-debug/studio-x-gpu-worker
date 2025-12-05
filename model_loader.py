from diffusers import CogVideoXPipeline
import torch

pipeline = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
)
pipeline.to("cuda")
pipeline.enable_model_cpu_offload()
