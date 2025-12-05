from model_loader import cog_model, svd_model

def generate_video(prompt, duration_sec, resolution, model_name):
    if model_name == "svd":
        return svd_model.generate(prompt, duration_sec, resolution)
    else:
        return cog_model.generate(prompt, duration_sec, resolution)
