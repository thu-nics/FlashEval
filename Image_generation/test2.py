from huggingface_hub import login
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler
import torch
import time

model_id = "stabilityai/stable-diffusion-2-1-base"
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe.to("cuda")


prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]