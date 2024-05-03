from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch
import json
import os
prompt_path = '/homez/nfs_data/zhaolin/projects/coco-caption-master/prompt1-1000.json'
batch_size = 1
data = []
with open(prompt_path, 'r', encoding = 'utf-8') as f:
    for i, j in enumerate(f.readlines()):
        j = json.loads(j)
        if i%batch_size==0:
            data1 = []
        data1.append(j['caption'])
        if i%batch_size == batch_size-1:
            data.append(data1)
# stage 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16)
stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
#stage_1.enable_model_cpu_offload()
generator = torch.Generator("cuda").manual_seed(1000)
stage_1=stage_1.to("cuda")
stage_2=stage_2.to("cuda")
stage_3=stage_3.to("cuda")
root = '/homez/nfs_data/zhaolin/generated_images/DeepFloyd/'
path = root + 'solver50_seed1000'
os.makedirs(path, exist_ok=True)
for i,prompt in enumerate(data):
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt[0])
    # stage 1
    image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
    image = stage_2(image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
    image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
    pt_to_pil(image)[0].save("deepfloyd/"+f"{i:05}.png")
