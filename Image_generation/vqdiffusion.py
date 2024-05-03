from diffusers import VQDiffusionPipeline
import torch
import json
prompt_path = '/homez/nfs_data/zhaolin/projects/coco-caption-master/prompt1-1000.json'
batch_size = 4
data = []
with open(prompt_path, 'r', encoding = 'utf-8') as f:
    for i, j in enumerate(f.readlines()):
        j = json.loads(j)
        if i%batch_size==0:
            data1 = []
        data1.append(j['caption'])
        if i%batch_size == batch_size-1:
            data.append(data1)
generator = torch.Generator("cuda").manual_seed(1000)
pipe = VQDiffusionPipeline.from_pretrained("microsoft/vq-diffusion-ithq", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "retro serie of different cars with different colors and shapes, mdjrny-v4 style"
for i, data1 in enumerate(data):
    images = pipe(prompt=data1, generator=generator, guidance_scale=7.5).images
    for j, image in enumerate(images):
        base_count = i*4+j
        image.save("vqdiffusion1.4/"+f"{base_count:05}.png")


# prompt = "a pretty girl for lesbian"
# image = pipe(prompt=prompt, generator=generator).images[0]
# image.save('./girl.png')