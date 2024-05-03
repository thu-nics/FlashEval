import os
from diffusers import DiffusionPipeline,StableDiffusionPipeline, VersatileDiffusionTextToImagePipeline, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler,PNDMScheduler
import torch
import argparse
import json
import time
from huggingface_hub import login
# login()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_path",
        default='Coco_caption/prompt-image.json',
        type=str,
        help="json for prompts path",
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="batch size for generating images",
    )
    parser.add_argument(
        "--model_name",
        default='stablediffusion1.4',
        type=str,
        help="model name: small-stablediffusion, stablediffusion1.2, stablediffusion1.4, stablediffusion1.5, stablediffusion2.1, stablediffusionXL, dreamlike-photoreal",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="the seed for generating image",
    )
    parser.add_argument(
        "--step",
        default=10,
        type=int,
        help="the step for generating image",
    )
    parser.add_argument(
        "--guidance_scale",
        default=7.5,
        type=float,
        help="clip guidance scale",
    )
    parser.add_argument(
        "--scheduler",
        default="PNDM",
        type=str,
        help="scheduler, choose from PNDM, EulerDiscrete, DPM, DDIM",
    )

    parser.add_argument(
        "--gpu_id",
        default='6',
        type=str,
        help="gpu_id",
    )

    parser.add_argument(
        "--save_dir",
        default = 'image_dir',
        type=str,
        help="save dir of generated images",
    )



    args = parser.parse_args()


    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        )
    else:
        device = torch.device("cpu")


    data = []
    id = []
    with open(args.prompt_path, 'r', encoding = 'utf-8') as f:
        for i, j in enumerate(f.readlines()):
            j = json.loads(j)
            if i%args.batch_size==0:
                data1 = []
                id1 = []
            data1.append(j['caption'])
            id1.append(j['id'])
            if i%args.batch_size == args.batch_size-1:
                data.append(data1)
                id.append(id1)

    if args.model_name == 'versatile':
        model_id = "shi-labs/versatile-diffusion"
    elif args.model_name == 'openjourney':
        model_id = "prompthero/openjourney"
    elif args.model_name == 'stablediffusion1.2':
        model_id = 'CompVis/stable-diffusion-v1-2'
    elif args.model_name == 'stablediffusion1.4':
        # model_id = path + 'models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b/'
        model_id = "CompVis/stable-diffusion-v1-4"
    elif args.model_name == 'stablediffusion1.5':
        model_id = 'runwayml/stable-diffusion-v1-5'
    elif args.model_name == 'dreamlike-photoreal':
        model_id = "dreamlike-art/dreamlike-photoreal-2.0"
        # model_id = "/homez/nfs_data/zhaolin/pretrained_model/dreamlike-photoreal/d9e27ac81cfa72def39d74ca673219c349f0a0d5/"
    elif args.model_name == 'stablediffusion2.1':
        model_id = 'stabilityai/stable-diffusion-2-1-base'
    elif args.model_name == 'stablediffusionXL':
        model_id = "stabilityai/stable-diffusion-xl-base-0.9"
    elif args.model_name == 'small-stablediffusion':
        model_id = "OFA-Sys/small-stable-diffusion-v0"
    # elif args.model_name == 'vqdiffusion':
    #      model_id = "microsoft/vq-diffusion-ithq"
    generator = torch.Generator("cuda").manual_seed(args.seed)

    if args.scheduler == 'PNDM':
        scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif args.scheduler == 'DPM':
        if args.model_name != 'stablediffusionXL':
            scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif args.scheduler == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    elif args.scheduler == 'DDIM':
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    if args.model_name == 'versatile':
        pipe = VersatileDiffusionTextToImagePipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    # elif args.model_name == 'vqdiffusion':
    #     pipe = VQDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    elif args.model_name == 'stablediffusionXL':
        # time.sleep(3)
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)

        # pipe.safety_checker = lambda images, clip_input: (images, False)
    # pipe.remove_unused_weights()
    scheduler.set_timesteps(args.step)
    pipe = pipe.to(device)
    path = args.save_dir+'/'+ args.model_name + f'/{args.scheduler}' + f"{args.step}"+ f'/fp/' +f"seed{args.seed}" + '/'
    os.makedirs(path, exist_ok=True)
    for i, data1 in enumerate(data):
        if os.path.exists(path+'/'+f"{id[i][0]}.png"):
            continue
        images = pipe(prompt=data1, generator=generator, num_inference_steps=args.step, guidance_scale=7.5).images
        for j, image in enumerate(images):
            # base_count = i*4+j
            image.save(path+'/'+f"{id[i][j]}.png")
