import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import random
import Evaluation as RM
import matplotlib.pyplot as plt
import yaml

    
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_benchmark(args):
    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        )
    else:
        device = torch.device("cpu")

    batch_size = args.batch_size


    prompts_ids = []
    with open(args.prompts_path, 'r', encoding = 'utf-8') as f:
        for i, j in enumerate(f.readlines()):
            j = json.loads(j)
            # if i%batch_size==0:
            #     data1 = []
            # data1.append(j)
            # if i%batch_size == batch_size-1:
            prompts_ids.append(j)
   
    benchmark_types = args.benchmark.split(",")
    benchmark_types = [x.strip() for x in benchmark_types]


    # resolve the generation model list to be evaluated
    model_names = args.model_names.split(",")
    model_names = [x.strip() for x in model_names]
    solver_names = args.solver_names.split(",")
    solver_names = [x.strip() for x in solver_names]
    enames = args.enames.split(",")
    enames = [x.strip() for x in enames]

    for benchmark_type in benchmark_types:
        # load the model for benchmark
        print(f"Loading {benchmark_type} model...")

        model = RM.load_score(
            name=benchmark_type, device=device, download_root=rm_path
        )

        print(f"{benchmark_type} benchmark begins!")

        # evaluate the model(s)
        with torch.no_grad():
            for model_name in model_names:
                for solver_name in solver_names:
                    model_name1 = model_name
                    for quan_name in enames:
                        result11 = []
                        score_list = []
                        for seed in [0]:
                            model_img_dirpath = os.path.join(args.img_dir, model_name1, solver_name, quan_name)
                            for i, prompt_id in tqdm(enumerate(prompts_ids)):
                                id = prompt_id['id']
                                prompt = prompt_id['caption']
                                image_path = os.path.join(model_img_dirpath, f"seed{seed}", f"{id}.png")
                                if os.path.exists(image_path):
                                    rewards = model.score(prompt, image_path)
                                    score = rewards
                                    score_list.append(score)
                                    result11.append((id,score))
                                else:
                                    continue

                        path = args.result_dir + f'/{benchmark_type}/{model_name}/'
                        os.makedirs(path, exist_ok=True)
                        with open(f'{path}/{benchmark_type}-{solver_name}step-{quan_name}-score.json',"w",) as f:
                            json.dump(dict(result11), f, indent=4, ensure_ascii=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)

    parser.add_argument(
        "--prompts_path",
        default='data/COCO_40504.json',
        type=str,
        help="Path to the prompts json list file, each item of which is a dict with keys `id` and `prompt`.",
    )
    parser.add_argument(
        "--result_dir",
        default="Image_score/COCO",
        type=str,
        help="Path to the metric results directory",
    )
    parser.add_argument(
        "--img_dir",
        default="generated_images/COCO/",
        type=str,
        help="Path to the generated images directory. The sub-level directory name should be the name of the model and should correspond to the name specified in `model`.",
    )
    parser.add_argument(
        "--model_names",
        default='dreamlike-photoreal, small-stablediffusion, stablediffusion1.2, stablediffusion1.4, stablediffusion1.5, stablediffusion2.1',
        type=str
    )
    parser.add_argument(
        "--benchmark",
        default="ImageReward, HPS, Aesthetic, CLIP",
        type=str,
        help="ImageReward, HPS, Aesthetic, CLIP, splitted with comma(,) if there are multiple benchmarks.",
    )
    parser.add_argument(
        "--rm_path",
        default=None,
        type=str,
        help="Path to place downloaded reward model in.",
    )
    parser.add_argument(
        "--gpu_id",
        default='1',
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )

    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="batch size of the prompts and images",
    )

    parser.add_argument(
        "--solver_names",
        default='DDIM10, DDIM20, DDIM50, PNDM10, PNDM20, PNDM50, DPM10, DPM20',
        type=str
    )

    parser.add_argument(
        "--enames",
        default='fp',
        type=str
    )


    args = parser.parse_args()
    if args.config is not None: # load config file and update the args
        args_dict = vars(args)
        args_dict.update(load_config(args.config))
        args = argparse.Namespace(**args_dict)

    if args.rm_path is not None:
        rm_path = os.path.expanduser(args.rm_path)
        if not os.path.exists(rm_path):
            os.makedirs(rm_path)
    else:
        rm_path = None

    test_benchmark(args)