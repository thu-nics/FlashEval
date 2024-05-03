# -*- coding: utf-8 -*-
import json
import torch
import argparse
import yaml
import os
import random

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def split_train_test(args, all_scores, use_pretrain):
    full_range = torch.arange(0, 78)
    train_id = torch.tensor(random.sample(range(79), 39))
    os.makedirs(f'{args.save_dir}/ranking_different_model/', exist_ok=True)
    torch.save(train_id, f'{args.save_dir}/ranking_different_model/random/train_id.pth')
    test_id = torch.tensor(list(set(full_range.numpy()) - set(train_id.numpy())))
    if use_pretrain==True:
        values_all = torch.load(f'{args.save_dir}/all_metrics/all_metric_COCO.pth')
    else:
        values_all = all_scores

    benchmarks = args.benchmark.split(",")
    benchmarks = [x.strip() for x in benchmarks]

    values_ori = values_all 


    score_all =[]
    value_all = []
    for i, benchmark in enumerate(benchmarks):
        score_all.append([])
        value_all.append([])
        value_all[i] = values_ori[i]
        score_all[i] = torch.mean(value_all[i], axis=1)

        os.makedirs(f'{args.save_dir}/ranking_different_model/'+'random/'+benchmark, exist_ok=True)

        score_all_normalize = score_all[i]
        torch.save(score_all_normalize, f'{args.save_dir}/ranking_different_model/random/'+benchmark+'/78model_all.pth')



    values_ori = values_all[:, train_id,:]
    score_all =[]
    value_all = []
    for i, benchmark in enumerate(benchmarks):
        score_all.append([])
        value_all.append([])
        value_all[i] = values_ori[i]
        score_all[i] = torch.mean(value_all[i], axis=1)

        score_all_normalize = score_all[i]
        torch.save(score_all_normalize, f'{args.save_dir}/ranking_different_model/random/'+benchmark+'/39model_train.pth')



    values_ori = values_all[:, test_id,:]
    score_all =[]
    value_all = []
    for i, benchmark in enumerate(benchmarks):
        score_all.append([])
        value_all.append([])
        value_all[i] = values_ori[i]
        score_all[i] = torch.mean(value_all[i], axis=1)

        score_all_normalize = score_all[i]
        torch.save(score_all_normalize, f'{args.save_dir}/ranking_different_model/random/'+benchmark+'/39model_test.pth')


def get_score(args):

    model_names = args.model_names.split(",")
    model_names = [x.strip() for x in model_names]
    solver_names = args.solver_names.split(",")
    solver_names = [x.strip() for x in solver_names]
    enames = args.enames.split(",")
    enames = [x.strip() for x in enames]
    benchmarks = args.benchmark.split(",")
    benchmarks = [x.strip() for x in benchmarks]

    q_model_names = args.q_model_names.split(",")
    q_model_names = [x.strip() for x in q_model_names]
    q_solver_names = args.q_solver_names.split(",")
    q_solver_names = [x.strip() for x in q_solver_names]
    q_enames = args.q_enames.split(",")
    q_enames = [x.strip() for x in q_enames]


    # full precision model
    choose_names = []
    for i, model_name in enumerate(model_names):
        for j, solver_name in enumerate(solver_names):
            for quan_name in enames:
                choose_names.append([model_name, solver_name, quan_name])
    # quant models
    choose_names1 = []
    for i, model_name in enumerate(q_model_names):
        for j, solver_name in enumerate(q_solver_names):
            for quan_name in q_enames:
                choose_names1.append([model_name, solver_name, quan_name])

    value_all = []
    score_dict = {}
    for i, benchmark in enumerate(benchmarks):
        value_all.append([])
        for j, name in enumerate(choose_names+choose_names1):
            value_all[i].append([])
            with open(args.load_dir+f'/{benchmark}/{name[0]}/{benchmark}-{name[1]}step-{name[2]}-score.json', 'r', encoding = 'utf-8') as f:
                score_dict[f'{name[0]}-{benchmark}-{name[1]}step-{name[2]}']=json.load(f)
                value_list = []
            for _, value in score_dict[f'{name[0]}-{benchmark}-{name[1]}step-{name[2]}'].items():
                value_list.append(value)
                value_all[i][j].append(value)
            value_all[i][j] = torch.tensor(value_all[i][j])
        value_all[i]=torch.stack(value_all[i])

    values = torch.stack(value_all)
    os.makedirs(args.save_path+'/all_metrics/')
    torch.save(values, args.save_path+'/all_metrics/all_metric_'+args.set_name+'.pth')
    return values




       
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument(
        "--save_dir",
        default='Image_score',
        type=str,
    )
    parser.add_argument(
        "--load_dir",
        default='Image_score/COCO',
        type=str,
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
        "--solver_names",
        default='DDIM10, DDIM20, DDIM50, PNDM10, PNDM20, PNDM50, DPM10, DPM20',
        type=str
    )

    parser.add_argument(
        "--enames",
        default='fp',
        type=str
    )
    parser.add_argument(
        "--q_model_names",
        default='stablediffusion1.2, stablediffusion1.4, stablediffusion1.5',
        type=str
    )
    parser.add_argument(
        "--q_solver_names",
        default='DDIM10, DDIM20, DDIM50, DPM10, DPM20',
        type=str
    )
    parser.add_argument(
        "--q_enames",
        default='_6, _8',
        type=str
    )
    parser.add_argument(
        "--set_name",
        default='COCO',
        type=str
    )
    parser.add_argument(
        "--use_pretrain",
        default=True,
        type=bool
    )

    args = parser.parse_args()

    if args.config is not None: # load config file and update the args
        args_dict = vars(args)
        args_dict.update(load_config(args.config))
        args = argparse.Namespace(**args_dict)
    if args.use_pretrain == True:
        all_scores = []
    else:
        all_scores = get_score(args)
    split_train_test(args, all_scores, args.use_pretrain)
