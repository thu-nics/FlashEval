import torch
import argparse
import numpy as np
import yaml
import os
import json

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config



'''get the KD score for some idx'''
def get_kd(l1,l2,benchmark,device):
    # torch tensor: [N_model,N_sample]
    # print(l1.shape, l2.shape)
    assert l1.shape == l2.shape  # broadcast gt into same shape as input
    N_prompt_per_set, N_set = l1.shape
    # kd calculation args
    C = torch.zeros([1,N_set],device=device)  # concordant pairs
    D = torch.zeros([1,N_set],device=device)  # discordant pairs
    if benchmark == 'ImageReward':
        margin = 3.1007744780752645e-05
    elif benchmark == 'Aesthetic':
        margin = 3.855898086388239e-06
    elif benchmark == 'CLIP':
        margin = 4.690507962729988e-06
    elif benchmark == 'HPS':
        margin = 2.748667954632944e-06
    l1_datapair_within_margin = torch.zeros([1,N_set],device=device)
    l2_datapair_within_margin = torch.zeros([1,N_set],device=device)
    all_within_margin = torch.zeros([1,N_set],device=device)  
    for i in range(l1.shape[0]):
        for j in range(i+1,l1.shape[0]):
            l1_datapair_within_margin += ((torch.abs(l1[i]-l1[j])<=margin)*(torch.abs(l2[i]-l2[j])>margin)).int()  # [N_set]
            l2_datapair_within_margin += ((torch.abs(l2[i]-l2[j])<=margin)*(torch.abs(l1[i]-l1[j])>margin)).int()
            all_within_margin += ((torch.abs(l2[i]-l2[j])<=margin)*(torch.abs(l1[i]-l1[j])<=margin)).int()
            
            C += (((l1[i]-l1[j])*(l2[i]-l2[j])>0)*(torch.abs(l1[i]-l1[j])>margin)*(torch.abs(l2[i]-l2[j])>margin)).int()
            D += (((l1[i]-l1[j])*(l2[i]-l2[j])<0)*(torch.abs(l1[i]-l1[j])>margin)*(torch.abs(l2[i]-l2[j])>margin)).int()
    kendall_tau = (C-D)/torch.sqrt((C+D+l1_datapair_within_margin)*(C+D+l2_datapair_within_margin))
    return kendall_tau


def to_choose_top_set(all_random_set, values_all, train_id, gt_ranking, random_n, top_number, benchmark, per_iter, device):
    if random_n <= per_iter:
        random_n_per_eee = random_n
        eee_number = 1
    else:
        random_n_per_eee = per_iter
        eee_number = random_n // random_n_per_eee
    #在n_eval中进行多次sample
    kds = []
    for eee in range(eee_number):
        current_lst = all_random_set[:,random_n_per_eee*eee:random_n_per_eee*(eee+1)].to(device)
        values_train = values_all[train_id,:]
        value_10 = values_train[:, current_lst]
        score_10_set = torch.mean(value_10, axis=1)
        score_10_normalize = score_10_set.to(device)
        kd = get_kd(score_10_normalize, gt_ranking.squeeze(0).unsqueeze(-1).repeat(1, score_10_normalize.shape[1]), benchmark, device)
        kds.append(kd.squeeze())

    kds = torch.cat(kds, dim=0)

    _, topk_indice = torch.topk(kds, top_number, largest=True) 
    getted_top_set = all_random_set[:, topk_indice.squeeze()].transpose(0,1)
    return getted_top_set, kds



def to_choose_top_prompts(getted_top_set, frequency, random_n, item,device):
    unique_values, counts = torch.unique(getted_top_set, return_counts=True)
    sorted_index = torch.argsort(counts, descending=True)

    sorted_unique_values = unique_values[sorted_index]
    choose_prompts = sorted_unique_values[:frequency]

    if choose_prompts.shape[0] == item:
        return choose_prompts,choose_prompts,choose_prompts

    random_indices = torch.randint(0, frequency, (item, random_n), device=device)
    current_lst = choose_prompts[random_indices]

    return current_lst, sorted_unique_values[frequency:], choose_prompts

def list_of_ints(arg):
	return list(map(int, arg.split(',')))

def search_set():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument(
        "--gpu_id",
        default='0',
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )

    parser.add_argument(
        "--mode",
        default='random',
        type=str,
    )

    parser.add_argument(
        "--range",
        default='40504',
        type=int,
    )

    parser.add_argument(
        "--benchmark",
        default='HPS',
        type=str,
        help = 'ImageReward, CLIP, HPS, Aesthetic, FID'
    )

    parser.add_argument(
        "--item",
        default=10,
        type=int,
    )


    parser.add_argument(
        "--per_iter",
        default=1000000,
        type=int,
    )

    parser.add_argument(
        "--load_root",
        default='Image_score',
        type=str,
    )

    parser.add_argument(
        "--iter_range",
        default=11,
        type=int,
    )
    parser.add_argument(
        "--set_name",
        default='COCO',
        type=str
    )
    parser.add_argument(
        "--n",
        default=3000000,
        type=int,
    )
    parser.add_argument(
        "--top_k",
        default=100000,
        type=int,
    )
    parser.add_argument(
        "--frequencys",
        default='30000, 20000, 15000, 12000, 10000, 8000, 5000, 3000, 1000, 500, 100',    
        type=str
    )
    parser.add_argument(
        "--source_path",
        default='data/COCO_40504.json',    
        type=str
    )
    parser.add_argument(
        "--save_dir",
        default='result/',    
        type=str
    )     

    args = parser.parse_args()

    if args.config is not None: # load config file and update the args
        args_dict = vars(args)
        args_dict.update(load_config(args.config))
        args = argparse.Namespace(**args_dict)

    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        )
    else:
        device = torch.device("cpu")


    full_range = torch.arange(0, 78)
    random_id_train = torch.load(f'{args.load_root}/ranking_different_model/train_id.pth')
    random_id_test = torch.tensor(list(set(full_range.numpy()) - set(random_id_train.numpy())))

    
    scores = torch.load(f'{args.load_root}/all_metrics/all_metric_'+args.set_name+'.pth')
    if args.benchmark == 'ImageReward':
        values_all = scores[0].to(device)
    elif args.benchmark == 'HPS':
        values_all = scores[1].to(device)
    elif args.benchmark == 'Aesthetic':
        values_all = scores[2].to(device)
    elif args.benchmark == 'CLIP':
        values_all = scores[3].to(device)

    random_ranking_test = torch.load(os.path.join(f'{args.load_root}/ranking_different_model/random', args.benchmark, '39model_test.pth')).to(device)
    random_ranking_train = torch.load(os.path.join(f'{args.load_root}/ranking_different_model/random', args.benchmark, '39model_train.pth')).to(device)
   


    frequencys = args.frequencys.split(",")
    frequencys = [int(x.strip()) for x in frequencys]
    range1 = args.iter_range

    n = args.n
    random_n_choose_prompt = [n,n,n,n,n,n,n,n,n,n,n,n,n]
    random_n_choose_set = [n,n,n,n,n,n,n,n,n,n,n,n,n]       
    top_k = args.top_k
    top_number = [top_k,top_k,top_k,top_k,top_k,top_k,top_k,top_k,top_k,top_k,top_k,top_k]


    all_tensor = torch.arange(0,args.range).to(device)

   
    random_indices = torch.randint(0, args.range, (args.item, n), device=device)
    current_top_set = all_tensor[random_indices].to(device)

    kd_trains = [] 
    kd_tests = []
    top_sets_iters = []
    for iter in range(range1):
        getted_top_set, kds = to_choose_top_set(current_top_set, values_all, random_id_train, random_ranking_train, random_n_choose_set[iter], top_number[iter], args.benchmark, args.per_iter, device)
        current_top_set, prompts, choose_prompts = to_choose_top_prompts(getted_top_set, frequencys[iter], random_n_choose_prompt[iter], args.item, device)
        values_train = values_all[random_id_train,:]
        random_n_per_eee = args.per_iter
        if random_n_choose_set[iter] <= args.per_iter:
            random_n_per_eee = random_n_choose_set[iter]
            eee_number = 1
        else:
            random_n_per_eee = args.per_iter
            eee_number = random_n_choose_set[iter] // random_n_per_eee
        
        if len(current_top_set.shape) !=1:
            kds = []
            for eee in range(eee_number):
                current_lst = current_top_set[:,random_n_per_eee*eee:random_n_per_eee*(eee+1)].to(device)
                value_10 = values_train[:, current_lst]
                score_10_set = torch.mean(value_10, axis=1)
                score_10_normalize = score_10_set.to(device)
                kd = get_kd(score_10_normalize, random_ranking_train.squeeze(0).unsqueeze(-1).repeat(1, score_10_normalize.shape[1]), args.benchmark, device)
                kds.append(kd.squeeze())

            kds = torch.cat(kds, dim=0)
            current_top_set_one = current_top_set[:,torch.argmax(kds)]
        else:
            current_top_set_one = current_top_set

        top_sets_iters.append(current_top_set_one)
        values_train = values_all[random_id_train,:] 

        value_10 = values_train[:, current_top_set_one]
        score_10_set = torch.mean(value_10, axis=1)
        score_10_normalize = score_10_set.to(device).unsqueeze(1)
        kd_train = get_kd(score_10_normalize, random_ranking_train.squeeze(0).unsqueeze(-1).repeat(1, score_10_normalize.shape[1]), args.benchmark, device).squeeze()
        kd_trains.append(kd_train)

        values_val = values_all[random_id_test,:]
        value_10 = values_val[:, current_top_set_one].squeeze()
        score_10_set = torch.mean(value_10, axis=1)
        score_10_normalize = score_10_set.to(device).unsqueeze(1)
        kd_test = get_kd(score_10_normalize, random_ranking_test.squeeze(0).unsqueeze(-1).repeat(1, score_10_normalize.shape[1]), args.benchmark, device).squeeze()
        kd_tests.append(kd_test)
        print(f"KD values for ranking training models of {iter+1} iteration:", round(float(kd_train),3))
        print(f"KD values for ranking testing models of {iter+1} iteration:", round(float(kd_test),3))

    top_set = top_sets_iters[torch.argmax(torch.tensor(kd_tests))]
    os.makedirs(args.save_dir, exist_ok=True)
    # save the representative subset
    with open(args.save_dir+f'/{args.set_name}_{args.benchmark}_searched_subset_{args.item}prompts.json', "w") as fw:
        for i in top_set:
            with open(args.source_path, 'r', encoding = 'utf-8') as f:
                for j, line in enumerate(f):
                    if i == j:
                        json.dump(json.loads(line), fw)
                        fw.write('\n')

    #有时候取50-item，导出是48或者49个，是因为有的prompt出现了两次

        
      

if __name__ == "__main__":
    search_set()