import json
import torch
import os
import argparse
import random
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import sys
import os

current_path = os.path.dirname(os.path.abspath(sys.argv[0]))

from Evaluation.matrix_sqrt import sqrt_newton_schulz_autograd
from scipy import linalg
import time


class PrDataset(Dataset):
    def __init__(self, args):
        self.model_img_dirpath = args.img_dir
        self.prompts_ids = []
        with open(args.prompts_path, 'r', encoding = 'utf-8') as f:
            for i, j in enumerate(f.readlines()):
                j = json.loads(j)
                self.prompts_ids.append(j)

    def __getitem__(self,index):
        id = self.prompts_ids[index]['id']
        prompt = self.prompts_ids[index]['caption']
        image_path = os.path.join(self.model_img_dirpath, f"{id}.png")
        return id, prompt, image_path

    def __len__(self):
        return len(self.prompts_ids)






def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)







def frechet_distance(mu1, sigma1, mu2, sigma2, device):
    diff = mu1 - mu2
    covmean, error = sqrt_newton_schulz_autograd(torch.bmm(sigma1, sigma2), 10, dtype=torch.cuda.FloatTensor, device=device)
    # if torch.iscomplex(covmean):
    #     covmean = covmean.real

    return torch.norm(diff, dim=1) + torch.diagonal((sigma1 + sigma2 - 2 * covmean), dim1=-2, dim2=-1).sum(dim=-1)

def cal_fid(features_10, feature_gt_10, device):
    size = features_10.shape[2]
    number = features_10.shape[0]
    fids = []
    for i in range(number):
        fids.append([])
    for i in range(size):
        mu_gt = torch.mean(feature_gt_10[:,i,:], axis=0)
        sigma_gt = torch.cov(feature_gt_10[:,i,:].T)
        for j in range(number):
            mu_ge = torch.mean(features_10[j,:,i,:], axis=0)
            sigma_ge = torch.cov(features_10[j,:,i,:].T)
            print(time.time())
            # fids[j].append(torch.tensor(calculate_frechet_distance(mu_ge.cpu().numpy(), sigma_ge.cpu().numpy(), mu_gt.cpu().numpy(), sigma_gt.cpu().numpy())).to(device))
            fids[j].append(frechet_distance(mu_ge, sigma_ge, mu_gt, sigma_gt, device))
            print(time.time())

    for i in range(number):
        fids[i] = torch.stack(fids[i],dim=0)
    fids = torch.stack(fids)
    print(fids.shape)
    return fids


def batch_cov(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def cal_fid_batch(features_10, feature_gt_10, batch_size, device):
    size = features_10.shape[2]
    number = features_10.shape[0]
    fids = []
    for i in range(number):
        fids.append([])
    for i in range(size//batch_size):
        mu_gt = torch.mean(feature_gt_10[:, i*batch_size:(i+1)*batch_size,:].to(device), axis=0)
        sigma_gt = batch_cov(feature_gt_10[:, i*batch_size:(i+1)*batch_size,:].to(device).transpose(0,1))
        for j in range(number):
            mu_ge = torch.mean(features_10[j,:, i*batch_size:(i+1)*batch_size,:].to(device), axis=0)
            sigma_ge = batch_cov(features_10[j,:, i*batch_size:(i+1)*batch_size,:].to(device).transpose(0,1))
            # fids[j].append(torch.tensor(calculate_frechet_distance(mu_ge.cpu().numpy(), sigma_ge.cpu().numpy(), mu_gt.cpu().numpy(), sigma_gt.cpu().numpy())).to(device))
            fids[j].append(frechet_distance(mu_ge, sigma_ge, mu_gt, sigma_gt, device))


    for i in range(number):
        fids[i] = torch.cat(fids[i],dim=0)
    fids = torch.stack(fids)
    return fids


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{0}" if '0' is not None else "cuda"
        )
    else:
        device = torch.device("cpu")
    cal_fid_batch('feature1.pth', 'feature2.pth', 10, device)