import torch
import torch.nn.functional as F
import os
import argparse
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from collections import defaultdict

import json

import moco.loader
import moco.builder

import models
from util import *

import numpy as np

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR',
                    help='path to the dataset to be sampled on.')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--gpu', default=0, type=int,
                    help='which gpu to use.')
parser.add_argument('-s', '--sample-per-class', default=50, type=int,
                    help='train=100, val=50')


def _lunif_cpu(x, t=2):
    x = x.cpu().numpy()
    # sq_pdist = torch.pdist(x, p=2).pow(2)     # not supported in AMP
    sq_pdist = torch.Tensor(spatial.distance.pdist(x, 'minkowski', p=2)).pow(2)
    return sq_pdist.mul(-t).exp().mean().log()

def _lunif_gpu(x, t=2):
    sq_pdist = torch.pdist(x, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().log().detach().item()

def uniform(x, t=2):
    """
    This function requires the representation to be unit vecs.
    """
    try:
        return _lunif_gpu(x)
    except:
        return _lunif_cpu(x)

def alignment(x1, x2):
    return (2 - 2 * (x1 * x2).sum()/x1.shape[0]).item()

def main():
    args = parser.parse_args()

    # this load_model 
    model = load_model(args.arch, args.pretrained)
    
    # delete the final FC layer
    model.fc = torch.nn.Identity()
    model = model.cuda(args.gpu)
    model.eval() # not updating anyway.

    # data set and loader
    dataset = PositivePairDataset(args.data)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    # list of the representations
    print("=> iterating through the dataset...")
    r0s, r1s, r2s, sample_names = [], [], [], []
    with torch.no_grad():
        for step, ((v0, v1, v2), sample_name) in enumerate(tqdm(train_loader)):
            sample_names.extend(sample_name)

            # forward pass of the model.
            v0, v1, v2 = v0.cuda(args.gpu), v1.cuda(args.gpu), v2.cuda(args.gpu)
            r0, r1, r2 = model(v0), model(v1), model(v2)

            r0s.append(r0), r1s.append(r1), r2s.append(r2)
            
            # if step >= 10:
            #     break
    
    # all the representations
    r0, r1, r2 = F.normalize(torch.cat(r0s)), F.normalize(torch.cat(r1s)), F.normalize(torch.cat(r2s))

    #===========================================================================================
    # Alignment: instance-level statistic
    #===========================================================================================
    print("=> calculating instance-level alignment...")

    align_d_sd = alignment(r0, r1)
    align_sd_sd = alignment(r1, r2)
    
    #===========================================================================================
    # Alignment: class-level distribution
    #===========================================================================================
    class_num, sample_num = 1000, args.sample_per_class
    assert(len(sample_names) == class_num * sample_num)
    for c in range(class_num):
        names = sample_names[c*sample_num:(c+1)*sample_num]
        names = [x.split('_')[0] for x in names]
        assert(names[1:] == names[:-1])
    
    print("=> calculating class-level alignment...")

    def intra_cls_alignment(x):
        return torch.pdist(x, p=2).pow(2).mean().detach().item()

    r0_cls = torch.split(r0, sample_num)
    align_cls_d = [intra_cls_alignment(x) for x in r0_cls]
    r1_cls = torch.split(r1, sample_num)
    align_cls_sd = [intra_cls_alignment(x) for x in r1_cls]
    
    avg_align_d = np.array(align_cls_d).mean()
    avg_align_sd = np.array(align_cls_sd).mean()

    #===========================================================================================
    # Uniformity: instance-level statistic
    #===========================================================================================
    # Instead of directly computing the uniformity requires n*n complexity, 
    # we compute the averaged uniformity.
    print("=> calculating instance-level uniformity...")
    ind, n, t, uniform_d, uniform_sd = torch.randperm(r0.shape[0]), 128, 3, 0., 0.
    for i in range(t):
        local_ind = ind[i*n:(i+1)*n]
    
        uniform_d += uniform(r0[local_ind])
        uniform_sd += uniform(r1[local_ind])
    
    uniform_d /= t
    uniform_sd /= t

    print(f"{args.pretrained},{args.data}\n{'D-SD align':<15}{'SD-SD align':<15}{'D cls-align':<15}{'SD cls-align':<15}{'D uniform':<15}{'SD uniform':<15}\n{align_d_sd:<15.4f}{align_sd_sd:<15.4f}{avg_align_d:<15.4f}{avg_align_sd:<15.4f}{uniform_d:<15.4f}{uniform_sd:<15.4f}")

if __name__ == '__main__':
    main()


