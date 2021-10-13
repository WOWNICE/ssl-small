# This evaluates the accuracy of the pretext task. 
# first we get the v0 and v1
# next we compute the accuracy of the instance discrimination. 

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

from matplotlib import pyplot as plt
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
parser.add_argument('-s', '--sample-per-class', default=100, type=int,
                    help='train=100, val=50')
parser.add_argument('--topk', default=5, type=int,
                    help='topk accuracy of the model to estimate.')

parser.add_argument('--mode', default='d-sd', type=str,
                    help='which two views to be used for retrieval.')

parser.add_argument('--dataset-mode', default='normal', type=str,
                    help='whether use the augmented views whose shared regions are masked by gaussian noise.')


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    print(args.data)
    model = load_model(args.arch, args.pretrained)
    
    # delete the final FC layer
    model.fc = torch.nn.Identity()
    model = model.cuda()
    model.eval() # not updating anyway.

    # dataset and loader
    dataset = PositivePairDataset(args.data, args.dataset_mode)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    # list of the representations
    print("=> iterating through the dataset...")
    r0s, r1s, r2s, sample_names = [], [], [], []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(train_loader)):
            if args.dataset_mode == 'all':
                (v0, _, _, v1, v2), sample_name = batch
            else:
                (v0, v1, v2), sample_name = batch
            sample_names.extend(sample_name)

            # forward pass of the model.
            v0, v1, v2 = v0.cuda(), v1.cuda(), v2.cuda()
            r0, r1, r2 = model(v0), model(v1), model(v2)

            r0s.append(r0), r1s.append(r1), r2s.append(r2)
            
            # if step >= 10:
            #     break
    
    # all the representations
    if args.mode == 'd-sd':
        r0, r1 = F.normalize(torch.cat(r0s)), F.normalize(torch.cat(r1s))
    elif args.mode == 'sd-sd':
        r0, r1 = F.normalize(torch.cat(r1s)), F.normalize(torch.cat(r2s))
    clsr0, clsr1 = torch.split(r0, args.sample_per_class), torch.split(r1, args.sample_per_class)

    # this setting is for fair comparison between training set and val set. 
    if args.sample_per_class == 50 and 'auged-train' in args.data:
        clsr0 = clsr0[::2]
        clsr1 = clsr1[::2]

    # compute the accuracy
    # Einstein sum is more intuitive

    print("=> iterating through the representations...")
    accs = [0. for _ in range(args.topk)]
    for i, pr0 in enumerate(tqdm(clsr0)):
       
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [pr0, r0.T])
    
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [pr0, clsr1[i]]).unsqueeze(-1)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # target: positive key indicators
        target = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # since sample itself is calculated as negative, which has inner product value 1.
        # top1 must has 0 accuracy. top2 acc is in fact top1 acc.
        accs_new = accuracy(logits, target, topk=range(2, args.topk+2))
        for i in range(len(accs)):
            accs[i] += accs_new[i].detach().item()

    for i in range(len(accs)):
        accs[i] /= len(clsr0)

    print(f"Masked: {args.dataset_mode}; Mode: {args.mode}; Top1-{args.topk}: {accs}; Ckpt: {args.pretrained}; Dataset: {args.data}")

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res    

if __name__ == '__main__':
    main()
