# This script shows to what degree the neighborhood is filled with noise.
# we only care about the precision of the positive sample under certain threshold. 

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
parser.add_argument('--encoder-q', action='store_true',
                    help='which encoder to evaluate.')

thresholds = [i / 20 - 1 for i in range(41)]
thresholds[-1] = 0.999

def main():
    args = parser.parse_args()
    print(args.data)
    model = load_model(args.arch, args.pretrained, args.encoder_q)
    
    # delete the final FC layer
    model.fc = torch.nn.Identity()
    model = model.cuda(args.gpu)
    model.eval() # not updating anyway.

    # dataset and loader
    dataset = PositivePairDataset(args.data)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    # list of the representations
    print("=> iterating through the dataset...")
    r0s, r1s, sample_names = [], [], []
    with torch.no_grad():
        for step, ((v0, v1, _), sample_name) in enumerate(tqdm(train_loader)):
            sample_names.extend(sample_name)

            # forward pass of the model.
            v0, v1 = v0.cuda(args.gpu), v1.cuda(args.gpu)
            r0, r1 = model(v0), model(v1)

            r0s.append(r0), r1s.append(r1)
            
            # if step >= 10:
            #     break
    
    # all the representations
    r0, r1 = F.normalize(torch.cat(r0s)), F.normalize(torch.cat(r1s))
    
    # this setting is for fair comparison between training set and val set. 
    if args.sample_per_class == 50 and 'auged-train' in args.data:
        clsr0, clsr1 = torch.split(r0, args.sample_per_class), torch.split(r1, args.sample_per_class)
        clsr0 = clsr0[::2]
        clsr1 = clsr1[::2]
        r0, r1 = torch.cat(clsr0), torch.cat(clsr1)

    print("=> put reps to other gpus...")
    r0, r1 = r0.cuda(args.gpu+1), r1.cuda(args.gpu+2)

    # similarity matrix for the original view and the augmented view.
    sim0, sim1 = r0.matmul(r0.T), r1.matmul(r1.T)

    origin_prec = prec_cal(sim0, thresholds, 50)
    auged_prec = prec_cal(sim1, thresholds, 50)

    # print("=> precision for original space.")
    # print(origin_prec)
    # print("=> precision for auged space.")
    # print(auged_prec)

    x = np.arange(len(origin_prec))
    x_labels = [f"{thresholds[i]:.2f}" for i in range(len(thresholds))]
    x_labels = ['-'.join(x_labels[i:i+2]) for i in range(len(x_labels)-1)]
    
    total_width, n = 1., 2
    width = total_width / n
    x = x - (total_width - width)

    plt.rcParams['figure.figsize'] = [10, 10]
    plt.bar(x, origin_prec, width=width, label='original')
    plt.bar(x + width, auged_prec, width=width, label='augmented')
    
    plt.ylim(0, 0.6)
    plt.ylabel("Precision")
    plt.xlabel("Threshold")

    plt.xticks(x, x_labels, rotation='vertical')

    plt.legend(loc='best')
    plt.savefig('prec.pdf')
    print("=> saved as prec.pdf")


def prec_cal(sim_matrix, thresholds, sample_per_class):
    assert(sim_matrix.shape[0] % sample_per_class == 0)

    class_num = sim_matrix.shape[0] // sample_per_class

    precs = [0 for _ in range(len(thresholds)-1)]
    for i in range(class_num):
        pos = sim_matrix[i*sample_per_class:(i+1)*sample_per_class, i*sample_per_class:(i+1)*sample_per_class]
        pos_neg = sim_matrix[i*sample_per_class:(i+1)*sample_per_class,:]
        
        # exclude sample itself.
        for j in range(len(thresholds)-1):
            lo, hi = thresholds[j], thresholds[j+1]

            # between lo and hi
            recalled = ((pos_neg >= lo).logical_and(pos_neg < hi)).sum().detach().item() 
            true_pos = ((pos >= lo).logical_and(pos < hi)).sum().detach().item()
            if recalled:
                precs[j] += true_pos / recalled

    for i in range(len(precs)):
        precs[i] /= class_num
    
    return precs

if __name__ == '__main__':
    main()
