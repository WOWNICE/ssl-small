# adapted from https://github.com/mxl1990/tsne-pytorch/blob/master/tsne_torch.py

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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
parser.add_argument('--pic-name', default='tsne.pdf', type=str,
                    help='the figure file name to be saved as.')
parser.add_argument('--perplexity', default=30., type=float,
                    help='5-50 recommended. Denser the data, higher the perplexity.')
parser.add_argument('--paint', action='store_true')
parser.add_argument('--zoomin', action='store_true')

# make sure all the new tensors be created in the cuda device.
torch.set_default_tensor_type(torch.cuda.FloatTensor)

def main(args): 
    # load_model 
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
            
            if step >= 2:
                break
    
    # all the representations
    r0, r1, r2 = F.normalize(torch.cat(r0s)), F.normalize(torch.cat(r1s)), F.normalize(torch.cat(r2s))

    # concat all the representations and visualize them
    X = torch.cat([r0[:1000,:], r1[:1000,:], r2[:1000,:]])
    with torch.no_grad():
        Y = tsne(X, perplexity=args.perplexity)
        Y0, Y1, Y2 = [y.cpu().numpy() for y in Y.split(Y.shape[0]//3)]

    # save the numpy file.
    np.save(f'{args.pic_name}.Y0.npy', Y0)
    np.save(f'{args.pic_name}.Y1.npy', Y1)
    np.save(f'{args.pic_name}.Y2.npy', Y2)
    

def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P

def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("=> computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in tqdm(range(n)):
        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P

def pca_torch(X, no_dims=50):
    print("=> preprocessing the features using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    for i in range(d):
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    print('=> t-sne optimization starts.')
    for iter in tqdm(range(max_iter)):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = (num / torch.sum(num)).type(torch.cuda.FloatTensor)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    print(f"=> final error: {torch.sum(P * torch.log(P / Q))}.")
    # Return solution
    return Y


def paint(args):
    # load the vectors
    Y0 = np.load(f'{args.pic_name}.Y0.npy')
    Y1 = np.load(f'{args.pic_name}.Y1.npy')
    Y2 = np.load(f'{args.pic_name}.Y2.npy')

    # pick first 10 classes to visualize
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams.update({'font.size': 22})
    colors = plt.cm.jet

    # make sure two representation spaces are of the same x,y limit.
    Y = np.hstack((Y0, Y1, Y2))
    xrange = 1.1 * Y[:,0].min(), 1.1 * Y[:,0].max()
    yrange = 1.1 * Y[:,1].min(), 1.1 * Y[:,1].max()

    # figure: original representation space
    fig, ax = plt.subplots()
    ax.set_xlim(xrange), ax.set_ylim(yrange)
    ax.set_xticks([]), ax.set_yticks([])
    c = []
    for i in range(10):
        c.extend([colors(i/10) for _ in range(100)])
    ax.scatter(Y0[:,0], Y0[:,1], c=c)
    ax.set_facecolor('lavender')
    plt.savefig(f'{args.pic_name}.full_rep.pdf')
    # plt.title('original representation space')

    # visualize the relationship about the samples, augmented views, and class concentration.
    # only visualize the representation space of the first class
    plt.clf()
    fig, ax = plt.subplots()
    ax.set_facecolor('lavender')
    ax.set_xlim(xrange), ax.set_ylim(yrange)
    ax.set_xticks([]), ax.set_yticks([])
    
    # original class 
    c = [colors(0) for _ in range(100)]
    ax.scatter(Y0[:100,0], Y0[:100,1], c=c, alpha=0.5)

    # augmented class
    c_aug = [colors(1) for _ in range(100)]
    ax.scatter(Y1[:100,0], Y1[:100,1], c=c_aug, alpha=0.5)
    ax.scatter(Y2[:100,0], Y2[:100,1], c=c_aug, alpha=0.5)
    # plt.title('augmented representation space')

    for i in range(100):
        ax.plot([Y0[i,0], Y1[i,0], Y2[i,0], Y0[i,0]], [Y0[i,1], Y1[i,1], Y2[i,1], Y0[i,1]], color='black')
    
    # get the best 
    if args.zoomin:
        axins = inset_axes(ax, width="50%", height="50%", loc='lower left',
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=ax.transAxes)
        
        axins.set_xticks([]), axins.set_yticks([])
        axins.set_facecolor('lightsteelblue')
        axins.scatter(Y0[:100,0], Y0[:100,1], c=c, alpha=0.5)
        axins.scatter(Y1[:100,0], Y1[:100,1], c=c_aug, alpha=0.5)
        axins.scatter(Y2[:100,0], Y2[:100,1], c=c_aug, alpha=0.5)
        for i in range(100):
            axins.plot([Y0[i,0], Y1[i,0], Y2[i,0], Y0[i,0]], [Y0[i,1], Y1[i,1], Y2[i,1], Y0[i,1]], color='black')
        
        xlim0, xlim1 = -10, -8
        ylim0, ylim1 = 41, 43
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)

        mark_inset(ax, axins, loc1=1, loc2=2, ls='--', lw=3, alpha=0.5)
        axins.text(-9.75, 42.5, 'Augmented Views', fontsize=21, fontweight='bold')

        axins.text(-9, 41.05, 'zoom in (100X)', fontsize=18)

    plt.savefig(f'{args.pic_name}.auged_rep.pdf')

if __name__ == '__main__':
    args = parser.parse_args()
    if not args.paint:
        main(args)
    else:
        paint(args)
    

