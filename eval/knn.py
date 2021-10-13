import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import argparse
from torch.multiprocessing import Process, Queue
import importlib
import time
import torchvision.datasets as datasets

from util import *

import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def eval_knn(model, args):
    start_time = time.time()
    # prep
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = datasets.ImageFolder(traindir, trans)
    
    inds = np.array_split(range(len(trainset)), args.gpus)
    train_loaders = [torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset, ind),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    ) for ind in inds]

    testset = datasets.ImageFolder(valdir, trans)
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # queue
    queue, exit_queue = Queue(), Queue()
    processes = [
        Process(target=load_tensor_single,
                args=(gpu, model, train_loader, test_loader, args.k, queue, exit_queue, args.rep_space))
        for (gpu, train_loader) in enumerate(train_loaders)
    ]

    ##########################################
    for p in processes:
        p.start()

    # synchronization
    lst = []
    for i in range(len(processes)):
        item = queue.get()
        lst.append(item)
        exit_queue.put(item[0]) # put the pid back.

    for p in processes:
        p.join()
    ##########################################

    labels = torch.cat([item[1].cuda(0) for item in lst], dim=1)
    distances = torch.cat([item[2].cuda(0) for item in lst], dim=1)
    test_y = lst[0][-1].cuda(0)

    # print(labels.shape, distances.shape, test_y.shape)
    # from 1-nn to k-nn
    best_knn, best_k = 0., 0
    for k in range(1, args.k+1, 2):
        topk = torch.topk(distances, dim=1, k=k, largest=False)
        new_labels = torch.cat([labels[range(labels.shape[0]), topk.indices[:,i]].expand(1,-1).T for i in range(topk.indices.shape[1])], dim=1)
        pred = torch.empty_like(test_y)
        for i in range(len(new_labels)):
            x = new_labels[i].unique(return_counts=True)
            pred[i] = x[0][x[1].argmax()]

        acc = (pred == test_y).float().mean().item()
        if acc > best_knn:
            best_knn = acc
            best_k = k

    print(f"best-NN={best_knn*100.:3.2f}%, k={best_k}.")

    # print(f"[TIME]\t[KNN-EVAL-TIME={time.time()-start_time:.2f}]s")

@torch.no_grad()
def load_tensor_single(gpu, model, train_loader, test_loader, k, queue, exit_queue, rep_space):
    torch.cuda.set_device(gpu) # deals with imbalanced gpu usage.
    model = model.cuda()

    if rep_space == 'norm':
        f = F.normalize
    elif rep_space == 'shift-norm':
        f = lambda x: F.normalize(x - x.mean(0))
    else:
        f = lambda x: x

    # test features
    xs, ys = [], []
    for step, (images, labels) in enumerate(test_loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        reps = f(model(images))

        xs.append(reps)
        ys.append(labels)

    test_x, test_y = torch.cat(xs), torch.cat(ys)
    del xs, ys

    # training features
    # instead of storing all the features, we maintain a priority queue.
    ds, ys = None, None
    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        reps = f(model(images))

        # concat and order
        ds = torch.cdist(test_x, reps) if ds is None else torch.cat([ds, torch.cdist(test_x, reps)], dim=1)
        ys = labels.expand(size=[test_x.shape[0], labels.shape[-1]]) if ys is None else torch.cat([ys, labels.expand(size=[test_x.shape[0], labels.shape[-1]])], dim=1)

        # compute local knn to save memory cost
        topk = torch.topk(ds, k=k, dim=1, largest=False)
        ds = topk.values
        new_ys = torch.zeros_like(topk.indices)

        # TODO: can be further optimized？
        for i in range(ys.shape[0]):
            new_ys[i, :] = ys[i][topk.indices[i]]
        ys = new_ys

    # put to the queue, test_y is global
    queue.put((gpu, ys.cpu(), ds.cpu(), test_y.cpu()))

    # manual synchronization
    # there is some issue with queue.get(torch.Tensor) if no synchronization measure is taken.
    while True:
        allow_exit = exit_queue.get()
        if allow_exit != gpu:
            # print(f'proc.{gpu} get key.{allow_exit}, put it back.')
            exit_queue.put(allow_exit)
            time.sleep(1)
        else:
            # print(f'proc.{gpu} get key.{allow_exit}, exiting.')
            break

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    # model
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

    # knn details
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                        help='batch size per node')
    parser.add_argument('--pretrained', default='', type=str, metavar='N',
                        help='checkpoint file path')
    parser.add_argument('--num-workers', default=8, type=int, metavar='N',
                        help='how many sub-processes when loading data')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='split the training features into {gpus} parts.')
    parser.add_argument('-k', '--k', default=5, type=int,
                        help='k neighbors.')
    parser.add_argument('--rep-space', default='original', type=str,
                        help='whether do transformation to the representation space.')

    args = parser.parse_args()

    # load the model
    model = load_model(args.arch, args.pretrained)
    
    # delete the final FC layer
    model.fc = torch.nn.Identity()
    model.eval() # not updating anyway.

    eval_knn(model=model, args=args)
