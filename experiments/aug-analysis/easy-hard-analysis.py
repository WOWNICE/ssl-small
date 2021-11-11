import torch
import torch.nn.functional as F
import os
import argparse
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from collections import defaultdict

import json

import moco.loader
import moco.builder

import models
from util import load_model

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
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--ckpt-name', default='resnet50-mocov2-200epoch', type=str,
                    help='the name of the model to be tested.')
parser.add_argument('--gpu', default=0, type=int,
                    help='which gpu to use.')

class PositivePairDataset(Dataset):
    """For loading positive-pair dataset."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir

        # use f"{name}-{i}.jpeg" to get the full name of the image_path for different views, where:
        # i=0: original image
        # i=1: view1
        # i=2: view2
        image_names = set()
        for cls_name in os.listdir(root_dir):
            cls_path = os.path.join(root_dir, cls_name)
            for image_file in os.listdir(cls_path):
                image_name = image_file.split('-')[0]
                image_name = os.path.join(cls_path, image_name)
                image_names.add(image_name)
        
        self.image_paths = sorted(list(image_names))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])

        self.nonaug_transform =  transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_name = self.image_paths[idx].split('/')[-1]
        image_names = [f"{self.image_paths[idx]}-{i}.jpeg" for i in range(3)]

        def greyscale2rgb(img):
            rgbimg = Image.new('RGB', img.size)
            rgbimg.paste(img)
            return rgbimg
        
        # Might come across the greyscale picture. 
        # need to convert the grey scale picture to 3-channel images.
        try: 
            images = [self.nonaug_transform(Image.open(image_names[0]))]
            images.extend([self.transform(Image.open(image_name)) for image_name in image_names[1:]])
        except:
            images = [self.nonaug_transform(greyscale2rgb(Image.open(image_names[0])))]
            images.extend([self.transform(greyscale2rgb(Image.open(image_name))) for image_name in image_names[1:]])
        return images, sample_name

def main():
    args = parser.parse_args()

    # this load_model 
    model = load_model(args.arch, args.pretrained)
    
    # delete the final FC layer
    model.fc = torch.nn.Identity()
    model = model.cuda(args.gpu)
    model.eval() # not updating anyway.

    #===========================================================================================
    # statistics calculation begins from here
    #===========================================================================================
    metric_names = ['cosine_dist', 'l2_dist']
    metric_records = [[], []]
    metric_funcs = [
        lambda x, y: 1. - (F.normalize(x) * F.normalize(y)).sum(dim=1), 
        lambda x, y: (x-y).square().sum(dim=1)
    ]
    sample_names = []

    # data set and loader
    dataset = PositivePairDataset(args.data)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=1, pin_memory=True
    )

    with torch.no_grad():
        for step, ((_, v1, v2), sample_name) in enumerate(train_loader):
            sample_names.extend(sample_name)

            # forward pass of the model.
            v1, v2 = v1.cuda(args.gpu), v2.cuda(args.gpu)
            r1, r2 =model(v1), model(v2)
            
            # benchmark the metric evaluation.
            for i, func in enumerate(metric_funcs):
                metric_records[i].append(func(r1, r2))
            
                # if step >= 5:
                #     break
    
    final_dict = {'model': args.ckpt_name}
    final_dict['samples'] = sample_names
    for i, metric_name in enumerate(metric_names):
        # process the record
        records = metric_records[i] # list of batch-tensors
        records = [eval(str(x)) for x in list(torch.cat(records).cpu().numpy())]  # deal with the problem of float32 not compatible with the json
        final_dict[metric_name] = records

    # dump to json file
    with open('metrics.json', 'w') as outfile:
        json.dump(final_dict, outfile)



if __name__ == '__main__':
    main()


