# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
from PIL import Image
import random
from torchvision import transforms
import torchvision.transforms.functional as transF
import torch

import math
import numpy as np
import random

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class HaSTransform:
    """apply Hide-and-Seek augmentation scheme to the model."""
    def __init__(self, base_transform, ):
        self.base_transform = base_transform
    
    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)

        # q, k are 224x224 pytorch tensor that are already normalized. 
        return [hide_patch(q), hide_patch(k)]

def hide_patch(img, hide_prob=0.5):
    # get width and height of the image
    # first element is channel.
    c, wd, ht = img.shape

    # possible grid size, 0 means no hiding
    grid_sizes=[0, 14, 28, 56]

    # randomly choose one grid size
    grid_size= random.choice(grid_sizes)
    
    # hide the patches using tensor notation.
    if grid_size == 0:
        return img
    
    # only works for the ideal case
    assert wd % grid_size == 0 and ht % grid_size == 0, 'HaS error: size of patch and shape not match.'

    # get the condensed mask
    sz_mask = (wd//grid_size, ht//grid_size)
    mask = (torch.rand(*sz_mask) > hide_prob).float().view(1,*sz_mask)
    mask = torch.cat([mask for _ in range(c)], axis=0)

    # interpolate is for the sequential data, which scale the dimension other than (B, C).
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0), scale_factor=grid_size).squeeze(0)

    return img * mask

class AugIoUThreshold(object):
    """Outputs two augmented views in the given threshold and not containment """
    def __init__(self, lo=0., hi=1., scale=(0.2, 1.), ratio=(0.75, 1.333333333), max_tolerance=100):
        assert lo < hi, 'lower threshold of iou should be smaller than higher threshold.'
        self.lo, self.hi = lo, hi
        
        self.scale = scale
        self.ratio = ratio
        
        self.max_tolerance = max_tolerance

        self.failed_aug = 0
        self.total_aug = 0

        self.aug_color_flip = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            ##### adapted from moco code
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, image):
        tolerance = 0
        while True:
            tolerance += 1
            crop1 = rand_crop(scale=self.scale, ratio=self.ratio, img_size=image.size)
            crop2 = rand_crop(scale=self.scale, ratio=self.ratio, img_size=image.size)
            iou_val = iou(crop1, crop2)

            # no containment involved. 
            if ((iou_val <= self.hi and iou_val >= self.lo) and not if_contain(crop1, crop2)) or (tolerance > self.max_tolerance):
                if tolerance > self.max_tolerance:
                    self.failed_aug += 1
                self.total_aug += 1

                res1 = self.aug_color_flip(transF.resize(transF.crop(image, *crop1), size=(224, 224)))
                res2 = self.aug_color_flip(transF.resize(transF.crop(image, *crop2), size=(224, 224)))
                return res1, res2 


class MixAug(object):
    def __init__(self, base_transform, lo=0., hi=1., scale=(0.2, 1.), ratio=(0.75, 1.333333333), max_tolerance=100):
        self.iou_aug = AugIoUThreshold(
            lo=lo,
            hi=hi,
            scale=scale,
            ratio=ratio,
            max_tolerance=max_tolerance
        )

        self.moco_aug = TwoCropsTransform(base_transform)

    def __call__(self, x):
        if random.randint(0,1):
            return self.moco_aug(x)
        else:
            return self.iou_aug(x)


class AugDropOverlap(object):
    """Randomly drop some of the overlapped region of two augmented views."""
    def __init__(self, scale=(0.2, 1.), ratio=(0.75, 1.33333333), p=0.5):
        self.aug_color = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        ])

        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image):
        crop1 = rand_crop(scale=self.scale, ratio=self.ratio, img_size=image.size)
        crop2 = rand_crop(scale=self.scale, ratio=self.ratio, img_size=image.size)

        # no random horizotal flip involved 
        res1 = self.aug_color(transF.resized_crop(image, *crop1, size=(224, 224)))
        res2 = self.aug_color(transF.resized_crop(image, *crop2, size=(224, 224)))

        # calculate the overlapped region
        # randomly drop the res1, res2 intersected part with respect to p
        box_a = _to_box(crop1)
        box_b = _to_box(crop2)

        box_c = [
            max(box_a[0], box_b[0]),
            max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]),
            min(box_a[3], box_b[3])
        ]


        mask1 = self.overlap_mask_cal(res1, box_a, box_c)
        mask2 = self.overlap_mask_cal(res2, box_b, box_c)


        # random horizontal flip, change mask region coordinate accordingly.
        if random.randint(0,1):
            res1 = transF.hflip(res1)
            if mask1:
                mask1[1] = res1.size[0] - (mask1[1] + mask1[3])
        if random.randint(0,1):
            res2 = transF.hflip(res2)
            if mask2:
                mask2[1] = res2.size[0] - (mask2[1] + mask2[3])
    
        # convert to tensor
        res1 = self.to_tensor_norm(res1)
        res2 = self.to_tensor_norm(res2)

        # random replace the masked region with zero.
        res1 = self.random_drop(res1, mask1)
        res2 = self.random_drop(res2, mask2)
        
        return res1, res2

    def overlap_mask_cal(self, image, box, box_inter):
        if box_inter[0] >= box_inter[2] or box_inter[1] >= box_inter[3]:
            return

        width, height = image.size

        # ratio of the intersected region. 
        rat_h, rat_w = (box_inter[2] - box_inter[0]+1) / (box[2] - box[0]+1), (box_inter[3] - box_inter[1]+1) / (box[3] - box[1]+1)
        mask_h, mask_w = max(int(rat_h*height), 1), max(int(rat_w*width), 1)

        # calculate the top left corner of the masked area
        top, left = int( (box_inter[0]-box[0]) / (box[2]-box[0]) * height ), int( (box_inter[1]-box[1]) / (box[3]-box[1]) * width )

        return [top, left, mask_h, mask_w]

    def random_drop(self, ts, mask):
        if mask is None:
            return ts
        
        mask_ts = torch.ones_like(ts)

        top, left, height, width = mask
        
        rand_mask = np.random.choice([0, 1], [height, width], p=[self.p, 1-self.p])
        rand_mask = torch.from_numpy(rand_mask).reshape([1, height, width])
        rand_mask = torch.cat([rand_mask, rand_mask, rand_mask], dim=0)
        
        target_shape = mask_ts[:, top:top+height, left:left+width].shape
        if target_shape != rand_mask.shape:
            rand_mask = rand_mask[:, 0:target_shape[1], 0:target_shape[2]]
        mask_ts[:, top:top+height, left:left+width] = rand_mask
        
        # instead of zero masking, which suprisingly introduces a shortcut, try average masking
        res = ts * mask_ts
        avg_mask = (1.-mask_ts) * ts.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True)
        
        return res + avg_mask


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# 
def rand_crop(scale=(0.2, 1.), ratio=(0.75, 1.333333333), img_size=(512, 512)):
    """returns """
    width, height = img_size
    area = height * width

    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(20):
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = torch.exp(
            torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
        ).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()
            return i, j, h, w
    
    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w

def _to_box(a):
    return (a[0], a[1], a[0]+a[2], a[1]+a[3])

def intersect(a, b):
    box_a = _to_box(a)
    box_b = _to_box(b)
    
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    return max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
def iou(a, b):
    inter = intersect(a, b)
    uni = a[2] * a[3] + b[2] * b[3] - inter

    return inter / uni

def if_contain(crop1, crop2):
    box1, box2 = _to_box(crop1), _to_box(crop2)
    if (box1[0] <= box2[0] and box1[1] <= box2[1] and box1[2] >= box2[2] and box1[3] >= box2[3]):
        return True
    if (box2[0] <= box1[0] and box2[1] <= box1[1] and box2[2] >= box1[2] and box2[3] >= box1[3]):
        return True
    return False


# testing functions 
def test_overlap():
    aug = AugDropOverlap(p=0.5)
    im = Image.open('/raid/ssl-positive-eval/auged-train/n01440764/n01440764_44-0.jpeg')

    result = aug(im)
    print(result[0].shape)
    print((result[0].abs() < 1e-6).sum() * 1. / (3*224*224))

    reverse_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1/0.229, 1/0.224, 1/0.225]
        ), 
        transforms.Normalize(
            mean=[-0.485, -0.456, -0.406],
            std=[1., 1., 1.]
        )
    ])

    for ind, ts in enumerate(result):
        img = transF.to_pil_image(reverse_normalize(ts))
        img.save(f"test_{ind+1}.jpeg")

def test_HaS():
    a = torch.rand(3,224,224)
    print(hide_patch(a))

if __name__ == '__main__':
    # test_overlap()
    test_HaS()