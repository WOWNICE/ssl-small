import torch
import os
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import random 
import math
import numpy as np
from tqdm import tqdm

import torchvision.transforms.functional as transF

import moco.loader
import moco.builder

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to the dataset to be sampled on.')
parser.add_argument('savedir', metavar='DIR',
                    help='path where to save the auged pics.')
parser.add_argument('--aug', default='moco', type=str,
                    help='which augmentatioin to use.')
parser.add_argument('--images-per-class', default=50, type=int,
                    help='Images per class to be sampled.')


# different augmentation methods 
aug_color_flip = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    ##### adapted from moco code
    # transforms.ToTensor(),
    # normalize
])

aug_color = transforms.Compose([
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
])

class AugMoCo(object):
    def __init__(self):
        self.aug_shape = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.2, 1.))])
    def __call__(self, image):
        return aug_color_flip(self.aug_shape(image)), aug_color_flip(self.aug_shape(image))


class AugNoIntersection(object):
    def __init__(self):
        self.aug_shape = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.4, 1.))])
    def __call__(self, image):
        # sample two image regions that are not intersected with each other 
        width, height = image.size
        five_crop = transforms.Compose([transforms.FiveCrop((height//2, width//2))])
        four_crops = five_crop(image)[:-1]
        
        inds = list(range(4))
        random.shuffle(inds)

        res1, res2 = four_crops[inds[0]], four_crops[inds[1]]
        return aug_color_flip(self.aug_shape(res1)), aug_color_flip(self.aug_shape(res2))  


class AugIoUThreshold(object):
    """Outputs two augmented views in the given threshold and not containment """
    def __init__(self, lo=0., hi=1., scale=(0.2, 1.), ratio=(0.75, 1.333333333), max_tolerance=1000):
        assert lo < hi, 'lower threshold of iou should be smaller than higher threshold.'
        self.lo, self.hi = lo, hi
        
        self.scale = scale
        self.ratio = ratio
        
        self.max_tolerance = max_tolerance

        self.failed_aug = 0
        self.total_aug = 0

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

                res1 = aug_color_flip(transF.resize(transF.crop(image, *crop1), size=(224, 224)))
                res2 = aug_color_flip(transF.resize(transF.crop(image, *crop2), size=(224, 224)))
                return res1, res2, crop1, crop2, iou_val


class AugIoUThresholdGaussianMask(object):
    def __init__(self, lo=0., hi=1., scale=(0.2, 1.), ratio=(0.75, 1.333333333), max_tolerance=1000):
        assert lo < hi, 'lower threshold of iou should be smaller than higher threshold.'
        self.lo, self.hi = lo, hi
        
        self.scale = scale
        self.ratio = ratio
        
        self.max_tolerance = max_tolerance

        self.failed_aug = 0
        self.total_aug = 0
    
    def __call__(self, image):
        tolerance = 0
        while True:
            tolerance += 1
            crop1 = rand_crop(scale=self.scale, ratio=self.ratio, img_size=image.size)
            crop2 = rand_crop(scale=self.scale, ratio=self.ratio, img_size=image.size)
            iou_val = iou(crop1, crop2)

            # no containment involved.
            # in the threshold range. 
            if ((iou_val <= self.hi and iou_val >= self.lo) and not if_contain(crop1, crop2)) or (tolerance > self.max_tolerance):
                if tolerance > self.max_tolerance:
                    self.failed_aug += 1
                self.total_aug += 1
                
                # no random horizotal flip involved 
                res1 = aug_color(transF.resized_crop(image, *crop1, size=(224, 224)))
                res2 = aug_color(transF.resized_crop(image, *crop2, size=(224, 224)))

                # replace the res1, res2 intersected part to independent gaussian noise 
                box_a = _to_box(crop1)
                box_b = _to_box(crop2)

                box_c = [
                    max(box_a[0], box_b[0]),
                    max(box_a[1], box_b[1]),
                    min(box_a[2], box_b[2]),
                    min(box_a[3], box_b[3])
                ]

                # gaussian masked
                masked1, mask_bs1 = gaussian_mask(res1, box_a, box_c)
                masked2, mask_bs2 = gaussian_mask(res2, box_b, box_c)

                # average masked
                avg1, avg_bs1 = average_mask(res1, box_a, box_c)
                avg2, avg_bs2 = average_mask(res2, box_b, box_c)

                # manually horizontal flip
                if random.randint(0,1):
                    res1, masked1, mask_bs1, avg1, avg_bs1 = transF.hflip(res1), transF.hflip(masked1), transF.hflip(mask_bs1), transF.hflip(avg1), transF.hflip(avg_bs1)
                if random.randint(0,1):
                    res2, masked2, mask_bs2, avg2, avg_bs2 = transF.hflip(res2), transF.hflip(masked2), transF.hflip(mask_bs2), transF.hflip(avg2), transF.hflip(avg_bs2)

                return res1, res2, masked1, masked2, mask_bs1, mask_bs2, avg1, avg2, avg_bs1, avg_bs2, iou_val


def main():
    args = parser.parse_args()

    if args.aug == 'moco':
        augmentation = AugMoCo()
    elif args.aug == 'nointer':
        augmentation = AugNoIntersection()
    elif args.aug == 'super-hard':
        augmentation = AugIoUThresholdGaussianMask(lo=1e-4, hi=0.05, max_tolerance=2000)
    elif args.aug == 'hard':
        augmentation = AugIoUThresholdGaussianMask(lo=0.05, hi=0.25)
    elif args.aug == 'easy':
        augmentation = AugIoUThresholdGaussianMask(lo=0.25, hi=1.)

    class_lst = os.listdir(args.data)
    existing_class = os.listdir(args.savedir)

    # mkdir classes 
    data_class_paths, target_class_paths = [], []
    for class_name in class_lst:
        class_path = os.path.join(args.savedir, class_name)
        if class_name not in existing_class:
            os.mkdir(class_path)
        target_class_paths.append(class_path)

        class_path = os.path.join(args.data, class_name)
        data_class_paths.append(class_path)
    
    # sample different images in different classes
    # print(data_class_paths)
    for cls_ind, data_class_path in enumerate(tqdm(data_class_paths)):
        image_files = os.listdir(data_class_path)
        image_paths = [os.path.join(data_class_path, x) for x in image_files]
        image_names = [x.split('.')[0] for x in image_files]
        
        # images per class
        for i, image_path in enumerate(image_paths[:args.images_per_class]):
            image = Image.open(image_path)
            image_save_path = f"{os.path.join(target_class_paths[cls_ind], image_names[i])}"
            image.save(f"{image_save_path}-0.jpeg")

            auged_images = augmentation(image)
            if args.aug in ['moco', 'nointer', 'contain']:
                im_number = 2
            else: 
                im_number = 10
            for view in range(im_number):
                auged_images[view].save(f"{image_save_path}-{view+1}.jpeg")

#==============================utils================================
# These utils are not used in the paper.
# Feel free to take a look. 
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

#==============Gaussian Masking=================
def get_white_noise_image(h, w):
    pil_map = Image.fromarray(np.random.randint(0,255,(h, w, 3),dtype=np.dtype('uint8')))
    return pil_map

def gaussian_mask(image, box, box_inter):
    width, height = image.size
    # ratio of the intersected region. 
    rat_h, rat_w = (box_inter[2] - box_inter[0]+1) / (box[2] - box[0]+1), (box_inter[3] - box_inter[1]+1) / (box[3] - box[1]+1)
    mask_h, mask_w = max(int(rat_h*height), 1), max(int(rat_w*width), 1)

    # create white noise PIL Image object
    white_noise = get_white_noise_image(h=mask_h, w=mask_w)

    # calculate the top left corner of the masked area
    top, left = int( (box_inter[0]-box[0]) / (box[2]-box[0]) * height ), int( (box_inter[1]-box[1]) / (box[3]-box[1]) * width )

    pasted = image.copy()
    pasted.paste(white_noise, (left, top)) # height and width exchanged. 

    # baseline masked sample: randomly pick one location and mask: 
    top = random.randint(0, height-mask_h)
    left = random.randint(0, width-mask_w)
    baseline = image.copy()
    baseline.paste(white_noise, (left, top))

    return pasted, baseline


#==============Average Masking==================
def get_zero_image(h, w):
    rgb = np.array([124, 116, 104]).reshape([1, 1, 3])
    val = np.zeros([h, w, 3]) + rgb
    pil_map = Image.fromarray(val.astype('uint8'))

    return pil_map

def average_mask(image, box, box_inter):
    width, height = image.size
    print(box, box_inter)
    # ratio of the intersected region. 
    rat_h, rat_w = (box_inter[2] - box_inter[0]+1) / (box[2] - box[0]+1), (box_inter[3] - box_inter[1]+1) / (box[3] - box[1]+1)
    mask_h, mask_w = max(int(rat_h*height), 1), max(int(rat_w*width), 1)

    white_noise = get_zero_image(h=mask_h, w=mask_w)

    # calculate the top left corner of the masked area
    top, left = int( (box_inter[0]-box[0]) / (box[2]-box[0]) * height ), int( (box_inter[1]-box[1]) / (box[3]-box[1]) * width )

    pasted = image.copy()
    pasted.paste(white_noise, (left, top)) # height and width exchanged. 

    # baseline masked sample: randomly pick one location and mask: 
    top = random.randint(0, height-mask_h)
    left = random.randint(0, width-mask_w)
    baseline = image.copy()
    baseline.paste(white_noise, (left, top))

    return pasted, baseline

if __name__ == '__main__':
    main()
    # crop = AugIoUThresholdGaussianMask(lo=1e-4, hi=0.05)
    # im = Image.open('/raid/ssl-positive-eval/auged-train/n01440764/n01440764_44-0.jpeg')
    # # im = transF.resize(im, size=(224, 224))

    # result = crop(im)
    # # print(crop1, crop2, iou_val)
    # for ind, i in enumerate(result[:-1]):
    #     i.save(f"test_{ind+1}.jpeg")
        
    
    # im = Image.open('/home/luodongliang/Projects/ssl-small-main/experiments/aug-analysis/auged-train/n01440764/n01440764_44-0.jpeg')
    # im = transF.resize(im, size=(224, 224))
    # # crop1, crop2 = rand_crop(scale=(0.2, 0.8), img_size=im.size), rand_crop(scale=(0.2, 0.8), img_size=im.size)


    # mask = get_white_noise_image(h=100, w=10)
    # print(mask.size)
    # im.paste(mask, (0, 100, 10, 200))
    # im.save('test_pos.png')

    # out = im.transpose(Image.FLIP_LEFT_RIGHT)
    # out.save('test_pos_flipped.png')