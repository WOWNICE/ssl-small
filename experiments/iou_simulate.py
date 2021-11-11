import random
import torch
import math
from collections import defaultdict

from matplotlib import pyplot as plt

def rand_crop(scale=(0.2, 1.), ratio=(0.75, 1.333333333), img_size=(512, 512)):
    width, height = img_size
    area = height * width

    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(10):
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


def iou_simulate(sample_times=10000, img_size=(512, 512)):
    ious = []
    for i in range(sample_times):
        crop1, crop2 = rand_crop(img_size=img_size), rand_crop(img_size=img_size)
        ious.append(iou(crop1, crop2))

    return ious

def three_augs_freq(sample_times=10000, img_size=(512, 512)):
    dic = defaultdict(int)

    for i in range(sample_times):
        crop1, crop2 = rand_crop(img_size=img_size), rand_crop(img_size=img_size)
        box1, box2 = _to_box(crop1), _to_box(crop2)
        
        # equal and contain
        if (box1[0] == box2[0] and box1[1] == box2[1] and box1[2] == box2[2] and box1[3] == box2[3]):
            dic['equal'] += 1
        elif (box1[0] <= box2[0] and box1[1] <= box2[1] and box1[2] >= box2[2] and box1[3] >= box2[3]):
            dic['contain'] += 1
        elif (box2[0] <= box1[0] and box2[1] <= box1[1] and box2[2] >= box1[2] and box2[3] >= box1[3]):
            dic['contain'] += 1
        elif iou(crop1, crop2) < 1e-6:
            dic['adjacent'] += 1
        else:
            iou_val = iou(crop1, crop2)
            if iou_val > 1e-4 and iou_val < 0.05:
                dic['super-hard'] += 1
            elif iou_val < 0.25:
                dic['hard']
            elif iou_val < 0.5:
                dic['semi-hard'] += 1
            elif iou_val < 0.75:
                dic['semi-easy'] += 1
            else:
                dic['easy'] += 1
    
    for k in dic:
        dic[k] = dic[k] / sample_times
    
    return dic


if __name__ == '__main__':
    # ious = iou_simulate(1000000)
    # plt.rcParams['figure.figsize'] = [10, 10]

    # plt.hist(ious, bins=4, range=(0, 1))
    # plt.title('IoU distribution')
    # plt.savefig('iou_distribution.png')
    type_dic = three_augs_freq(sample_times=100000)
    print(type_dic)
    