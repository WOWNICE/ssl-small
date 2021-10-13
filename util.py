import models
import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def load_model(arch, pretrained, q=True):
    # if q==True: reload encoder q; else: reload encoder k.
    # create model
    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch]()

    # load from pre-trained, before DistributedDataParallel constructor
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location="cpu")

            if q:
                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
            else:
                # rename moco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('module.encoder_k') and not k.startswith('module.encoder_k.fc'):
                        # remove prefix
                        state_dict[k[len("module.encoder_k."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(pretrained))
    
    return model

def distill_ckpt2moco_ckpt(pretrained):
    if pretrained:
        if os.path.isfile(pretrained):
            print("=> loading checkpoint '{}'".format(pretrained))
            checkpoint = torch.load(pretrained, map_location="cpu")
            
            state_dict = checkpoint['state_dict']

            for k in list(state_dict.keys()):
                if k.startswith('module.teacher'):
                    del state_dict[k]
                elif k.startswith('module.student'):
                    state_dict[f"module.encoder_q{k[len('module.student'):]}"] = state_dict[k]
            
            assert list(state_dict.keys()) == list(checkpoint['state_dict'].keys()), 'state dict and checkpoint not match.'

            # no overwrite it yet
            dirs = pretrained.split('/')
            filename = dirs[-1].split('-')[-1]
            dir = '/'.join(dirs[:-1])
            output_file = os.path.join(dir, f'converted_{filename}')
            
            torch.save(checkpoint, output_file)

class PositivePairDataset(Dataset):
    """For loading positive-pair dataset."""

    def __init__(self, root_dir, mode='normal'):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.mode = mode

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
        cache = [f"{self.image_paths[idx]}-{i}.jpeg" for i in range(11)]
        if self.mode == 'all':
            image_names = cache
        elif self.mode == 'normal':
            image_names = cache[:3]
        elif self.mode == 'g-mask':
            image_names = [cache[0], cache[3], cache[4]]
        elif self.mode == 'rand-g-mask':
            image_names = [cache[0], cache[5], cache[6]]
        elif self.mode == 'avg-mask':
            image_names = [cache[0], cache[7], cache[8]]
        elif self.mode == 'rand-avg-mask':
            image_names = [cache[0], cache[9], cache[10]]
        else:
            raise Exception('not supported dataset.')

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



if __name__ == '__main__':
    # model = load_model('mobilenetv3_large', 'checkpoints/checkpoint_0000.pth.tar')
    ckpt_path = './distill_checkpoints'
    for x in os.listdir(ckpt_path):
        if x.startswith('convert'):
            continue
        file_path = os.path.join(ckpt_path, x)
        if os.path.isfile(file_path):
            distill_ckpt2moco_ckpt(file_path)