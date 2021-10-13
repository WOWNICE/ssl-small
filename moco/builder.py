# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp == 'simple':  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        elif mlp == 'deep':  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.ReLU(), 
                nn.Linear(dim_mlp, dim_mlp), 
                nn.ReLU(), 
                self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), 
                nn.ReLU(), 
                nn.Linear(dim_mlp, dim_mlp), 
                nn.ReLU(), 
                self.encoder_k.fc
            )
        elif mlp == 'dropout':
            dim_mlp = self.encoder_q.fc.weight.shape[1] # set to two times wider than before.
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Dropout(0.5), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Dropout(0.5), self.encoder_k.fc)
        elif mlp == 'wide':  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, 2*dim_mlp), 
                nn.ReLU(), 
                nn.Linear(2*dim_mlp, dim), 
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, 2*dim_mlp), 
                nn.ReLU(), 
                nn.Linear(2*dim_mlp, dim), 
            )
        

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return logits, labels

class AdMoCo(MoCo):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=None, max_iters=5, alpha=0.008, epsilon=0.0157):
        super(AdMoCo, self).__init__(base_encoder, dim=dim, K=K, m=m, T=T, mlp=mlp)
        self.max_iters = max_iters
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, im_q, im_k):
        # return the adversarially attacked images. 
        im_q_adv, im_k_adv = self.attack(im_q, im_k)
        return super(AdMoCo, self).forward(im_q_adv, im_k_adv)

    def attack(self, im_q, im_k):
        # clone the original images
        x1, x2 = im_q.clone(), im_k.clone()
        x1.requires_grad = True
        x2.requires_grad = True

        # fix the encoder network
        self.encoder_q.eval()
        
        with torch.enable_grad():
            for _iter in range(self.max_iters):
                r1 = nn.functional.normalize(self.encoder_q(x1))
                r2 = nn.functional.normalize(self.encoder_k(x2))

                loss = torch.einsum('nc,nc->n', [r1, r2]).unsqueeze(-1).mean()

                grads = torch.autograd.grad(loss, [x1, x2], grad_outputs=None, 
                        only_inputs=True)[0]

                x1.data += self.alpha * torch.sign(grads[0].data) 
                x2.data += self.alpha * torch.sign(grads[1].data) 

                # the adversaries' pixel value should within max_x and min_x due 
                # to the l_infinity / l2 restriction
                def project(x, original_x, epsilon):
                    max_x = original_x + epsilon
                    min_x = original_x - epsilon
                    return torch.max(torch.min(x, max_x), min_x)

                x1 = project(x1, im_q, self.epsilon)
                x2 = project(x2, im_k, self.epsilon)

                # the adversaries' value should be valid pixel value
                def clamp_imagenet(x):
                    mean=[0.485, 0.456, 0.406]
                    std=[0.229, 0.224, 0.225]
                    for i in range(3):
                        min_val = (0-mean[i]) / std[i]
                        max_val = (1-mean[i]) / std[i]
                        x[i].clamp_(min_val, max_val)  

                clamp_imagenet(x1)
                clamp_imagenet(x2)

        self.encoder_q.train()

        return x1, x2

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
