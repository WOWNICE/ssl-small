# On the Efficacy of Small Self-Supervised Contrastive Models without Distillation Signals

This is the code base for effectively training the [self-supervised small models](https://arxiv.org/abs/2107.14762):
```
@article{shi2021efficacy,
  title	  = {On the Efficacy of Small Self-Supervised Contrastive Models without Distillation Signals},
  author  = {Shi, Haizhou and Zhang, Youcai and Tang, Siliang and Zhu, Wenjie and Li, Yaqian and Guo, Yandong and Zhuang, Yueting},
  journal = {arXiv preprint arXiv:2107.14762},
  year    = {2021}
}
```

## Credit to the previous work
This work is done with the help of the amazing code base of the self-supervised learning method [MoCo](https://github.com/facebookresearch/moco) and the distillation-based method [SEED](https://github.com/jacobswan1/SEED).

## Benchmark models with evaluation metrics 
### Static dataset generation
The static dataset will sample the augmented images from the original imagenet data. This will be used for the fast evaluation on the various metrics.
```
python -m experiments.aug-analysis.sample-augs <path/to/imagenet/train-or-val/folder> <path/to/target/folder>
```

### Instance discrimination accuracy
```
python -m eval.instdisc -a <model-arch> -b <batch-size> -j <job-numbers> --pretrained <path/to/pretrained/model> <path/to/static/dataset> -s <samples-per-class> --gpu 0
```

### Alignment, uniformity, and intra-class alignment
```
python -m eval.stats-dist -a <model-arch> -b <batch-size> -j <job-number> --pretrained <path/to/pretrained/model> <path/to/static/dataset>
```

### best-NN
This is NOT a fassi-gpu implementation for the k-NN evaluation. 
```
python -m eval.knn <path/to/imagenet/folder> -a <model-arch> -b <batch-size> -k <max-kNN> -g <number-of-gpus> --pretrained <path/to/pretrained/model>
```

Specifically in the original paper, we set batch size to 1024, gpu number to 8, and max-kNN to 101 for a fast and fair comparison.
```
python -m eval.knn <path/to/imagenet/folder> -a <model-arch> -b 1024 -k 101 -g 8 --pretrained <path/to/pretrained/model>
```

### Linear evaluation accuracy
Please refer to the original MoCo repo for measuring the linear evaluation accuracy.


### Unsupervised Training
Please refer to the original MoCo repo for training the self-supervised models.


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

