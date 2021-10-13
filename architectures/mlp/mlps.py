import torch
from torch import nn


mlp_simple = nn.Sequential(
    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
)