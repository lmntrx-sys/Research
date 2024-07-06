""" 
Defining a model for DDPM: 

    DDPM: https://arxiv.org/abs/2006.11239

    Model:  The research paper reccommends a U-net model
    First, we will define the encoder block used in the contraction path
    The second part is the decoder block, which takes the feature map from the lower layer, 
    upconverts it, crops and concatenates it with the encoder data of the same level,
    and then performs two 3×3 convolution layers followed by ReLU activation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Block(nn.Module):

    def __init__(self, in_ch, out_ch, time_embed_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_embed_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        # First conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # First, we compute the time embedding 
        t = self.relu(self.time_mlp(t))
        # Add time channel
        t = t.unsqueeze(2).unsqueeze(3)
        h = h + t
        # Second conv
        h = self.relu(self.bnorm1(self.conv2(h)))
        # Downsample
        h = self.pool(h)
        # Transform
        h = self.transform(h)
        return h
        