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
import math

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
    
class PositionalEmbedding(nn.Module):
    def __init__(self, channels):
        super(PositionalEmbedding, self).__init__()
        self.channels = channels

    def forward(self, t):
        device = t.device
        half_dim = self.channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SimpleUnet(nn.Module):

    def __init__(self):
        super().__init__()
        img_channels = 3
        time_embed_dim = 32
        out_dim = 3
        down_channels = (32, 64, 128, 256, 512)
        up_channels = (512, 256, 128, 64, 32)  

        self.time_mlp = nn.Sequential(
            PositionalEmbedding(channels=time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(img_channels, down_channels[0], 3, padding=1)

        # Down sample
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1], time_embed_dim)
            for i in range(len(down_channels)-1)
        
        ])  

        # Up sample
        self.up = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], time_embed_dim, up=True)
            for i in range(len(up_channels)-1)
        ])

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, t):
        # Embedd time
        t = self.time_mlp(t)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)
    
model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
print(model)