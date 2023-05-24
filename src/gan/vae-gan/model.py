import sys
from pathlib import Path
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))

from vae.vanilla.model import VAE, Encoder, Decoder, mid_shape, inter_size

mid_num = 1

for i in mid_shape:
    mid_num *= i

import torch
import torch.nn as nn
import torch.functional as F

class Generator(nn.Module):
    def __init__(self, in_channels) -> None:
        super(Generator, self).__init__()
        children = list(Decoder(in_channels).children())
        decoder_children = list(children[-1].children())
        decoder_children.append(nn.Tanh())

        children.pop()

        self.lin = nn.Sequential(*children)

        self.gen = nn.Sequential(*decoder_children)

    def forward(self, X):
        # (batch_size, latent_size)

        # (batch_size, mid_num)
        X =  self.lin(X)
        # (batch_size, *mid_shape)
        X = X.view(X.shape[0], *mid_shape)

        return self.gen(X)
        # (batch_size, in_channels, image_size, image_size)



class Descriminator(nn.Module):
    def __init__(self, in_channels):
        super(Descriminator, self).__init__()
        children = list(list(Encoder(in_channels).children())[0].children())

        children.pop()
        
        self.feature_extractor = nn.Sequential(*children)

        self.judge = nn.Sequential(
            nn.Linear(mid_num, inter_size),
            nn.Linear(inter_size, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        # (batch_size, in_channels, image_size, image_size)

        # (batch_size, *mid_shape)
        X = self.feature_extractor(X)

        X = X.view(X.shape[0], -1)

        # (batch_size, 1)
        score = self.judge(X)

        return score.squeeze(dim=1), X
