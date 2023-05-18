import torch
import torch.nn as nn
from torch import distributions

image_size = (3, 40, 40)

inter_size = 1024

mid_shape = (128, 8, 8)

mid_num = 1

for i in mid_shape:
    mid_num *= i

latent_size = 32

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.Sigmoid()
        )

        self.fc1 = nn.Linear(mid_num, inter_size)

        self.fc2 = nn.Linear(inter_size, latent_size * 2)

    def forward(self, X):
        enc_X = self.image_encoder(X)

        enc_X = enc_X.view(X.shape[0], -1)

        enc_X = self.fc2(self.fc1(enc_X))

        mean, log_var = torch.chunk(enc_X, 2, dim=-1)

        return distributions.Normal(mean, torch.exp(log_var)), mean, log_var

class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()

        self.fcr2 = nn.Linear(latent_size, inter_size)
        self.fcr1 = nn.Linear(inter_size, mid_num)

        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 2),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),
        )

    def forward(self, X):
        dec_X = self.fcr2(X)
        dec_X = self.fcr1(dec_X)
        dec_X = dec_X.view(X.shape[0], *mid_shape)
        return self.image_decoder(dec_X)
    
class VAE(nn.Module):
    def __init__(self) -> None:
        super(VAE, self).__init__()

        self.encoder = Encoder(image_size[0])

        self.decoder = Decoder(image_size[0])
    
    def forward(self, X):
        distribution, mean, log_var = self.encoder(X)

        # (batch_size, latent_size)
        z = distribution.rsample()

        output = self.decoder(z)

        return output, distribution, mean, log_var
