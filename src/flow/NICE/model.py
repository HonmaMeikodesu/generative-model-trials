import torch
import torch.nn as nn
from torch import distributions

additive_coupling_law = lambda x2, mx1, invert = False: x2 + mx1 if not invert  else x2 - mx1

class CouplingLayer(nn.Module):
    def __init__(self, image_size, batch_size, mask) -> None:
        super(CouplingLayer, self).__init__()
        self.image_size = image_size
        self.batch_size = batch_size
        data_dim = 1
        for i in image_size:
            data_dim = data_dim * i

        self.data_dim = data_dim

        self.mask = mask

        self.coupling_function = nn.Sequential(
            nn.Conv2d(image_size[0], 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
            nn.ConvTranspose2d(128, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, image_size[0], 4, 2, 2),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),
        )

    def forward(self, X: torch.Tensor, log_jacob_det, invert = False):
        if not invert:

            X1 = X * self.mask
            Y1 = X1

            X2 = X * (1 - self.mask)
            mX1 = self.coupling_function(X1.view(X.shape[0], *self.image_size)).view(X.shape[0], -1)
            Y2 = additive_coupling_law(X2, mX1) * (1 - self.mask)
            return Y1 + Y2 , log_jacob_det
        else:

            Y1 = X * self.mask
            X1 = Y1
            
            Y2 = X * (1 - self.mask)
            mY1 = self.coupling_function(Y1.view(X.shape[0], *self.image_size)).view(X.shape[0], -1)
            X2 = additive_coupling_law(Y2, mY1, True) * (1 - self.mask)
            return X1 + X2, 1

class ScalingLayer(nn.Module):
    def __init__(self, data_dim) -> None:
        super(ScalingLayer, self).__init__()
        self.log_scaling_vector = nn.Parameter(torch.randn(1, data_dim, requires_grad=True))

    def forward(self, X, log_jacob_det, invert = False):
        if not invert:
            return  torch.exp(self.log_scaling_vector) * X, log_jacob_det + torch.sum(self.log_scaling_vector)
        else:
            return torch.exp(-self.log_scaling_vector) * X, log_jacob_det - torch.sum(self.log_scaling_vector)

class NICE(nn.Module):
    def __init__(self, image_size, batch_size, device) -> None:
        super(NICE, self).__init__()

        assert image_size[1] % 2 == 0
        assert image_size[2] % 2 == 0
        data_dim = 1
        for i in image_size:
            data_dim = data_dim * i

        self.data_dim = data_dim
        
        self.device = device

        
        self.coupling_layers = nn.ModuleList([ CouplingLayer(image_size, batch_size, self.get_mask(i, data_dim).to(device)) for i in range(4) ])

        self.scaling_layer = ScalingLayer(data_dim)

    def get_mask(self, index, data_dim):
        mask = torch.ones(1, data_dim).detach()
        if (index % 2 == 0):
            mask[0, 1::2] = 0
        else:
            mask[0, ::2] = 0
        return mask
    
    def forward(self, X, invert = False):
        # (batch_size, in_channels, width, height)

        # (batch_size, data_dim)
        X = X.view(X.shape[0], -1)

        if not invert:
            Z, log_jacob_det = self.f(X)

            prior_distribution = distributions.normal.Normal (torch.zeros(X.shape[0], self.data_dim).to(self.device), torch.ones(X.shape[0], self.data_dim).to(self.device))

            MLE = prior_distribution.log_prob(Z).mean()

            return Z, MLE + log_jacob_det
        else:
            Z = X
            X = self.f_invert(Z)
            return X

    
    def f(self, X):

        Z = X

        log_jacob_det = 0

        for i, flow in enumerate(self.coupling_layers):
            Z, log_jacob_det = flow(Z, log_jacob_det)
        
        Z, log_jacob_det = self.scaling_layer(Z, log_jacob_det)

        return Z, log_jacob_det
    
    def f_invert(self, Z):
        X = Z

        X, log_jacob_det = self.scaling_layer(X, 0)

        for i, flow in enumerate(self.coupling_layers):
            X, log_jacob_det = flow(X, log_jacob_det)
        return X