import torch
from model import VAE
from torch import distributions
import matplotlib.pyplot as plt
import numpy as np

model = VAE()

model.load_state_dict(torch.load("model.pt"))

z = distributions.Normal(torch.zeros(64,3), torch.ones(64,3,40,40)).sample()

output, distribution, mean, log_var = model.decoder(z)

for img in output:
    img_np = img.numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()
