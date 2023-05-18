from torchvision import datasets
from torchvision.utils import save_image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import json
import torch
import torch.nn as nn
from torch import distributions
from model import VAE
from torch.optim import Adam
from torch.nn.utils import clip_grad
import time

with open(Path(__file__).parent / "hps.json") as hps_fd:
    hps = json.load(hps_fd)

train_dataset = datasets.ImageFolder(Path(__file__).parent.parent.parent.parent / "datasets" / "pokemon" / "train", transform=Compose([ToTensor(), Normalize(-0.5, 1)]))

val_dataset = datasets.ImageFolder(Path(__file__).parent.parent.parent.parent / "datasets" / "pokemon" / "train", transform=Compose([ToTensor(), Normalize(-0.5, 1)]))

train_loader = DataLoader(train_dataset, hps["batch_size"], True)

val_loader = DataLoader(val_dataset, hps["batch_size"], True)

recon_loss = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE()

model.to(device=device)

optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

def train():
    time_steps = 0
    for i in range(hps["epoch"]):
        for __, (img, label) in enumerate(train_loader):
            time_steps+=1
            img = img.to(device)
            output, posterior_pdf, mean, log_var = model(img)

            normal_distribution = distributions.Normal(torch.zeros_like(mean).to(device), torch.ones_like(log_var).to(device))

            loss1 = distributions.kl_divergence(
                posterior_pdf,
                normal_distribution
            ).sum(-1).mean().to(device)

            loss2 = recon_loss(output, img).to(device)

            total_loss = loss1 + loss2

            if time_steps % 10 == 0:
                print(f"Current time_step: {time_steps}, kl_loss:{loss1.item():.3f}, recon_loss: {loss2.item():.3f}")

                with torch.no_grad():
                    z = normal_distribution.sample()
                    output = model.decoder(z)
                    save_image(output, Path(__file__).parent.parent.parent.parent / "eval" / (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ".png"), format="png")

            optimizer.zero_grad()

            total_loss.backward()

            clip_grad.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()


train()