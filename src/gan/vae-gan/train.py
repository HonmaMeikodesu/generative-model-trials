import time
from torchvision import datasets
from torchvision.utils import save_image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import json
import torch
from torch import distributions
from model import Generator, Descriminator, Encoder
from torch.optim import Adam
from torch.nn.utils import clip_grad
import torch.nn.functional as F
from torch.nn.functional import mse_loss
import sys
sys.path.append(str((Path(__file__).parent.parent.parent).absolute()))
from vae.vanilla.model import image_size


with open(Path(__file__).parent / "hps.json") as hps_fd:
    hps = json.load(hps_fd)

train_dataset = datasets.ImageFolder(Path(__file__).parent.parent.parent.parent / "datasets" / "pokemon" / "train", transform=Compose([ToTensor(), Normalize(-0.5, 1)]))

val_dataset = datasets.ImageFolder(Path(__file__).parent.parent.parent.parent / "datasets" / "pokemon" / "train", transform=Compose([ToTensor(), Normalize(-0.5, 1)]))

train_loader = DataLoader(train_dataset, hps["batch_size"], True)

val_loader = DataLoader(val_dataset, hps["batch_size"], True)

kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
recon_loss = lambda recon_x, x: mse_loss(recon_x, x, size_average=False)

# pixel-wise recon loss and discriminator feature mapping loss
loss_recon_fm = lambda x_r, x_f, fd_r, fd_f: 0.5 * (0.01 * recon_loss(x_f, x_r) + (fd_f - fd_r).pow(2).sum()) / hps["batch_size"]

# E[log(D(G(Z)))]
loss_G = lambda ld_p: F.binary_cross_entropy(ld_p, torch.ones_like(ld_p))

# E[log(D(Y))] + E[log(1 - D(G(Z)))]
# 正样本 + 负样本
loss_D = lambda ld_r, ld_f, ld_p: F.binary_cross_entropy(ld_r, torch.ones_like(ld_r)) + 0.5 * (F.binary_cross_entropy(ld_f, torch.zeros_like(ld_f)) + F.binary_cross_entropy(ld_p, torch.zeros_like(ld_p)))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(image_size[0])

generator = Generator(image_size[0])

discriminator = Descriminator(image_size[0])

encoder.to(device=device)

generator.to(device)

discriminator.to(device)

e_optim = Adam(encoder.parameters(), lr=0.0001, weight_decay=0.01)

g_optim = Adam(generator.parameters(), lr=0.0001, weight_decay=0.01)

d_optim = Adam(discriminator.parameters(), lr=0.0001, weight_decay=0.01)

def train():
    time_steps = 0
    for i in range(hps["epoch"]):
        for __, (img, label) in enumerate(train_loader):
            time_steps+=1
            img = img.to(device)
            distribution, mean, log_var = encoder(img)
            posterior_z = distribution.rsample()
            
            kl = kl_loss(mean, log_var)

            priori_z = distributions.Normal(torch.zeros_like(mean), torch.ones_like(log_var)).sample()

            priori_z = priori_z.detach()

            post_out = generator(posterior_z)

            pri_out = generator(priori_z)

            real_score, __ = discriminator(img)

            post_score, __ = discriminator(post_out)

            pri_score, __ = discriminator(pri_out)

            loss_disc = loss_D(real_score, post_score, pri_score)

            # 先对Discriminator做反向传播，权重更新
            d_optim.zero_grad()
            loss_disc.backward(retain_graph=True)
            clip_grad.clip_grad_norm_(discriminator.parameters(), max_norm=10)
            d_optim.step()

            # 基于更新过权重的Discriminator重新进行一次前向传播，生成新的计算图给Generator做反向传播
            __, real_feature = discriminator(img)

            __, post_feature = discriminator(post_out)

            pri_score, __ = discriminator(pri_out)

            loss_g = loss_G(pri_score)

            loss_r_fm = loss_recon_fm(img, post_out, real_feature, post_feature)

            loss_gen = loss_g + 0.01 * loss_r_fm + kl
            e_optim.zero_grad()
            g_optim.zero_grad()
            loss_gen.backward()
            clip_grad.clip_grad_norm_(encoder.parameters(), max_norm=10)
            clip_grad.clip_grad_norm_(generator.parameters(), max_norm=10)
            e_optim.step()
            g_optim.step()

            if time_steps % 10 == 0:
                print(f"Current time_step: {time_steps}, Generator: total loss: {loss_gen.item():.3f} loss_g: {loss_g.item():.3f}, kl_loss: {kl.item():.3f}, recon_and_feature_mapping_loss: {loss_r_fm.item():.3f}")
                print(f"Current time_step: {time_steps}, Discriminator: total loss: {loss_disc.item():.3f}")

                if time_steps % 200 == 0:
                    with torch.no_grad():
                        save_image(post_out, Path(__file__).parent.parent.parent.parent / "eval" / (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ".png"), format="png")
                        torch.save({
                            "encoder": encoder.state_dict(),
                            "generator": generator.state_dict(),
                            "discriminator": discriminator.state_dict()
                        }, "VAE-GAN.pt")

train()