from torchvision import datasets
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
from model import VAE
from torch.optim import Adam
from model import NICE

train_dataset = datasets.ImageFolder(Path(__file__).parent.parent.parent.parent / "datasets" / "pokemon" / "train", transform=Compose([ToTensor(), Normalize(-0.5, 1)]))

val_dataset = datasets.ImageFolder(Path(__file__).parent.parent.parent.parent / "datasets" / "pokemon" / "train", transform=Compose([ToTensor(), Normalize(-0.5, 1)]))

train_loader = DataLoader(train_dataset, 64, True)

val_loader = DataLoader(val_dataset, 64, True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NICE((3, 40, 40))

model.to(device=device)

optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

def train():
    time_steps = 0
    for i in range(20000):
        for __, (img, label) in enumerate(train_loader):
            time_steps+=1
            img = img.to(device)

            Z, prob = NICE(img)

            loss = -prob

            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if time_steps % 10 == 0:
                print(f"Current time_step: {time_steps}, loss:{loss.item():.3f}")


train()