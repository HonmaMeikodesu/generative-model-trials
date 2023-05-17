from torchvision import datasets
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import json

with open(Path(__file__).parent / "hps.json") as hps_fd:
    hps = json.load(hps_fd)

train_dataset = datasets.ImageFolder(Path(__file__).parent.parent.parent.parent / "datasets" / "pokemon" / "train", transform=Compose([ToTensor(), Normalize(0.5, 1)]))

val_dataset = datasets.ImageFolder(Path(__file__).parent.parent.parent.parent / "datasets" / "pokemon" / "train", transform=Compose([ToTensor(), Normalize(0.5, 1)]))

train_loader = DataLoader(train_dataset, hps.batch_size, True)

val_loader = DataLoader(val_dataset, hps.batch_size, True)