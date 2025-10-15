from pathlib import Path
import os, torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = Path("ADNI/AD_NC")
# Precalculated values
MEAN = 0.1155
STD = 0.2254

def calculate_mean_std():
    print("starting")
    dataset = datasets.ImageFolder(
    root=r"ADNI/AD_NC/train",
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    )

    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)

    mean = 0.0
    sq_mean = 0.0
    num_batches = 0

    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.view(imgs.size(0), -1)   # flatten
        mean += imgs.mean(1).sum()
        sq_mean += (imgs ** 2).mean(1).sum()
        num_batches += imgs.size(0)
        if i // 1000 == 0:
            print(f"done :", i)

    mean = mean / num_batches
    std = (sq_mean / num_batches - mean ** 2) ** 0.5

    print(f"Dataset mean: {mean.item():.6f}, std: {std.item():.6f}")

def create_dataloaders(img_size, batch_size):

    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(MEAN,), std=(STD,)),
    ])

    test_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(MEAN,), std=(STD,)),
    ])

    train_ds = datasets.ImageFolder(DATA_ROOT/"train", transform=train_tfms)
    test_ds  = datasets.ImageFolder(DATA_ROOT/"test",  transform=test_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=4,
                            pin_memory=torch.cuda.is_available(),
                            persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4,
                            pin_memory=torch.cuda.is_available(),
                            persistent_workers=True)
    
    return train_loader, test_loader, train_ds.classes

if __name__ == "__main__":
    calculate_mean_std()