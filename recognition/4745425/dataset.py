from pathlib import Path
import os, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import Counter


DATA_ROOT = Path("ADNI/AD_NC")
# Precalculated values
MEAN = 0.1155
STD = 0.2254

def count_classes(ds):
    counts = Counter(ds.targets)

    for cls, idx in ds.class_to_idx.items():
        print(f"{cls}: {counts[idx]} images")
    print(ds.classes)
    print(ds.class_to_idx)

def calculate_mean_std():
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
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(MEAN,), std=(STD,)),
        transforms.RandomErasing(p=0.25),
    ])

    test_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(MEAN,), std=(STD,)),
    ])

    train_dss = datasets.ImageFolder(DATA_ROOT/"train", transform=train_tfms)
    count_classes(train_dss)
    exit()
    test_ds  = datasets.ImageFolder(DATA_ROOT/"test",  transform=test_tfms)

    val_frac = 0.1
    n_val = int(len(train_dss) * val_frac)
    n_train = len(train_dss) - n_val
    train_ds, val_ds = random_split(train_dss, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    # Important: give val the deterministic transforms
    val_ds.dataset.transform = test_tfms

    eval_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4,
                            pin_memory=torch.cuda.is_available(),
                            persistent_workers=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=4,
                            pin_memory=torch.cuda.is_available(),
                            persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4,
                            pin_memory=torch.cuda.is_available(),
                            persistent_workers=True)
    
    return train_loader, test_loader, train_dss.classes, eval_loader

if __name__ == "__main__":
    # calculate_mean_std()
    create_dataloaders(224, 96)