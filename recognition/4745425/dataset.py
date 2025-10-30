from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from collections import Counter
from sklearn.model_selection import train_test_split

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

def create_dataloaders(img_size: int, batch_size: int, val_frac: float = 0.2):
    # --- transforms ---
    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=10, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=(MEAN,), std=(STD,)),
    ])

    eval_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(int(round(img_size / 0.875))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(MEAN,), std=(STD,)),
    ])

    # --- base dataset (no transform) just to get samples/targets for a stratified split ---
    base_train = datasets.ImageFolder(DATA_ROOT / "train", transform=None)
    targets = np.array(base_train.targets)

    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_frac,
        stratify=targets,
        random_state=42,
    )

    # stratified split
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        n_cls_val = int(round(len(cls_idx) * val_frac))
        val_idx.extend(cls_idx[:n_cls_val].tolist())
        train_idx.extend(cls_idx[n_cls_val:].tolist())

    # --- two separate ImageFolders with their own transforms ---
    train_full = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tfms)
    val_full   = datasets.ImageFolder(DATA_ROOT / "train", transform=eval_tfms)
    test_ds    = datasets.ImageFolder(DATA_ROOT / "test",  transform=eval_tfms)

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(val_full,   val_idx)

    # --- loaders ---
    common_kwargs = dict(num_workers=4,
                         pin_memory=torch.cuda.is_available(),
                         persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **common_kwargs)
    eval_loader  = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **common_kwargs)

    return train_loader, eval_loader, test_loader, train_full.classes

if __name__ == "__main__":
    # calculate_mean_std()
    create_dataloaders(224, 96)