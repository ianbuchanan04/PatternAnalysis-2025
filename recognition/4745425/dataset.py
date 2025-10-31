from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from collections import Counter
from sklearn.model_selection import train_test_split

DATA_ROOT = Path("ADNI/AD_NC")
MEAN = 0.115892
STD = 0.225568

def count_classes(ds):
    counts = Counter(ds.targets)

    for cls, idx in ds.class_to_idx.items():
        print(f"{cls}: {counts[idx]} images")
    print(ds.classes)
    print(ds.class_to_idx)

def calculate_mean_std():
    dataset = datasets.ImageFolder(
    root=r"ADNI/AD_NC",
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

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from collections import Counter

DATA_ROOT = Path("ADNI/AD_NC")
MEAN = 0.115892
STD = 0.225568

def normalize_per_image(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + 1e-6)

def create_dataloaders(img_size: int, batch_size: int, val_frac: float = 0.2):
    # ---------- 1) define transforms ----------
    train_tfms = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Lambda(normalize_per_image),
    ])

    aug_tfms = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomApply([transforms.RandomRotation(10, fill=0)], p=0.8),
        transforms.ToTensor(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.5),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),
        transforms.Lambda(normalize_per_image),
    ])

    eval_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(int(round(img_size / 0.875))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Lambda(normalize_per_image),
    ])

    # ---------- 2) base dataset (no tfm) just for splitting ----------
    base_train = datasets.ImageFolder(DATA_ROOT / "train", transform=None)
    targets = np.array(base_train.targets)

    # manual stratified split (your style)
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    for cls in np.unique(targets):
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        n_cls_val = int(round(len(cls_idx) * val_frac))
        val_idx.extend(cls_idx[:n_cls_val].tolist())
        train_idx.extend(cls_idx[n_cls_val:].tolist())

    # ---------- 3) build real train/val/test datasets ----------
    # these three MUST see the same folder structure
    train_full = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tfms)
    aug_full   = datasets.ImageFolder(DATA_ROOT / "train", transform=aug_tfms)
    val_full   = datasets.ImageFolder(DATA_ROOT / "train", transform=eval_tfms)
    test_ds    = datasets.ImageFolder(DATA_ROOT / "test",  transform=eval_tfms)

    print("TRAIN:", train_full.class_to_idx)
    print("VAL  :", val_full.class_to_idx)
    print("TEST :", test_ds.class_to_idx)

    # ---------- 4) apply split ----------
    real_train_ds = Subset(train_full, train_idx)
    val_ds        = Subset(val_full,   val_idx)

    # ---------- 5) make the +50% augmented train ----------
    # we ONLY augment from the train indices, not from the whole folder
    n_aug = int(0.5 * len(train_idx))            # 50% of train
    aug_indices = train_idx[:n_aug]              # could also shuffle first
    aug_train_ds = Subset(aug_full, aug_indices)

    # final train = 100% real + 50% aug
    final_train_ds = ConcatDataset([real_train_ds, aug_train_ds])

    # ---------- 6) loaders ----------
    common_kwargs = dict(
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    train_loader = DataLoader(final_train_ds, batch_size=batch_size, shuffle=True, **common_kwargs)
    val_loader   = DataLoader(val_ds,         batch_size=batch_size, shuffle=False, **common_kwargs)
    test_loader  = DataLoader(test_ds,        batch_size=batch_size, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader, train_full.classes

if __name__ == "__main__":
    calculate_mean_std()
    # create_dataloaders(224, 96)