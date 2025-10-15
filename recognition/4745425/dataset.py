from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---- basic setup ----
DATA_ROOT = Path("ADNI/AD_NC")   # change if needed
BATCH_SIZE = 32
IMG_SIZE = 224                    # 224 for ConvNeXt/ResNet; change if you like
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# If your images are grayscale (common in medical), set this True to force 3-channels
GRAYSCALE_TO_RGB = True

# ---- transforms ----
# For pretrained ConvNeXt/ResNet use ImageNet stats:
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)

to_three_channels = (
    [transforms.Grayscale(num_output_channels=3)] if GRAYSCALE_TO_RGB else []
)

train_tfms = transforms.Compose(
    to_three_channels + [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # light augments; tweak as you like
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
)

test_tfms = transforms.Compose(
    to_three_channels + [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ]
)

# ---- datasets ----
train_dir = DATA_ROOT / "train"
test_dir  = DATA_ROOT / "test"

train_ds = datasets.ImageFolder(root=train_dir, transform=train_tfms)
test_ds  = datasets.ImageFolder(root=test_dir,  transform=test_tfms)

# Class names & mapping (handy for metrics/plots)
idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
print("Classes:", train_ds.classes)         # e.g. ['AD', 'NC']
print("Mapping:", train_ds.class_to_idx)    # e.g. {'AD': 0, 'NC': 1}

# ---- loaders ----
def make_loader(ds, shuffle, batch_size=BATCH_SIZE):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(1, os.cpu_count() // 2),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if os.name != "nt" else False,
    )

train_loader = make_loader(train_ds, shuffle=True)
test_loader  = make_loader(test_ds,  shuffle=False)

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    # ... forward pass ...
    break
