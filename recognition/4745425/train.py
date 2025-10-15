import torch
from torch import nn
from dataset import create_dataloaders
from modules import ConvNeXt
import contextlib

import time

# Config
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 1
LR = 3e-4
SEED = 42

total = 0.0

torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using: {device}")

def run_epoch(model, loader, criterion, optimizer, scaler, training: bool):
    model.train(training)

    # Keep running sums on GPU to avoid per-batch syncs
    running_loss = torch.zeros((), device=device, dtype=torch.float32)
    correct      = torch.zeros((), device=device, dtype=torch.long)
    total = 0
    btotal = 0.0

    LOG_EVERY = 10

    grad_ctx = contextlib.nullcontext() if training else torch.inference_mode()

    for i, (imgs, labels) in enumerate(loader):
        b0 = time.perf_counter()
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with grad_ctx:
            with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        running_loss += loss.detach() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum()
        total += labels.size(0)

        del outputs, loss, preds, imgs, labels

        bt = time.perf_counter() - b0
        btotal += bt
        if (i % LOG_EVERY) == 0:
            print(f"Batch: {i} | Average time per batch = {(btotal / max(1, i)):.3f}")
        

    avg_loss = (running_loss / total).item()
    acc = (correct.float() / total).item()
    print(f"Average time per batch = {(btotal / max(1, i)):.3f}")
    return avg_loss, acc


def main():
    train_dl, test_dl, classes = create_dataloaders(IMG_SIZE, BATCH_SIZE)
    print("created dataloaders")
    model = ConvNeXt(in_chans=1, num_classes=2).to(device)
    print("created model")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc, best_path = 0.0, "best_model.pt"
    for ep in range(1, EPOCHS + 1):
        ep0 = time.perf_counter()
        torch.cuda.memory_summary(device="cuda")
        train_loss, train_acc = run_epoch(model, train_dl, criterion, optimizer, scaler, training=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        val_loss, val_acc     = run_epoch(model, test_dl,  criterion, optimizer, scaler, training=False)
        
        scheduler.step()
        epoch_total = time.perf_counter() - ep0
        total += epoch_total
        
        print(f"Epoch {ep:02d}/{EPOCHS} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
              f"Time taken = {epoch_total:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "classes": classes,
                        "in_channels": 1}, best_path)
            print(f"  ↳ Saved new best to {best_path} (acc={best_acc:.3f})")
    print(f"Done. Best val acc = {best_acc:.3f}")

if __name__ == "__main__":
    # This line is harmless on other OSes and avoids edge-cases on Windows.
    import multiprocessing as mp
    mp.freeze_support()
    main()