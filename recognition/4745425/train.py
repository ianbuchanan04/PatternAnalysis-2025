import torch
from torch import nn
import torch.nn.functional as F
from dataset import create_dataloaders
from modules import ConvNeXt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import roc_auc_score, f1_score

import time

# Config
IMG_SIZE = 224
BATCH_SIZE = 96
WARMUP = 5
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-3
SEED = 42

w0 = 10400
w1 = 11120
total = 0.0

def forward_step(model: nn.Module, imgs, labels, criterion: nn.Module):
    outputs = model(imgs)
    loss = criterion(outputs, labels)
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum()
    return loss, correct, preds, outputs

def train_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip=None):
    model.train()
    running_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = 0

    for imgs, labels in loader:
        # t0 = time.perf_counter()
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            loss, batch_correct, _, _ = forward_step(model, imgs, labels, criterion)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.detach() * imgs.size(0)
        correct += batch_correct
        total += labels.size(0)

        del imgs, labels, loss, batch_correct
        # print(f"{total / BATCH_SIZE}/{21520/BATCH_SIZE} | {time.perf_counter() - t0}s")

    avg_loss = (running_loss / total).item()
    acc = (correct.float() / total).item()
    return avg_loss, acc

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, running_correct, n = 0.0, 0, 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            # standard accuracy
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    avg_loss = running_loss / max(1, n)
    acc = running_correct / max(1, n)

    # ---- AUC & F1 (binary, 2 classes) ----
    auc, f1 = float("nan"), float("nan")
       
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0).numpy()

    probs = F.softmax(logits_cat, dim=1).numpy()
    pos_probs = probs[:, 1]                  # probability of class 1

    bin_preds = (pos_probs >= 0.5).astype("int64")
    auc = roc_auc_score(labels_cat, pos_probs)
    f1  = f1_score(labels_cat, bin_preds)

    return avg_loss, acc, auc, f1

def main():

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, test_dl, classes, eval_dl = create_dataloaders(IMG_SIZE, BATCH_SIZE)
    print("created dataloaders")
    model = ConvNeXt(in_chans=1, num_classes=2, drop_path_rate=0.2).to(device)
    print("created model")

    model.head = nn.Sequential(nn.Dropout(p=0.3), *model.head)
    
    weights = torch.tensor([w0, w1], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    main_epochs = EPOCHS - WARMUP
    sched_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP)
    sched_cosine = CosineAnnealingLR(optimizer, T_max=main_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[WARMUP])    
    
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc, best_path = 0.0, "best_model.pt"
    best_val_auc = 0.0
    total = 0.0
    best_epoch = 0
    for ep in range(1, EPOCHS + 1):
        ep0 = time.perf_counter()
        train_loss, train_acc = train_epoch(model, train_dl, criterion, optimizer, scaler, device, grad_clip=1.0)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        val_loss, val_acc, val_auc, val_f1 = eval_epoch(model, eval_dl, criterion, device="cuda")
        
        scheduler.step()
        epoch_total = time.perf_counter() - ep0
        total += epoch_total
        
        print(f"Epoch {ep:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} auc {val_auc:.3f} f1 {val_f1:.3f} | "
            f"Time taken = {total:.3f} | Learning rate = {optimizer.param_groups[0]['lr']}")

        # Keep your existing "best acc" checkpoint, and optionally add "best AUC"
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = ep
            torch.save(model.state_dict(), best_path)
        elif ep - best_epoch >= 10:
            break
    print(f"Done. Best val acc = {best_acc:.3f}")
    print(f"took {total/60} min | {total/60/ep} mins per epoch")

if __name__ == "__main__":
    # This line is harmless on other OSes and avoids edge-cases on Windows.
    import multiprocessing as mp
    mp.freeze_support()
    main()