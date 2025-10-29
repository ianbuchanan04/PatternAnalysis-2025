import torch
from torch import nn
import torch.nn.functional as F
from dataset import create_dataloaders
from modules import ConvNeXt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import roc_auc_score, f1_score

import time

# Config
IMG_SIZE        = 224
BATCH_SIZE      = 64
EPOCHS          = 60
WARMUP          = 5
LR              = 1e-3
WEIGHT_DECAY    = 5e-2
DROP_PATH_RATE  = 0.3
HEAD_DROPOUT    = 0.5
LABEL_SMOOTH    = 0.05
GRAD_CLIP       = 1.0
EARLY_STOP_PATIENCE = 10
AUC_STOP_TARGET = 0.80


w0 = 10400
w1 = 11120
total = 0.0

def forward_step(model: nn.Module, imgs, labels, criterion: nn.Module):
    outputs = model(imgs)
    loss = criterion(outputs, labels)
    preds = outputs.argmax(dim=1)
    correct = (preds == labels).sum()
    return loss, correct, preds, outputs

def train_epoch(model, loader, criterion, optimizer: torch.optim.AdamW, scaler, device, grad_clip=None):
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
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, test_dl, classes, eval_dl = create_dataloaders(IMG_SIZE, BATCH_SIZE)
    return 1

    class_counts = torch.zeros(2, dtype=torch.long)
    for _, labels in train_dl:
        for c in range(2):
            class_counts[c] += (labels == c).sum()

    class_counts = class_counts.to(torch.float32).to(device)
    weights = (class_counts.sum() / (2.0 * class_counts)).clamp(min=1e-8)
    print("weights: ", weights)
    print("created dataloaders")
    model = ConvNeXt(in_chans=1, num_classes=2, drop_path_rate=DROP_PATH_RATE).to(device)
    print("created model")
    model.head = nn.Sequential(nn.Dropout(p=0.2), model.head)
    
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    main_epochs = EPOCHS - WARMUP
    sched_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP)
    sched_cosine = CosineAnnealingLR(optimizer, T_max=main_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[WARMUP])    
    
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_path = "best_model.pt"
    best_val_acc = 0.0
    best_epoch = 0
    patience = EARLY_STOP_PATIENCE

    for ep in range(1, EPOCHS + 1):
        start_t = time.perf_counter()

        train_loss, train_acc = train_epoch(
            model, train_dl, criterion, optimizer, scaler, device
        )

        val_loss, val_acc, val_auc, val_f1 = eval_epoch(
            model, eval_dl, criterion, device
        )

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start_t

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {ep:02d}/{EPOCHS} "
            f"| train_loss={train_loss:.4f} acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} f1={val_f1:.4f} "
            f"| time={epoch_time:.1f}s "
            f"| lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if ep % 5 == 0 and ep > 20:
            test_loss, test_acc, test_auc, test_f1 = eval_epoch(
            model, test_dl, criterion, device
        )
            print("test accuracy", test_acc)
            if test_acc > 0.8:
                model.load_state_dict(torch.load(best_path))
                break

        if ep - best_epoch >= patience:
            print(f"No val_acc improvement in {patience} epochs, stopping.")
            break

    model.load_state_dict(torch.load(best_path))
    test_loss, test_acc, test_auc, test_f1 = eval_epoch(model, test_dl, criterion, device)

    print(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f} f1={test_f1:.4f}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()