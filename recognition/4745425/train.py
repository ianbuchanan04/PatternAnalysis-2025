import torch
from torch import nn
import torch.nn.functional as F
from dataset import create_dataloaders
from modules import ConvNeXt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score 
import time
import matplotlib.pyplot as plt

# Config
IMG_SIZE        = 224
BATCH_SIZE      = 64
EPOCHS          = 60 
WARMUP          = 5
LR              = 3e-4
WEIGHT_DECAY    = 1e-2
DROP_PATH_RATE  = 0.3
HEAD_DROPOUT    = 0.5
LABEL_SMOOTH    = 0.05
GRAD_CLIP       = 1.0
EARLY_STOP_PATIENCE = 20

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

@torch.no_grad()
def predict_probs(model, imgs):
    logits = model(imgs)
    probs = F.softmax(logits, dim=1)

    # horizontal flip test-time augmentation
    imgs_flip = torch.flip(imgs, dims=[-1])
    logits_flip = model(imgs_flip)
    probs_flip = F.softmax(logits_flip, dim=1)

    # average original + flipped predictions
    return (probs + probs_flip) / 2.0

def eval_epoch(model, loader, criterion, device, thr=None, test=False):
    """
    Evaluates a model.
    - Uses flip-TTA (predict_probs) for metrics (AUC/F1/thresholded acc).
    - Computes loss on raw logits (no TTA) to match training criterion.
    Returns:
      if not test:
        (avg_loss, acc_argmax, auc, f1_at_0p5_or_thr, best_thr, best_f1, best_acc_at_best_thr)
      else:
        (avg_loss, acc_argmax, auc, f1_at_0p5_or_thr)
    """
    model.eval()
    running_loss, n = 0.0, 0

    all_logits = []
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # loss on logits (no TTA)
            logits = model(imgs)
            loss = criterion(logits, labels)

            # probs for metrics with flip-TTA
            probs_tta = predict_probs(model, imgs)  # shape [B, 2]

            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

            all_logits.append(logits.detach().cpu())
            all_probs.append(probs_tta.detach().cpu())
            all_labels.append(labels.detach().cpu())

    # aggregate
    logits_cat = torch.cat(all_logits, dim=0)
    probs_cat  = torch.cat(all_probs,  dim=0).numpy()
    labels_cat = torch.cat(all_labels, dim=0).numpy()

    # accuracy with argmax (from logits, consistent with training printouts)
    preds_argmax = logits_cat.argmax(dim=1).numpy()
    acc_argmax = accuracy_score(labels_cat, preds_argmax)

    # base metrics from probs
    pos_probs = probs_cat[:, 1]

    # AUC can fail if only one class present; guard it
    try:
        auc = roc_auc_score(labels_cat, pos_probs)
    except ValueError:
        auc = float("nan")

    # F1 at default 0.5 threshold (or at provided thr if given)
    default_thr = 0.5 if thr is None else float(thr)
    bin_preds_default = (pos_probs >= default_thr).astype("int64")
    f1_default = f1_score(labels_cat, bin_preds_default)

    avg_loss = running_loss / max(1, n)

    # If we're doing validation, sweep thresholds to find best F1 (and report its acc)
    if not test:
        best_f1, best_thr, best_acc = -1.0, 0.5, acc_argmax
        # search from 0.05..0.95 inclusive
        for t in [i / 100 for i in range(5, 96)]:
            bp = (pos_probs >= t).astype("int64")
            f1 = f1_score(labels_cat, bp)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = t
                best_acc = accuracy_score(labels_cat, bp)
        return avg_loss, acc_argmax, auc, f1_default, best_thr, best_f1, best_acc
    else:      
        return avg_loss, acc_argmax, auc, f1_default

def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss')
    plt.show()

def main():

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, test_dl, _, eval_dl = create_dataloaders(IMG_SIZE, BATCH_SIZE)

    class_counts = torch.zeros(2, dtype=torch.long)
    for _, labels in train_dl:
        for c in range(2):
            class_counts[c] += (labels == c).sum()

    class_counts = class_counts.to(torch.float32).to(device)
    weights = (class_counts.sum() / (2.0 * class_counts)).clamp(min=1e-8)
    print("weights: ", weights)
    print("created dataloaders")
    # ConvNext Tiny
    model = ConvNeXt(in_chans=1, 
                     num_classes=2, 
                     drop_path_rate=DROP_PATH_RATE,
                     depths=[3, 3, 9, 3]).to(device)
    print("created model")
    model.head = nn.Sequential(nn.Dropout(p=HEAD_DROPOUT), model.head)
    
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    main_epochs = EPOCHS - WARMUP
    sched_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP)
    sched_cosine = CosineAnnealingLR(optimizer, T_max=main_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[WARMUP])    
    
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_path = "best_model.pt"
    best_val_auc = 0.0
    best_epoch = 0
    patience = EARLY_STOP_PATIENCE
    train_losses, val_losses = [], []

    for ep in range(1, EPOCHS + 1):
        start_t = time.perf_counter()

        train_loss, train_acc = train_epoch(
            model, train_dl, criterion, optimizer, scaler, device
        )
        train_losses.append(train_loss)

        val_loss, val_acc, val_auc, val_f1_default, val_thr, val_f1_best, val_acc_best = eval_epoch(
            model, eval_dl, criterion, device
        )
        val_losses.append(val_loss)

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start_t

        scheduler.step()   

        print(
            f"Epoch {ep:02d}/{EPOCHS} "
            f"| train_loss={train_loss:.4f} acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} acc={val_acc:.4f} auc={val_auc:.4f} f1={val_f1_default:.4f} bestF1={val_f1_best:.4f} @thr={val_thr:.2f} bestAcc={val_acc_best:.4f} "
            f"| time={epoch_time:.1f}s "
            f"| lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if ep > 10:
            avg_loss, acc_argmax, auc, f1_default = eval_epoch(
                model, test_dl, criterion, device, test=True, thr=val_thr
            )
            print(f"TEST | thr={val_thr:.2f} loss={avg_loss:.4f} acc={acc_argmax:.4f} auc={auc:.4f} f1={f1_default:.4f}")
            best_epoch = ep
            torch.save(model.state_dict(), best_path)
            if acc_argmax > 0.8:
                model.load_state_dict(torch.load(best_path))
                break

        if ep - best_epoch >= patience:
            print(f"No val_acc improvement in {patience} epochs, stopping.")
            break

    plot_losses(train_losses, val_losses)    

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()