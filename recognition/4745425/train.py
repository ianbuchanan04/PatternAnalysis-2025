import torch
import json
from torch import nn
import torch.nn.functional as F
from focal_loss.focal_loss import FocalLoss
from dataset import create_dataloaders
from modules import ConvNeXt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, f1_score
import time
import matplotlib.pyplot as plt

# Config
IMG_SIZE        = 224
BATCH_SIZE      = 64
EPOCHS          = 100
WARMUP          = 5
LR              = 3e-4
WEIGHT_DECAY    = 5e-2
DROP_PATH_RATE  = 0.4
HEAD_DROPOUT    = 0.5
LABEL_SMOOTH    = 0.1
GRAD_CLIP       = 2.0
EARLY_STOP_PATIENCE = 8

total = 0.0

history = []

def forward_step(model: nn.Module, imgs, labels, criterion: nn.Module):
    logits = model(imgs)
    probs = F.softmax(logits, dim=1)
    loss = criterion(logits, labels)
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum()
    return loss, correct, preds, probs, logits

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bs = labels.size(0)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum()

        total_loss += loss.item() * bs
        total_correct += correct.item()
        total_samples += bs

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


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

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, thr=None, test=False, tta=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_probs = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # base forward
        logits = model(imgs)
        base_probs = F.softmax(logits, dim=1)
        loss = criterion(logits, labels)

        # optional TTA (horizontal flip like your predict_probs)
        if tta:
            # do the simple 2-view TTA
            flipped = torch.flip(imgs, dims=[3])
            logits_flip = model(flipped)
            probs_flip = F.softmax(logits_flip, dim=1)
            probs_used = 0.5 * (base_probs + probs_flip)
        else:
            probs_used = base_probs

        preds = probs_used.argmax(dim=1)
        correct = (preds == labels).sum()

        bs = labels.size(0)
        total_samples += bs
        total_correct += correct.item()
        total_loss += loss.item() * bs

        all_labels.append(labels.cpu())
        all_probs.append(probs_used.detach().cpu())

    # stack
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_scores = all_probs[:, 1]

    # default argmax metrics
    avg_loss = total_loss / total_samples
    acc_argmax = total_correct / total_samples

    try:
        auc = roc_auc_score(all_labels.numpy(), all_scores.numpy())
    except Exception:
        auc = 0.0

    preds_argmax = all_probs.argmax(dim=1).numpy()
    f1_default = f1_score(all_labels.numpy(), preds_argmax, zero_division=0)

    best_thr = 0.5 if thr is None else thr
    best_f1 = f1_default
    best_acc = acc_argmax

    if thr is None:
        for t in [i / 100 for i in range(5, 96)]:
            bin_preds = (all_scores.numpy() >= t).astype("int32")
            f1_t = f1_score(all_labels.numpy(), bin_preds, zero_division=0)
            acc_t = (bin_preds == all_labels.numpy()).mean()
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thr = t
                best_acc = acc_t
    else:
        bin_preds = (all_scores.numpy() >= thr).astype("int32")
        best_f1 = f1_score(all_labels.numpy(), bin_preds, zero_division=0)
        best_acc = (bin_preds == all_labels.numpy()).mean()
        best_thr = thr

    return (avg_loss, acc_argmax, auc, f1_default, best_thr, best_f1, best_acc,)

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
    torch.manual_seed(2409)
    torch.cuda.manual_seed_all(2409)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl, eval_dl , test_dl, classes = create_dataloaders(IMG_SIZE, BATCH_SIZE)

    class_counts = torch.zeros(2, dtype=torch.long)
    for _, labels in train_dl:
        for c in range(2):
            class_counts[c] += (labels == c).sum()

    print("created dataloaders")
    # ConvNext Tiny
    model = ConvNeXt(in_chans=1, 
                     num_classes=2, 
                     drop_path_rate=DROP_PATH_RATE,
                     depths=[3, 3, 9, 3]).to(device)
    print("created model")
    model.head = nn.Sequential(nn.Dropout(p=HEAD_DROPOUT), model.head)
    
    weights = torch.tensor([1.0346, 0.9676], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    main_epochs = EPOCHS - WARMUP
    sched_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP)
    sched_cosine = CosineAnnealingLR(optimizer, T_max=main_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[WARMUP])
    
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_auc = 0.0
    best_val_thr = 0.5
    best_epoch = 0
    patience = EARLY_STOP_PATIENCE  # keep your const
    best_path = "best_model_val20.pt"

    history = []

    for ep in range(1, EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_dl, criterion, optimizer, device, scaler
        )

        # step schedulers that go every epoch (warmup+cosine)
        if scheduler is not None:
            scheduler.step()

        # VALIDATION (no TTA, thr search)
        (val_loss, val_acc_argmax, val_auc, val_f1_default, val_thr, val_f1_best, val_acc_best) = eval_epoch(model, eval_dl, criterion, device, thr=None, tta=False)

        elapsed = time.time() - start

        print(
            f"Epoch {ep:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc_argmax:.4f} "
            f"auc={val_auc:.4f} f1={val_f1_default:.4f} "
            f"bestF1={val_f1_best:.4f} @thr={val_thr:.2f} | "
            f"time={elapsed:.1f}s | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # log for report
        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc_argmax": val_acc_argmax,
            "val_auc": val_auc,
            "val_f1_default": val_f1_default,
            "val_thr": val_thr,
            "val_f1_best": val_f1_best,
            "val_acc_best": val_acc_best,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # check improvement on VAL
        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_val_thr = val_thr
            best_epoch = ep
            torch.save(model.state_dict(), best_path)
            print(f"->Saved new best to {best_path} (val_auc={val_auc:.4f})")

        # early stop
        if ep - best_epoch >= patience:
            print(f"Early stopping at epoch {ep} (no val improvement for {patience} epochs)")
            break

    # save history
    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Training done. Loading best model and running FINAL TEST...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)

    # FINAL TEST: use best val threshold, and TTA=True if you like
    (test_loss, test_acc_argmax, test_auc, test_f1_default, _, test_f1_best, test_acc_best) = eval_epoch(model, test_dl, criterion, device, thr=best_val_thr, test=True, tta=True)

    print(
        f"[FINAL TEST] loss={test_loss:.4f} | "
        f"acc_argmax={test_acc_argmax:.4f} | "
        f"acc@bestValThr={test_acc_best:.4f} | "
        f"auc={test_auc:.4f} | "
        f"f1_default={test_f1_default:.4f} | "
        f"f1@bestValThr={test_f1_best:.4f} | "
        f"thr_used={best_val_thr:.2f}"
    )

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()