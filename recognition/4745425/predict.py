import torch
from torch import nn
from train import ConvNeXt, eval_epoch
from dataset import create_dataloaders

device = "cuda"
LABEL_SMOOTH     = 0.05
w0 = 10400
w1 = 11120

def main():
    _, _, test_dl, _  = create_dataloaders(224, 64)
    w = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    weights = (w.sum() / (2.0 * w)).clamp(min=1e-8)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)
    model = ConvNeXt(in_chans=1, num_classes=2, drop_path_rate=0.3).to(device)
    model.head = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        model.head,
    ).to(device)

    # 2. Load the saved weights
    ckpt = torch.load("best_model_val20.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    best_val_thr = ckpt.get("best_val_thr", 0.5)

    test_loss, acc_argmax, test_auc, f1_default, _, f1_best, acc_best = \
        eval_epoch(model, test_dl, criterion, device, thr=best_val_thr, tta=True)

    # 4. Print results
    print(f"TEST | loss={test_loss:.4f} acc={acc_best:.4f} auc={test_auc:.4f} f1={f1_best:.4f}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()