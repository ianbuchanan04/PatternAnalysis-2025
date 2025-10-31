import torch
from torch import nn
from train import ConvNeXt, eval_epoch
from dataset import create_dataloaders

device = "cuda"
LABEL_SMOOTH = 0.05
w0 = 10400
w1 = 11120

def main():
    _, _, test_dl, _  = create_dataloaders(224, 64)
    weights = torch.tensor([1.0346, 0.9676], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)
    
    model = ConvNeXt(in_chans=1, num_classes=2, drop_path_rate=0.3).to(device)
    model.head = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        model.head,
    ).to(device)

    # 2. Load the saved weights
    ckpt = torch.load("models/best_model_val20ep.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    best_val_thr = ckpt.get("best_val_thr", 0.5)

    test_loss, acc_argmax, test_auc, f1_default, _, f1_best, acc_best = \
        eval_epoch(model, test_dl, criterion, device, thr=0.85, tta=True)

    # 4. Print results
    print(f"TEST | loss={test_loss:.4f}, acc={acc_argmax:.4f}, acc@best={acc_best:.4f} auc={test_auc:.4f} f1@0.5={f1_default:.4f}, f1@best={f1_default:.4f}, thr={best_val_thr}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()