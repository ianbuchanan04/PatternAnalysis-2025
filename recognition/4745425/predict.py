import torch
from torch import nn
from train import ConvNeXt, eval_epoch
from dataset import create_dataloaders

device = "cuda"
LABEL_SMOOTH     = 0.05
w0 = 10400
w1 = 11120

def main():
    _, test_dl, _, _  = create_dataloaders(224, 64)
    w = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    weights = (w.sum() / (2.0 * w)).clamp(min=1e-8)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTH)
    model = ConvNeXt(in_chans=1, num_classes=2, drop_path_rate=0.3).to(device)
    model.head = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        model.head,
    ).to(device)

    # 2. Load the saved weights
    best_path = "best_model_val20.pt"
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)

    # 3. Run evaluation on the test set
    model.eval()
    (test_loss, test_acc, test_auc, test_f1, test_thr, test_f1_best, test_acc_best) = eval_epoch(model, test_dl, criterion, device, thr=0.65)

    # 4. Print results
    print(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f} f1={test_f1:.4f}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()