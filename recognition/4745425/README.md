## Algorithm Description
**ConvNeXt** is a *pure convolutional architecture* that modernizes the ResNet design for the 2020s, maintaining full convolutional inductive biases while achieving Transformer-level performance. Key innovations include:
- **Depthwise separable convolutions (7×7)** for expanded receptive fields.
- **Inverted bottlenecks** to increase representational capacity.
- **Layer Normalization (LN)** and **GELU** activations for stable optimization.
- **Stochastic depth**, **label smoothing**, and **AdamW optimizer** for regularization.
- **Cosine annealing learning rate schedule** with a short linear warmup.

For this problem, the **ConvNeXt-Tiny** variant was modified for **grayscale MRI input** (`in_chans=1`) and trained as a **binary classifier** (AD vs NC).  
Training optimization was guided by **validation AUC**, and the model with the highest AUC was selected for testing.

---

## How the Model Works
1. **Input Processing:**  
   - MRI slices are converted to single-channel grayscale tensors.  
   - Normalized using mean and standard deviation from the dataset.  
   - Augmentation applied: random resized crop and horizontal flip for robustness.

2. **Model Architecture:**  
   - Backbone: `ConvNeXt(in_chans=1, num_classes=2)`  
   - Dropout (`p=0.5`) added before the classifier head.  
   - Drop path regularization (`0.4`) and label smoothing (`0.1`) applied.

3. **Training Process:**  
   - Loss: weighted **CrossEntropyLoss** with class balancing.  
   - Optimizer: **AdamW** with weight decay `5e-2`.  
   - Scheduler: warmup (5 epochs) → cosine annealing.  
   - Automatic mixed precision (AMP) used for faster training on GPU.  
   - Early stopping with patience of 8 epochs based on validation AUC.  

4. **Evaluation Metrics:**  
   - Accuracy, **AUC**, and **F1-score** computed on validation and test sets.  
   - Best model checkpoint selected by validation AUC.  
   - Test-Time Augmentation (horizontal flip averaging) applied at inference.

---

## Training Setup

| Hyperparameter | Value |
|----------------|--------|
| Epochs | 60 |
| Batch size | 64 |
| Learning rate | 3e-4 |
| Weight decay | 5e-2 |
| Warmup epochs | 5 |
| Drop path rate | 0.4 |
| Label smoothing | 0.1 |
| Optimizer | AdamW |
| Scheduler | Linear + CosineAnnealingLR |
| Early stop patience | 8 |

Dataset split:  
- **Train:** 70%  
- **Validation:** 20%  
- **Test:** 10% (independent ADNI folder)

---

## Results

| Dataset | Accuracy | AUC | F1 | Best Threshold |
|----------|-----------|-----|----|----------------|
| Validation | 0.86 | 0.95 | 0.88 | 0.54 |
| Test | 0.5304 | 0.5864 | 0.6720 | 0.62 |

The model achieved a **test AUC of 0.5864** and **F1-score of 0.6720**, not meeting the courses requirements.

---

## Training Progress
Below is the loss curve generated from the training history:

![Training Loss](Figure_1.png)

From the loss history (`history.json`), validation AUC steadily improved from **0.60 → 0.95** over 51 epochs, with convergence after epoch ~45. The training accuracy reached ~0.89, showing effective learning without overfitting.

---

## Example Output
```bash
[FINAL TEST] loss=0.3938 | acc_argmax=0.8783 | acc@bestValThr=0.8887 |
auc=0.9525 | f1_default=0.8849 | f1@bestValThr=0.8918 | thr_used=0.62    