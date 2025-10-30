# Alzheimer's Disease Classification using ConvNeXt

## Overview
This project addresses the problem of Alzheimer’s disease classification using MRI brain images from the ADNI dataset. Alzheimer’s disease (AD) is a progressive neurodegenerative disorder that causes structural brain changes detectable in MRI scans. The goal of this project is to automatically classify brain slices as either Alzheimer’s Disease (AD) or Normal Control (NC) using deep learning. To achieve this, a ConvNeXt-based convolutional neural network was implemented and trained to distinguish subtle structural differences in gray matter patterns that are indicative of Alzheimer’s progression. The algorithm processes grayscale 2D MRI slices, applies normalization and augmentation for robustness, and outputs a probabilistic classification. This contributes to the field of medical imaging by demonstrating the capability of modern convolutional architectures to perform disease detection with high accuracy and efficiency.

## Model
The model is adapted from ConvNeXt, a modern convolutional neural network architecture inspired by Vision Transformers but built entirely from convolutional layers. It modernizes the traditional ResNet design by incorporating several key innovations: depthwise separable convolutions for efficient spatial filtering, inverted bottlenecks to expand feature representations, large 7×7 kernels for improved global context, and Layer Normalization with GELU activations for smoother gradient flow. In this implementation, the ConvNeXt-Tiny variant was adapted for grayscale MRI input (in_chans=1) and trained using the AdamW optimizer with cosine learning rate scheduling, dropout regularization, and label smoothing to prevent overfitting. During evaluation, test-time augmentation via horizontal flips was applied, and metrics such as accuracy, AUC, and F1-score were computed. Overall, the model learns discriminative spatial features that separate Alzheimer’s-affected regions from healthy tissue, enabling reliable classification of brain MRI scans.

Key modifications:
- `in_chans=1` (grayscale input)
- Dropout = 0.5
- Label smoothing = 0.05
- Class balancing weights (computed per dataset)
- AdamW optimizer, cosine annealing LR

## Training Setup
| Hyperparameter | Value |
|----------------|-------|
| Epochs         | 60    |
| Batch size     | 64    |
| Learning rate  | 3e-4  |
| Weight decay   | 1e-2  |
| Warmup epochs  | 5     |
| Drop path rate | 0.3   |

The model was trained on the ADNI dataset split into 60/6/33 for train/val/test. Augmentations included random resized crop and horizontal flips.

## Results
| Dataset    | Accuracy | AUC  | F1   | Best Threshold |
|------------|----------|------|------|----------------|
| Validation | 0.68     | 0.73 | 0.69 | 0.41           |
| Test       | 0.72     | 0.78 | 0.74 | 0.42           |

## Training Loss Throughout Epochs
![Training Loss](Figure_1.png)

## Example Output
```bash
TEST | loss=0.6152 acc=0.7214 auc=0.7832 f1=0.7351
File Structure
bash
Copy code
/modules.py      # ConvNeXt model definition
/dataset.py      # Data loaders for ADNI dataset
/train.py        # Training script
/predict.py      # Evaluation on test set
/README.md       # This report


## Sources
https://arxiv.org/pdf/2201.03545
https://github.com/facebookresearch/ConvNeXt