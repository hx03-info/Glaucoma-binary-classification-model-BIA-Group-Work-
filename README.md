# Glaucoma Binary Classification Model (BIA Group Work)

This repository provides a framework for binary glaucoma classification using fundus images.  
It supports deep learning models (ResNet18, DenseNet, ConvNeXt, MobileNet) and traditional machine learning models (RF, SVM, XGBoost), with optional ExpCDR feature integration.

All evaluations can be run via the CLI script `main.py`.

---

## 1️⃣ Clone the repository and Environment Setup

### Clone the repository

```bash
git clone https://github.com/hx03-info/Glaucoma-binary-classification-model-BIA-Group-Work-.git
cd Glaucoma-binary-classification-model-BIA-Group-Work-
```

### Create and activate a virtual environment

```bash
# Using conda
conda create -n glaucoma python=3.9
conda activate glaucoma
```


### Install dependency

```bash
pip install -r requirements.txt
```
Note:
- PyTorch 2.8.0 is required
- CPU and Apple Silicon (MPS) are supported
- CUDA is optional

## 2️⃣ Download Models Weights

The folder for model weights is available here: https://drive.google.com/drive/folders/1fHm6mKsTe-DfgcYsIVzJlOzfAhN7mF-J?usp=sharing


## 3️⃣ Running Evaluations

Always run main.py from the repository root directory.

```bash
python main.py \
  --val_dir "/path/to/Validation" \
  --weights_dir "/path/to/weights" \
  --model resnet18
```
Supported models: 
<span style="background-color: #f0f0f0; padding: 4px 8px; border-radius: 12px; font-size: 0.9em; margin-right: 6px;">resnet18</span>,
<span style="background-color: #f0f0f0; padding: 4px 8px; border-radius: 12px; font-size: 0.9em; margin-right: 6px;">densenet</span>,
<span style="background-color: #f0f0f0; padding: 4px 8px; border-radius: 12px; font-size: 0.9em; margin-right: 6px;">convnext</span>,
<span style="background-color: #f0f0f0; padding: 4px 8px; border-radius: 12px; font-size: 0.9em; margin-right: 6px;">mobilenet</span>,
<span style="background-color: #f0f0f0; padding: 4px 8px; border-radius: 12px; font-size: 0.9em; margin-right: 6px;">rf</span>,
<span style="background-color: #f0f0f0; padding: 4px 8px; border-radius: 12px; font-size: 0.9em; margin-right: 6px;">svm</span>,
<span style="background-color: #f0f0f0; padding: 4px 8px; border-radius: 12px; font-size: 0.9em; margin-right: 6px;">xgb</span>
