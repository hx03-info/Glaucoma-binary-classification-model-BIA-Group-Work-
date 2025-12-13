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

Always run **`main.py`** from the repository root directory.

### Only-image Evaluation

```bash
python main.py \
  --val_dir "/path/to/Validation" \
  --weights_dir "/path/to/weights" \
  --model resnet18
```

Supported models: **`resnet18`**, **`densenet`**, **`convnext`**, **`mobilenet`**, **`rf`**, **`svm`**, **`xgb`**

### Enable ExpCDR Integration (Optional)

```bash
python main.py \
  --val_dir "/path/to/Validation" \
  --weights_dir "/path/to/weights" \
  --model resnet18 \
  --integrated
```
The script will automatically use **`glaucoma.csv`** in the repository root.

Supported models: **`resnet18`**, **`densenet`**, **`convnext`**, **`mobilenet`**, **`rf`**, **`svm`**, **`xgb`**

## 4️⃣ Outputs

The script prints:

- Device information (CPU / MPS / CUDA)
- Dataset statistics
- Classification report
- Summary metrics: Accuracy, AUROC, AUPRC
Evaluation curves are saved to the **`ICA/ folder.`**




