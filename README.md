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
conda create -n glaucoma python=3.9
conda activate glaucoma
```


### Install dependency

```bash
pip install -r requirement.txt
```
Note:
- PyTorch 2.8.0 is required
- CPU and Apple Silicon (MPS) are supported
- CUDA is optional

## 2️⃣ Download Models Weights and Validation Dataset

The folder for model weights is available here: https://drive.google.com/drive/folders/1fHm6mKsTe-DfgcYsIVzJlOzfAhN7mF-J?usp=sharing

The folder of validation dataset is available here: https://drive.google.com/drive/folders/1tbyHar2ekyJpfD4YxICrvmgEJ3DZzT19?usp=sharing


#### Note:
After downloading, use the path to the entire folder as the input to **`--val_dir`** and **`--weights_dir`**. For example, if you downloaded the validation dataset to **`C:/Users/YourName/Validation`**, then **`--val_dir`**  should point to that folder **`"C:/Users/YourName/Validation"`**.


## 3️⃣ Running Evaluations

Always run **`main.py`** from the repository root directory.

### Only-image Evaluation

```bash
python main.py --val_dir "/path/to/Validation" --weights_dir "/path/to/weights" --model resnet18
```

Supported models: **`resnet18`**, **`densenet`**, **`convnext`**, **`mobilenet`**, **`rf`**, **`svm`**, **`xgb`**

### Enable ExpCDR Integration (Optional)

```bash
python main.py --val_dir "/path/to/Validation" --weights_dir "/path/to/weights" --model resnet18 --integrated
```
The script will automatically use **`glaucoma.csv`** in the repository root.

Supported models: **`resnet18`**, **`densenet`**, **`convnext`**, **`mobilenet`**, **`rf`**, **`svm`**, **`xgb`**

### Outputs

The script prints:

- Device information (CPU / MPS / CUDA)
- Dataset statistics
- Classification report
- Summary metrics: Accuracy, AUROC, AUPRC

Evaluation curves are saved to the **`ICA/`** folder.



## 4️⃣ Multi-model Prediction

Run the **`all.py`** from the the repository root directory.

```bash
python all.py --expcdr 0.4803 --image_path "path/to/image" --weights_dir "path/to/weights"
```

For example, **`--image_path`** is the path of image in validation dataset, such as **`"/your path /Validation/Glaucoma_Negative/483.jpg".`**


### Outputs

The scripts prints:

- Model Name
- Probability of positive one (0-1)
- Prediction (negative:<0.35, uncertain:0.35-0.65, positive:>0.65)



