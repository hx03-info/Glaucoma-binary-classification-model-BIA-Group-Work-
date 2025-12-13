# Glaucoma Binary Classification Model (BIA Group Work)

This repository provides a unified framework for binary glaucoma classification
based on fundus images.  
It supports deep learning models (ResNet18, DenseNet, ConvNeXt, MobileNet)
and traditional machine learning models (RF, SVM, XGBoost),
with optional ExpCDR feature integration.

The project is designed for easy local testing via a single CLI script (main.py).

------------------------------------------------------------

PROJECT STRUCTURE

Glaucoma-binary-classification-model-BIA-Group-Work-/
├── ICA/
├── README.md
├── main.py
├── requirements.txt
├── glaucoma.csv
├── Tutorial All.ipynb
│
├── glaucoma-vision/
│   └── glaucoma_vision/
│       ├── glaucoma.csv
│       ├── models/
│       │   ├── evaluate_resnet18.py
│       │   ├── evaluate_densenet.py
│       │   ├── evaluate_convnext.py
│       │   ├── evaluate_mobilenet.py
│       │   ├── evaluate_rf.py
│       │   ├── evaluate_svm.py
│       │   └── evaluate_xgb.py
│       └── utils/
│           └── dl_utils.py

------------------------------------------------------------

ENVIRONMENT SETUP

1. Create and activate a virtual environment (recommended)

conda create -n glaucoma python=3.9
conda activate glaucoma

OR

python -m venv glaucoma_env
source glaucoma_env/bin/activate

------------------------------------------------------------

2. Install dependencies

pip install -r requirements.txt

Note:
- PyTorch 2.8.0 is required
- CPU and Apple Silicon (MPS) are supported
- CUDA is optional

------------------------------------------------------------

MODEL WEIGHTS (IMPORTANT)

Model weights are NOT included in this repository.

You must place trained weights under:

glaucoma-vision/glaucoma_vision/models/weights/

Example:

weights/
├── resnet18.pth
├── densenet.pth
├── convnext.pth
├── mobilenet.pth
├── RF/
├── svm.pkl
└── xgb.json

------------------------------------------------------------

RUNNING EVALUATION (main.py)

Always run main.py from the repository root directory.

------------------------------------------------------------

BASIC USAGE (Deep Learning Models)

python main.py \
  --val_dir "/path/to/Validation" \
  --weights_dir "glaucoma-vision/glaucoma_vision/models/weights" \
  --model resnet18

Supported DL models:
- resnet18
- densenet
- convnext
- mobilenet

------------------------------------------------------------

TRADITIONAL ML MODELS

python main.py \
  --val_dir "/path/to/Validation" \
  --weights_dir "glaucoma-vision/glaucoma_vision/models/weights" \
  --model rf

Supported ML models:
- rf
- svm
- xgb

------------------------------------------------------------

ENABLE ExpCDR INTEGRATION (OPTIONAL)

python main.py \
  --val_dir "/path/to/Validation" \
  --weights_dir "glaucoma-vision/glaucoma_vision/models/weights" \
  --model resnet18 \
  --integrated

The script will automatically use glaucoma.csv under the repository root.

------------------------------------------------------------

OUTPUT

During execution, the script prints:
- Device information
- Dataset statistics
- Classification report
- Summary metrics (Accuracy, AUROC, AUPRC)

Evaluation curves are saved to:

ICA/

------------------------------------------------------------

NOTES

- Always run main.py from the project root
- Tutorial All.ipynb is for reference only
- main.py is the recommended and reproducible entry point

------------------------------------------------------------

ACKNOWLEDGEMENT

This project was developed as part of a BIA Group Coursework
for glaucoma binary classification.
