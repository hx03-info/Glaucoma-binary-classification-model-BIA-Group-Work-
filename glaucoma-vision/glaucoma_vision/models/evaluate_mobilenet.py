import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, roc_curve, auc, confusion_matrix,
    accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve
)

# ================= Common Utilities =================
def get_device(device: str = None):
    """Auto-detect device (CUDA/CPU)"""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def create_save_dir(save_dir: str):
    """Create save directory with error handling"""
    if save_dir and not os.path.isdir(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create save directory {save_dir}: {e}")

# ================= Image-Only Dataset =================
class GlaucomaImageDataset(Dataset):
    """Dataset for image-only MobileNetV2 (no CSV features)"""
    def __init__(self, val_dir, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        self.class_map = {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1}
        self.image_paths = []
        self.labels = []
        
        # Load images directly from validation directory (no CSV dependency)
        for class_name, label in self.class_map.items():
            class_dir = os.path.join(val_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, filename))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# ================= Hybrid Dataset (Image + CSV) =================
class GlaucomaHybridDataset(Dataset):
    """Dataset for hybrid MobileNetV2 (image + ExpCDR from CSV)"""
    def __init__(self, df, val_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.val_dir = val_dir
        self.transform = transform
        self.folder_map = {0: 'Glaucoma_Negative', 1: 'Glaucoma_Positive'}
        
        # Process tabular features from CSV
        self.df['eye_code'] = self.df['Eye'].map({'OD': 0, 'OS': 1}).fillna(0)
        self.df['set_code'] = self.df['Set'].map({'A': 0, 'B': 1}).fillna(0)
        self.tabular_cols = ['ExpCDR', 'eye_code', 'set_code']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['Glaucoma'])
        filename = row['Filename']
        img_path = os.path.join(self.val_dir, self.folder_map[label], filename)

        try:
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(img)
        
        # Get tabular features (ExpCDR + metadata)
        tab_features = torch.tensor(
            row[self.tabular_cols].values.astype(float), 
            dtype=torch.float32
        )
        return img, tab_features, torch.tensor(label, dtype=torch.float32)

def load_hybrid_validation_data(csv_path, val_dir):
    """Load validation data matching CSV with actual image files"""
    print(f"Scanning Validation Dir: {val_dir} ...")
    df = pd.read_csv(csv_path)
    
    # Get all valid image files from validation directory
    files = set()
    for sub in ['Glaucoma_Positive', 'Glaucoma_Negative']:
        p = os.path.join(val_dir, sub)
        if os.path.exists(p):
            files.update(os.listdir(p))
    
    # Filter CSV to only include existing files
    val_df = df[df['Filename'].isin(files)].copy()
    return val_df

# ================= Model Architectures =================
def build_mobilenet_img_only():
    """Build image-only MobileNetV2 model"""
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 1)
    )
    return model

class HybridMobileNet(nn.Module):
    """Hybrid MobileNetV2 (image + tabular features)"""
    def __init__(self):
        super(HybridMobileNet, self).__init__()
        self.cnn = models.mobilenet_v2(weights=None)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.tab_net = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(1280 + 16, 150),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(150, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, img, tab):
        x1 = self.cnn.features(img)
        x1 = self.global_pool(x1)
        x1 = torch.flatten(x1, 1)
        x2 = self.tab_net(tab)
        return self.classifier(torch.cat((x1, x2), dim=1))

# ================= Image-Only Evaluation (No CSV) =================
def evaluate_mobilenet(
    model_path: str,
    val_dir: str,
    save_dir: str,
    device: str = None
):
    """
    Evaluate MobileNetV2 (image-only) for glaucoma binary classification
    (No CSV dependency - loads images directly from validation directory)
    
    Args:
        model_path: Path to trained MobileNetV2 model (.pth file)
        val_dir: Path to validation dataset directory (contains Glaucoma_Negative/Positive subfolders)
        save_dir: Path to save evaluation plots
        device: Device to use for inference (auto-detect if None)
    
    Returns:
        dict: Evaluation metrics including accuracy, AUROC, AUPRC, F1 scores
    """
    # Create save directory
    create_save_dir(save_dir)
    
    # Device setup
    device = get_device(device)
    print(f"Using device: {device}")
    
    # ================= Data Preparation =================
    # Define transforms (match training)
    tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset (no CSV dependency)
    dataset = GlaucomaImageDataset(val_dir, tfms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Print dataset info (unified format)
    class_to_idx = {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1}
    print(f"Validation Classes Mapping: {class_to_idx}")
    print(f"Validation Samples: {len(dataset)}")
    
    # ================= Model Loading =================
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = build_mobilenet_img_only().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # ================= Inference =================
    print("Running inference on validation set...")
    y_true, y_prob = [], []
    
    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(device)
            out = model(img)
            y_prob.append(torch.sigmoid(out).item())
            y_true.append(lbl.item())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)

    # ================= Evaluation Metrics =================
    # Calculate core metrics
    target_names = ['Glaucoma_Negative', 'Glaucoma_Positive']
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=3,
        output_dict=True
    )
    
    accuracy = float(accuracy_score(y_true, y_pred))
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    negative_f1 = float(report['Glaucoma_Negative']['f1-score'])
    positive_f1 = float(report['Glaucoma_Positive']['f1-score'])

    # Print classification report (unified format)
    print("\n" + "="*40)
    print("MobileNetV2 (Image Only) Classification Report")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3), end='')

    # Print summary metrics (unified format)
    print("="*40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {auroc:.4f}")
    print(f"AUPRC Score: {auprc:.4f}")
    print("="*40)

    # ================= Plot ROC + PR Curves (unified style) =================
    plt.figure(figsize=(14, 6))

    # ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # PR Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = np.sum(y_true == 1) / len(y_true)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {auprc:.3f})')
    plt.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline (AP = {baseline:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PRC)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save plot to specified directory
    if save_dir:
        try:
            plt.savefig(os.path.join(save_dir, 'mobilenet_img_only_curves.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save plot to {save_dir}: {e}")
    plt.show()
    plt.close()

    # ================= Prepare Results =================
    metrics = {
        "class_to_idx": class_to_idx,
        "num_samples": len(y_true),
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "negative_f1": negative_f1,
        "positive_f1": positive_f1,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "used_csv": False  # Mark no CSV usage
    }

    print("\n✅ MobileNetV2 (Image Only) evaluation completed successfully!")
    return metrics

# ================= Hybrid Evaluation (Image + CSV) =================
def mobilenet_integrate(
    model_path: str,
    val_dir: str,
    csv_path: str,
    save_dir: str,
    device: str = None
):
    """
    Evaluate Hybrid MobileNetV2 (image + ExpCDR from CSV) for glaucoma classification
    
    Args:
        model_path: Path to trained Hybrid MobileNetV2 model (.pth file)
        val_dir: Path to validation dataset directory
        csv_path: Path to glaucoma.csv file (contains ExpCDR values)
        save_dir: Path to save evaluation plots
        device: Device to use for inference (auto-detect if None)
    
    Returns:
        dict: Comprehensive evaluation results
    """
    # Create save directory
    create_save_dir(save_dir)
    
    # Device setup
    device = get_device(device)
    print(f"Using device: {device}")
    
    # ================= Data Preparation =================
    # Load and process CSV data
    print(f"Loading CSV metadata from: {csv_path}")
    val_df = load_hybrid_validation_data(csv_path, val_dir)
    
    # Define transforms (match training)
    tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load hybrid dataset (image + tabular features)
    dataset = GlaucomaHybridDataset(val_df, val_dir, tfms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Print dataset info (unified format)
    class_to_idx = {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1}
    print(f"Validation Classes Mapping: {class_to_idx}")
    print(f"Validation Samples: {len(dataset)}")
    
    # ================= Model Loading =================
    print(f"Loading hybrid model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = HybridMobileNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model weights loaded successfully.")

    # ================= Inference =================
    print("Running inference on validation set...")
    y_true, y_prob = [], []
    
    with torch.no_grad():
        for img, tab, lbl in loader:
            img, tab = img.to(device), tab.to(device)
            out = model(img, tab)
            y_prob.append(torch.sigmoid(out).item())
            y_true.append(lbl.item())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)

    # ================= Evaluation Metrics =================
    # Calculate core metrics
    target_names = ['Glaucoma_Negative', 'Glaucoma_Positive']
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=3,
        output_dict=True
    )
    
    accuracy = float(accuracy_score(y_true, y_pred))
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    negative_f1 = float(report['Glaucoma_Negative']['f1-score'])
    positive_f1 = float(report['Glaucoma_Positive']['f1-score'])

    # Print classification report (unified format)
    print("\n" + "="*40)
    print("MobileNetV2 (Hybrid) Classification Report")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3), end='')

    # Print summary metrics (unified format)
    print("="*40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {auroc:.4f}")
    print(f"AUPRC Score: {auprc:.4f}")
    print("="*40)

    # ================= Plot ROC + PR Curves (unified style) =================
    plt.figure(figsize=(14, 6))

    # ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # PR Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = np.sum(y_true == 1) / len(y_true)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {auprc:.3f})')
    plt.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline (AP = {baseline:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PRC)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save plot to specified directory
    if save_dir:
        try:
            plt.savefig(os.path.join(save_dir, 'mobilenet_hybrid_curves.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save plot to {save_dir}: {e}")
    plt.show()
    plt.close()

    # ================= Prepare Results =================
    metrics = {
        "class_to_idx": class_to_idx,
        "num_samples": len(y_true),
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "negative_f1": negative_f1,
        "positive_f1": positive_f1,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "used_csv": True,  # Mark CSV usage
        "feature_names": ['ExpCDR', 'eye_code', 'set_code', 'image_features']
    }

    print("\n✅ MobileNetV2 (Hybrid) evaluation completed successfully!")
    return metrics

# ================= Example Usage =================
if __name__ == "__main__":
    # Configuration parameters (unified path)
    SAVE_DIR = '/Users/apple/Desktop/BIA 4/ICA'  # Save plot directory
    
    # Image-only evaluation (no CSV)
    MODEL_PATH_IMG = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/mobilenet.pth'
    
    
    # Hybrid evaluation (with CSV)
    MODEL_PATH_HYBRID = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/mobilenet_integrated.pth'
    
    
    # Run image-only evaluation
    # img_only_metrics = evaluate_mobilenet(
    #     model_path=MODEL_PATH_IMG,
    #     val_dir=VAL_DIR,
    #     save_dir=SAVE_DIR
    # )
    
    # Run hybrid evaluation (correct parameter order)
    hybrid_metrics = mobilenet_integrate(
        model_path=MODEL_PATH_HYBRID,
        val_dir=VAL_DIR,
        csv_path=CSV_PATH,
        save_dir=SAVE_DIR
    )
    
    # Optional: Access specific metrics
    print(f"\nMobileNetV2 Hybrid AUROC: {hybrid_metrics['auroc']:.4f}")
    print(f"MobileNetV2 Hybrid Accuracy: {hybrid_metrics['accuracy']:.4f}")
