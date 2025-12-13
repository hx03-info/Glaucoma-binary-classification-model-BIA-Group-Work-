import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
)

def evaluate_densenet(model_path: str, test_dir: str, save_dir: str, img_size: int = 224, show_plots: bool = True):
    """
    Evaluate DenseNet121 model for glaucoma binary classification
    Args:
        model_path: Path to the trained DenseNet model weights
        test_dir: Path to the test dataset directory (contains Glaucoma_Negative/Positive subfolders)
        save_dir: Path to save evaluation plots
        img_size: Image size for preprocessing (default: 224)
        show_plots: Whether to display AUROC/AUPRC plots (default: True)
    Returns:
        dict: Evaluation metrics including accuracy, F1, AUROC, AUPRC and confusion matrix
    """
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # ===================== 1. Device Configuration =====================
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DenseNet Evaluator] Using device: {DEVICE}")

    # ===================== 2. Model Loading =====================
    print(f"[DenseNet Evaluator] Loading model from {model_path}...")
    # Initialize DenseNet121 (match training architecture)
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1)
    )
    
    # Load weights and set to evaluation mode
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("[DenseNet Evaluator] Model loaded successfully.")

    # ===================== 3. Image Preprocessing =====================
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ===================== 4. Inference on Test Set =====================
    y_true = []
    y_prob = []
    class_names = ['Glaucoma_Negative', 'Glaucoma_Positive']
    class_to_idx = {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1}
    
    # Print dataset info (match sample format)
    print(f"Validation Classes Mapping: {class_to_idx}")
    
    print(f"[DenseNet Evaluator] Running inference on validation set...")

    # Iterate through negative/positive folders
    for label, folder_name in enumerate(class_names):
        folder_path = os.path.join(test_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found, skipping...")
            continue
        
        # Iterate through all images in folder
        for img_name in os.listdir(folder_path):
            if not img_name.lower().endswith('.jpg'):
                continue
            
            # Read and preprocess image
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            # Inference
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()  # Probability of positive class
            
            # Collect results
            y_true.append(label)
            y_prob.append(prob)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Print sample count (match sample format)
    print(f"Validation Samples: {len(y_true)}")

    # ===================== 5. Generate Classification Report (match sample format) =====================
    print("\n" + "="*40)
    print("DenseNet Classification Report")
    print("="*40)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=3  # Match sample format
    )
    print(report, end='')  # Remove extra newline

    # ===================== 6. Calculate Summary Metrics (match sample format) =====================
    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    # Print summary metrics (exactly match sample format)
    print("="*40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {auroc:.4f}")
    print(f"AUPRC Score: {auprc:.4f}")
    print("="*40)

    # ===================== 7. Plot AUROC and AUPRC Curves (only two curves) =====================
    if show_plots and len(y_true) > 0:
        # Create 1x2 plot (only ROC + PR curves)
        plt.figure(figsize=(14, 6))

        # --- Left: ROC Curve (match sample style) ---
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # --- Right: Precision-Recall Curve (match sample style) ---
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (PRC)')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'densenet_evaluation_curves.png'), dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        plt.close()

    # ===================== 8. Prepare Metrics Dictionary =====================
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "negative_f1": float(classification_report(y_true, y_pred, target_names=class_names, output_dict=True)['Glaucoma_Negative']['f1-score']),
        "positive_f1": float(classification_report(y_true, y_pred, target_names=class_names, output_dict=True)['Glaucoma_Positive']['f1-score']),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred
    }

    return metrics


def densenet_integrate(MODEL_PATH, VAL_DIR, CSV_PATH, SAVE_DIR):
    """
    封装后的DenseNet混合模型评估函数 (匹配指定输出格式)
    参数:
        MODEL_PATH: 模型权重文件路径
        VAL_DIR: 验证集根目录 (Fundus_Scanes_Sorted/Validation)
        CSV_PATH: 青光眼CSV数据文件路径
        SAVE_DIR: 评估结果保存目录
    返回:
        dict: 评估结果字典
    """
    # Create save directory if not exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Device configuration (match sample format)
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Define class mapping (match sample format)
    class_to_idx = {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1}
    folder_map = {0: 'Glaucoma_Negative', 1: 'Glaucoma_Positive'}
    class_names = ['Glaucoma_Negative', 'Glaucoma_Positive']

    # 1. Dataset Class (simplified)
    class HybridDataset(Dataset):
        def __init__(self, df, val_path, transform=None):
            self.df = df.reset_index(drop=True)
            self.val_path = val_path
            self.transform = transform
            self.folder_map = folder_map
            self.df['eye_code'] = self.df['Eye'].map({'OD': 0, 'OS': 1}).fillna(0)
            self.df['set_code'] = self.df['Set'].map({'A': 0, 'B': 1}).fillna(0)
            self.tabular_cols = ['ExpCDR', 'eye_code', 'set_code']

        def __len__(self): return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            label = row['Glaucoma']
            path = os.path.join(self.val_path, self.folder_map[label], row['Filename'])
            
            # Read image
            if not os.path.exists(path):
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.imread(path)
                if img is None:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                img = self.transform(img)
            
            tab = torch.tensor(row[self.tabular_cols].values.astype(float), dtype=torch.float32)
            return img, tab, torch.tensor(label, dtype=torch.float32)

    # 2. Model Class (simplified)
    class HybridDenseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = models.densenet121(weights=None).features
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.tab_net = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
            self.classifier = nn.Sequential(
                nn.Linear(1024 + 16, 64), 
                nn.ReLU(), 
                nn.Dropout(0.3), 
                nn.Linear(64, 1)
            )

        def forward(self, img, tab):
            x1 = torch.flatten(self.global_pool(self.cnn(img)), 1)
            x2 = self.tab_net(tab)
            return self.classifier(torch.cat((x1, x2), dim=1))

    # ==================== Data Loading ====================
    # Print dataset info (match sample format)
    print(f"Validation Classes Mapping: {class_to_idx}")
    
    # Load and filter CSV data
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['Filename', 'ExpCDR', 'Glaucoma'])
    
    # Filter valid samples
    valid_filenames = []
    for label in [0, 1]:
        folder = os.path.join(VAL_DIR, folder_map[label])
        if os.path.exists(folder):
            valid_filenames.extend([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    df = df[df['Filename'].isin(valid_filenames)].reset_index(drop=True)

    # Data transform
    tfms = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset and loader
    val_dataset = HybridDataset(df, VAL_DIR, tfms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # ==================== Model Loading (match sample format) ====================
    print(f"Loading integrated model from {MODEL_PATH}...")
    model = HybridDenseNet().to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    model.to(DEVICE)
    model.eval()

    # ==================== Inference (match sample format) ====================
    print("Running inference on validation set...")
    y_true, y_prob = [], []

    with torch.no_grad():
        for img, tab, lbl in val_loader:
            img, tab = img.to(DEVICE), tab.to(DEVICE)
            out = model(img, tab)
            y_prob.extend(torch.sigmoid(out).cpu().numpy().flatten())
            y_true.extend(lbl.cpu().numpy().flatten())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Print sample count (match sample format)
    print(f"Validation Samples: {len(y_true)}")

    # ==================== Classification Report (exact match sample format) ====================
    print("\n" + "="*40)
    print("Integrated Model Classification Report")
    print("="*40)
    cls_report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=3
    )
    print(cls_report, end='')  # Remove extra newline

    # ==================== Summary Metrics (exact match sample format) ====================
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_prob)

    # Print summary metrics (exactly match sample)
    print("="*40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {auroc:.4f}")
    print(f"AUPRC Score: {auprc:.4f}")
    print("="*40)

    # ==================== Plot Only Two Curves (ROC + PR) ====================
    if len(y_true) > 0:
        # Create 1x2 plot (only two curves, no confusion matrix/Grad-CAM)
        plt.figure(figsize=(14, 6))

        # ROC Curve (match sample style)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # PR Curve (match sample style)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (PRC)')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        # Save plot (only two curves)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'densenet_integrated_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # ==================== Return Results ====================
    return {
        "class_to_idx": class_to_idx,
        "num_samples": len(y_true),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "negative_f1": float(classification_report(y_true, y_pred, target_names=class_names, output_dict=True)['Glaucoma_Negative']['f1-score']),
        "positive_f1": float(classification_report(y_true, y_pred, target_names=class_names, output_dict=True)['Glaucoma_Positive']['f1-score']),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

# Example usage (exact match sample path)
if __name__ == "__main__":
    # Configuration - match sample path
    SAVE_DIR = '/Users/apple/Desktop/BIA 4/ICA'
    MODEL_PATH_DENSENET = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/densenet.pth'
    MODEL_PATH_INTEGRATED = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/densenet_integrated.pth'
    VAL_DIR = '/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation'
    CSV_PATH = '/Users/apple/Desktop/BIA 4/glaucoma.csv.xls'
    
    # Run DenseNet evaluation
    # densenet_metrics = evaluate_densenet(MODEL_PATH_DENSENET, VAL_DIR, SAVE_DIR)
    
    # Run integrated DenseNet evaluation (matches sample output format)
    eval_results = densenet_integrate(MODEL_PATH_INTEGRATED, VAL_DIR, CSV_PATH, SAVE_DIR)

