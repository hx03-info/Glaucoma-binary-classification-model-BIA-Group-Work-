import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (
    classification_report,  
    accuracy_score,         
    roc_curve, auc,         
    average_precision_score,
    precision_recall_curve, 
    confusion_matrix        
)

def evaluate_convnext(model_path: str, val_dir: str, save_dir: str):
    """
    Evaluate ConvNeXt-Tiny model for glaucoma binary classification
    Args:
        model_path: Path to the trained ConvNeXt model weights
        val_dir: Path to the validation dataset directory (contains Glaucoma_Negative/Positive subfolders)
        save_dir: Path to save evaluation plots
    Returns:
        dict: Evaluation metrics including accuracy, AUROC, AUPRC, F1 scores and confusion matrix
    """
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Step 1: Data Loading
    print("[ConvNeXt Evaluator] Loading validation dataset...")
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Validation Classes Mapping: {val_dataset.class_to_idx}")
    print(f"Validation Samples: {len(val_dataset)}")

    # Step 2: Model Loading and Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ConvNeXt Evaluator] Loading model architecture and weights from {model_path}...")

    # Build ConvNeXt-Tiny backbone
    try:
        model = models.convnext_tiny(weights=None)
    except:
        model = models.convnext_tiny(pretrained=False)

    # Modify classifier layer for binary classification (2 classes)
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, 2)

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()  # Switch to evaluation mode
    print("[ConvNeXt Evaluator] Model loaded successfully.")

    # Step 3: Inference on Validation Set
    print("[ConvNeXt Evaluator] Running inference on validation set...")
    y_true = []      # True labels
    y_pred = []      # Predicted classes (0/1)
    y_scores = []    # Probability of positive class (1)

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate probabilities using Softmax
            probs = torch.softmax(outputs, dim=1)
            positive_probs = probs[:, 1].cpu().numpy()
            
            # Get hard classification results
            _, preds = torch.max(outputs, 1)
            
            # Collect results
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(positive_probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Step 4: Calculate Evaluation Metrics
    print("\n" + "="*40)
    print("Classification Report")
    print("="*40)
    target_names = list(val_dataset.class_to_idx.keys())
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Calculate additional metrics
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Step 5: Print Summary Metrics
    print("="*40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {roc_auc:.4f}")
    print(f"AUPRC Score: {pr_auc:.4f}")
    print(f"Confusion Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*40)
    
    # ================= 绘制并保存 AUROC 和 AUPRC 曲线图 =================
    print(f"[ConvNeXt Evaluator] Saving evaluation curves to {save_dir}...")
    
    # 创建图形
    plt.figure(figsize=(14, 6))
    plt.rcParams['font.size'] = 10
    
    # --- 绘制 ROC 曲线 ---
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ConvNeXt ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # --- 绘制 Precision-Recall 曲线 ---
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ConvNeXt Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    curve_path = os.path.join(save_dir, 'convnext_evaluation_curves.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.show()  # 显示图形（可选）
    plt.close()
    
    print(f"[ConvNeXt Evaluator] Curves saved to: {curve_path}")

    # Prepare metrics dictionary for return
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(roc_auc),
        "auprc": float(pr_auc),
        "negative_f1": float(report[target_names[0]]['f1-score']),
        "positive_f1": float(report[target_names[1]]['f1-score']),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "classification_report": report,
        "curve_path": curve_path
    }
    
    return metrics


def convnext_integrate(MODEL_PATH, VAL_DIR, CSV_PATH, SAVE_DIR):
    """
    Evaluate ConvNeXt fusion model (Image + ExpCDR) for glaucoma classification
    
    Args:
        MODEL_PATH: Path to trained model weights
        VAL_DIR: Root directory of validation images (class-separated folders)
        CSV_PATH: Path to CSV file containing ExpCDR values
        SAVE_DIR: Path to save evaluation plots
        
    Returns:
        dict: Evaluation results including metrics and predictions
    """
    # Create save directory if not exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Fixed parameters
    IMG_SIZE = 224

    # Dataset definition
    class FusionDataset(Dataset):
        def __init__(self, root_dir, csv_path, transform=None):
            self.transform = transform
            temp_dataset = datasets.ImageFolder(root=root_dir)
            self.classes = temp_dataset.classes
            self.samples = temp_dataset.samples 
            self.class_to_idx = temp_dataset.class_to_idx
            
            # Load CDR mapping
            df = pd.read_csv(csv_path)
            self.cdr_map = dict(zip(df['Filename'], df['ExpCDR']))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            filename = os.path.basename(path)
            
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
                
            cdr = self.cdr_map.get(filename, 0.5)  # Default CDR value
            cdr_tensor = torch.tensor([cdr], dtype=torch.float32)
            
            return image, cdr_tensor, label

    # Model definition
    class FusionConvNext(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            
            # Build ConvNeXt backbone
            try:
                self.backbone = models.convnext_tiny(weights=None)
            except:
                self.backbone = models.convnext_tiny(pretrained=False)
                
            # Modify classifier head
            n_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()
            
            # Fusion head (image features + CDR)
            self.fusion_head = nn.Sequential(
                nn.Linear(n_features + 1, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

        def forward(self, image, cdr):
            img_feat = self.backbone(image)
            combined_feat = torch.cat((img_feat, cdr), dim=1)
            return self.fusion_head(combined_feat)

    # Data loading
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = FusionDataset(VAL_DIR, CSV_PATH, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Print dataset info
    print(f"Validation Classes Mapping: {val_dataset.class_to_idx}")
    print(f"Validation Samples: {len(val_dataset)}")

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading integrated model from {MODEL_PATH}...")
    model = FusionConvNext(num_classes=2)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    model.to(device)
    model.eval()

    # Inference
    y_true, y_pred, y_scores = [], [], []
    print("Running inference on validation set...")

    with torch.no_grad():
        for images, cdrs, labels in val_loader:
            # Move data to device
            images, cdrs = images.to(device), cdrs.to(device)
            
            # Forward pass
            outputs = model(images, cdrs)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions and probabilities
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs[:, 1].cpu().numpy())

    # Convert to numpy arrays
    y_true, y_pred, y_scores = np.array(y_true), np.array(y_pred), np.array(y_scores)

    # Print classification report
    print("\n" + "="*40)
    print("Integrated Model Classification Report")
    print("="*40)
    target_names = list(val_dataset.class_to_idx.keys())
    cls_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Calculate metrics
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true, y_scores)

    # Plot curves and save to SAVE_DIR
    plt.figure(figsize=(14, 6))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Integrated ROC (Image + ExpCDR)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Integrated Precision-Recall (PRC)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save plot to save directory
    curve_path = os.path.join(SAVE_DIR, 'integrated_model_curves.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Print summary metrics
    print("="*40)
    print(f"Summary Metrics:")
    print(f"AUROC Score: {roc_auc:.4f}")
    print(f"AUPRC Score: {pr_auc:.4f}")
    print("="*40)

    # Return results
    return {
        "class_to_idx": val_dataset.class_to_idx,
        "num_samples": len(val_dataset),
        "classification_report": cls_report,
        "auroc": roc_auc,
        "auprc": pr_auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "curve_path": curve_path
    }

# Example usage
if __name__ == '__main__':
    # Configuration - 使用指定的保存路径
    SAVE_DIR = '/Users/apple/Desktop/BIA 4/ICA'  # 模型评估曲线保存地址
    MODEL_PATH = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/convnext_integrated.pth'
    VAL_DIR = '/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation'
    CSV_PATH = '/Users/apple/Desktop/BIA 4/glaucoma.csv.xls'
    
    # 示例1: 运行集成模型评估 (带SAVE_DIR)
    eval_results = convnext_integrate(MODEL_PATH, VAL_DIR, CSV_PATH, SAVE_DIR)
    
    # 示例2: 运行普通ConvNeXt评估 (带SAVE_DIR)
    CONVNEXT_MODEL_PATH = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/convnext.pth'
    convnext_metrics = evaluate_convnext(CONVNEXT_MODEL_PATH, VAL_DIR, SAVE_DIR)