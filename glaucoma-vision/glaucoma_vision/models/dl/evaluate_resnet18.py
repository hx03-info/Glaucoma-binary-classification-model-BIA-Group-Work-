import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

def evaluate_resnet18(model_path: str, val_dir: str, img_size: int = 224, show_plots: bool = True):
    """
    Evaluate ResNet-18 model for glaucoma binary classification (no file saving)
    Args:
        model_path: Path to the trained ResNet-18 model weights
        val_dir: Path to the validation dataset directory (contains Glaucoma_Negative/Positive subfolders)
        img_size: Image size for preprocessing (default: 224)
        show_plots: Whether to display AUROC/AUPRC plots (default: True)
    Returns:
        dict: Evaluation metrics including accuracy, AUROC, AUPRC, F1 scores and confusion matrix
    """
    # ===================== 1. Initialization =====================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ResNet-18 Evaluator] Using device: {device}")

    # ===================== 2. Data Loading =====================
    print("[ResNet-18 Evaluator] Loading validation dataset...")
    # Validation transform (match training parameters)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load dataset with ImageFolder
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Validation Classes Mapping: {val_dataset.class_to_idx}")
    print(f"Validation Samples: {len(val_dataset)}")

    # ===================== 3. Model Loading =====================
    print(f"[ResNet-18 Evaluator] Loading model from {model_path}...")
    # Build ResNet-18 backbone (match training architecture)
    model = models.resnet18(pretrained=False)
    
    # Modify fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()  # Switch to evaluation mode
    print("[ResNet-18 Evaluator] Model loaded successfully.")

    # ===================== 4. Inference =====================
    y_true = []      # True labels
    y_pred = []      # Predicted classes (0/1)
    y_scores = []    # Probability of positive class (1)

    print("[ResNet-18 Evaluator] Running inference on validation set...")
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

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # ===================== 5. Classification Report =====================
    print("\n" + "="*40)
    print("ResNet-18 Classification Report")
    print("="*40)
    target_names = list(val_dataset.class_to_idx.keys())
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    )
    print(report)

    # ===================== 6. Summary Metrics =====================
    # Calculate core metrics
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = auc(*roc_curve(y_true, y_scores)[:2])
    pr_auc = average_precision_score(y_true, y_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Print summary metrics
    print("="*40)
    print(f"ResNet-18 Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {roc_auc:.4f}")
    print(f"AUPRC Score: {pr_auc:.4f}")
    print(f"Confusion Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*40)

    # ===================== 7. Plot AUROC/AUPRC (One row two columns) =====================
    if show_plots and len(y_true) > 0:
        # Create one row two columns figure (only display, no save)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Plot ROC Curve ---
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ResNet-18 ROC Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # --- Plot Precision-Recall Curve ---
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('ResNet-18 Precision-Recall Curve')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        # Adjust layout and show plot (no save)
        plt.tight_layout()
        plt.show()

    # ===================== 8. Prepare Metrics Dictionary =====================
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(roc_auc),
        "auprc": float(pr_auc),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "classification_report": classification_report(y_true, y_pred, target_names=target_names, output_dict=True),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores
    }

    return metrics




def resnet18_integrate(MODEL_PATH, VAL_DIR, CSV_PATH):
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
            
            # Load CDR mapping from CSV
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
                
            # Get CDR value (default to 0.5 if not found)
            cdr = self.cdr_map.get(filename, 0.5)
            cdr_tensor = torch.tensor([cdr], dtype=torch.float32)
            
            return image, cdr_tensor, label

    # Model definition
    class FusionResNet18(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            
            # Build ResNet18 backbone (no pretrained weights)
            try:
                self.backbone = models.resnet18(weights=None)
            except:
                self.backbone = models.resnet18(pretrained=False)
                
            # Modify backbone for feature extraction
            n_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
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

    # Data loading pipeline
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader (num_workers=0 to fix multiprocessing error)
    val_dataset = FusionDataset(VAL_DIR, CSV_PATH, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Print dataset information
    print(f"Validation Classes Mapping: {val_dataset.class_to_idx}")
    print(f"Validation Samples: {len(val_dataset)}")

    # Device configuration (MPS for Apple Silicon, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model weights
    print(f"Loading integrated ResNet-18 model from {MODEL_PATH}...")
    model = FusionResNet18(num_classes=2)
    
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise e
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model.to(device)
    model.eval()

    # Inference process
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
    print("Integrated ResNet-18 Classification Report")
    print("="*40)
    target_names = list(val_dataset.class_to_idx.keys())
    cls_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Calculate evaluation metrics
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true, y_scores)

    # Plot curves (no saving)
    plt.figure(figsize=(14, 6))

    # ROC Curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Integrated ResNet-18 ROC (Image + ExpCDR)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Integrated ResNet-18 Precision-Recall')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary metrics
    print("="*40)
    print(f"Summary Metrics (ResNet-18 + ExpCDR):")
    print(f"AUROC Score: {roc_auc:.4f}")
    print(f"AUPRC Score: {pr_auc:.4f}")
    print("="*40)

    # Return comprehensive results
    return {
        "class_to_idx": val_dataset.class_to_idx,
        "num_samples": len(val_dataset),
        "classification_report": cls_report,
        "auroc": roc_auc,
        "auprc": pr_auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores
    }

# Example usage
if __name__ == '__main__':
    # Configuration parameters
    MODEL_PATH = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/resnet18_integrated.pth'
    VAL_DIR = '/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation'  # Add your VAL_DIR path
    CSV_PATH = '/Users/apple/Desktop/BIA 4/glaucoma.csv.xls'  # Add your CSV_PATH path
    
    # Run evaluation
    eval_results = resnet18_integrate(MODEL_PATH, VAL_DIR, CSV_PATH)