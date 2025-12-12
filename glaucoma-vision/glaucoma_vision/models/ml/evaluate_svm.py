import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    average_precision_score, 
    confusion_matrix,
    accuracy_score
)
from skimage.io import imread
from skimage.transform import resize


def evaluate_svm(model_path: str, csv_path: str, val_dir: str, show_plots: bool = True):
    """
    Evaluate SVM model for glaucoma binary classification
    Args:
        model_path: Path to the trained SVM model (.pkl file)
        csv_path: Path to glaucoma.csv file (contains ExpCDR values)
        val_dir: Path to validation dataset directory (Glaucoma_Negative/Positive subfolders)
        show_plots: Whether to display AUROC/AUPRC plots (default: True)
    Returns:
        dict: Evaluation metrics including accuracy, AUROC, AUPRC, F1 scores and confusion matrix
    """
    # ================= 1. Data Loading & Feature Extraction =================
    def load_data_from_directory(directory, csv_df):
        """Internal function to load validation data and extract features"""
        features = []
        labels = []
        
        # Map filename to ExpCDR
        cdr_map = dict(zip(csv_df['Filename'], csv_df['ExpCDR']))
        
        class_map = {
            'Glaucoma_Negative': 0, 
            'Glaucoma_Positive': 1
        }
        
        print(f"[SVM Evaluator] Loading validation data from: {directory} ...")
        
        for class_name, label in class_map.items():
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found, skipping...")
                continue
                
            files = os.listdir(class_dir)
            print(f"[SVM Evaluator]   Processing class '{class_name}': {len(files)} images")
            
            for filename in files:
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                # Skip if no ExpCDR in CSV
                if filename not in cdr_map:
                    continue
                    
                cdr = float(cdr_map[filename])
                file_path = os.path.join(class_dir, filename)
                
                try:
                    # Image feature extraction (64x64, match training logic)
                    img = imread(file_path)
                    img = resize(img, (64, 64)) 
                    
                    # Handle grayscale/RGBA images
                    if len(img.shape) == 2:
                        img = np.stack((img,)*3, axis=-1)
                    if img.shape[2] == 4:
                        img = img[:, :, :3]
                    
                    # Calculate statistical features
                    img_mean = np.mean(img, axis=(0, 1)) # [R, G, B]
                    img_std = np.std(img, axis=(0, 1))   # [R, G, B]
                    
                    # Concatenate features: [ExpCDR, R_mean, G_mean, B_mean, R_std, G_std, B_std]
                    feat_vec = np.concatenate(([cdr], img_mean, img_std))
                    
                    features.append(feat_vec)
                    labels.append(label)
                    
                except Exception as e:
                    print(f"[SVM Evaluator] Error processing image {filename}: {e}")
                    
        return np.array(features), np.array(labels)

    # Read CSV file
    print("[SVM Evaluator] Reading CSV file...")
    df = pd.read_csv(csv_path)
    
    # Load validation features and labels
    X_val, y_true = load_data_from_directory(val_dir, df)
    print(f"[SVM Evaluator] Validation Set Shape: {X_val.shape}")

    # ================= 2. Load SVM Model =================
    print(f"[SVM Evaluator] Loading SVM model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    svm_pipeline = joblib.load(model_path)
    print("[SVM Evaluator] Model loaded successfully.")

    # ================= 3. Inference =================
    print("[SVM Evaluator] Running inference...")
    
    # Hard classification results (0/1)
    y_pred = svm_pipeline.predict(X_val)
    
    # Probability/decision scores for positive class (1)
    if hasattr(svm_pipeline, "predict_proba"):
        y_probs = svm_pipeline.predict_proba(X_val)
        y_scores = y_probs[:, 1]  # Positive class probability
    else:
        # Fallback to decision function if no probability estimation
        print("[SVM Evaluator] Warning: Model does not support predict_proba, using decision_function.")
        y_scores = svm_pipeline.decision_function(X_val)
        # Normalize to 0-1 range for visualization
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    # ================= 4. Classification Report =================
    print("\n" + "="*40)
    print("SVM Classification Report")
    print("="*40)
    target_names = ['Glaucoma_Negative', 'Glaucoma_Positive']
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    )
    print(report)

    # ================= 5. Summary Metrics =================
    # Calculate core metrics
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = auc(*roc_curve(y_true, y_scores)[:2])
    pr_auc = average_precision_score(y_true, y_scores)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Print confusion matrix and summary
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("="*40)
    print(f"Summary Metrics (SVM):")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {roc_auc:.4f}")
    print(f"AUPRC Score: {pr_auc:.4f}")
    print(f"Confusion Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*40)

    # ================= 6. Plot AUROC/AUPRC (One row two columns) =================
    if show_plots and len(y_true) > 0:
        # Create one row two columns figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Plot ROC Curve ---
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('SVM: Receiver Operating Characteristic (ROC)')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # --- Plot Precision-Recall Curve ---
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('SVM: Precision-Recall Curve (PRC)')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

    # ================= 7. Prepare Metrics Dictionary =================
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

# Example usage (uncomment to test)
if __name__ == "__main__":
    MODEL_PATH = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/svm.pkl'
    CSV_PATH = '/Users/apple/Desktop/BIA 4/glaucoma.csv.xls'
    VAL_DIR = '/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation'
    
    # Call SVM evaluation function
    metrics = evaluate_svm(MODEL_PATH, CSV_PATH, VAL_DIR)


def svm_integrate(MODEL_PATH, VAL_DIR, CSV_PATH):
    """
    Evaluate SVM integrated model (Image stats + ExpCDR) for glaucoma classification
    
    Args:
        MODEL_PATH: Path to trained SVM model (.pkl file)
        VAL_DIR: Root directory of validation images (class-separated folders)
        CSV_PATH: Path to CSV file containing ExpCDR values
        
    Returns:
        dict: Evaluation results including metrics and predictions
    """
    # ================= 1. Data Loading Function =================
    def load_integrated_data(directory, csv_df):
        """Load validation data: image statistical features + ExpCDR from CSV"""
        features = []
        labels = []
        
        # Create mapping for fast lookup
        cdr_map = dict(zip(csv_df['Filename'], csv_df['ExpCDR']))
        class_map = {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1}
        
        print(f"Loading validation data from: {directory} ...")
        
        count = 0
        for class_name, label in class_map.items():
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir): 
                continue
                
            files = os.listdir(class_dir)
            for filename in files:
                # Skip non-image files
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')): 
                    continue
                
                # Skip if no ExpCDR data (ensure feature alignment)
                if filename not in cdr_map: 
                    continue
                
                try:
                    # 1. Get clinical feature (ExpCDR)
                    cdr = float(cdr_map[filename])
                    
                    # 2. Get image features (64x64 mean and std)
                    file_path = os.path.join(class_dir, filename)
                    img = imread(file_path)
                    img = resize(img, (64, 64))
                    
                    # Ensure RGB format
                    if len(img.shape) == 2: 
                        img = np.stack((img,)*3, axis=-1)
                    if img.shape[2] == 4: 
                        img = img[:, :, :3]
                    
                    # Calculate statistical features
                    means = np.mean(img, axis=(0, 1))  # [R_mean, G_mean, B_mean]
                    stds = np.std(img, axis=(0, 1))    # [R_std, G_std, B_std]
                    
                    # 3. Fusion: [ExpCDR, R_mean, G_mean, B_mean, R_std, G_std, B_std]
                    feat_vec = np.concatenate(([cdr], means, stds))
                    
                    features.append(feat_vec)
                    labels.append(label)
                    count += 1
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
        print(f"Successfully loaded {count} samples.")
        return np.array(features), np.array(labels)

    # ================= 2. Load Data & Model =================
    # Read CSV metadata
    print("Reading CSV metadata...")
    df = pd.read_csv(CSV_PATH)

    # Load validation set
    X_val, y_true = load_integrated_data(VAL_DIR, df)

    # Load SVM model
    print(f"Loading SVM model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Please run svm_integrate.py first.")

    svm_pipeline = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

    # ================= 3. Inference =================
    print("Running inference on validation set...")

    # Get hard predictions (0 or 1)
    y_pred = svm_pipeline.predict(X_val)

    # Get probability predictions (for ROC/PR curves)
    # predict_proba returns [N, 2] array - second column = Positive (1) probability
    y_probs = svm_pipeline.predict_proba(X_val)
    y_scores = y_probs[:, 1] 

    # ================= 4. Evaluation Metrics =================
    print("\n" + "="*40)
    print("Classification Report (SVM Integrated)")
    print("="*40)
    target_names = ['Glaucoma_Negative', 'Glaucoma_Positive']
    cls_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Print classification report and confusion matrix
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # ================= 5. Plot Curves (No Saving) =================
    plt.figure(figsize=(14, 6))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM: Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('SVM: Precision-Recall Curve (PRC)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary metrics
    print("="*40)
    print(f"Summary Metrics:")
    print(f"AUROC Score: {roc_auc:.4f}")
    print(f"AUPRC Score: {pr_auc:.4f}")
    print("="*40)

    # Compile results
    results = {
        "num_samples": len(X_val),
        "classification_report": cls_report,
        "confusion_matrix": cm,
        "auroc": roc_auc,
        "auprc": pr_auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "X_val": X_val  # Optional: include features if needed
    }

    return results

# ================= Example Usage =================
if __name__ == '__main__':
    # Configuration parameters
    MODEL_PATH = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/svm_integrated.pkl'
    VAL_DIR = '/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation'  # Add your VAL_DIR
    CSV_PATH = '/Users/apple/Desktop/BIA 4/glaucoma.csv.xls'  # Add your CSV_PATH
    
    # Run evaluation
    eval_results = svm_integrate(MODEL_PATH, VAL_DIR, CSV_PATH)
    
    # Optional: Access specific metrics
    print(f"\nSVM AUROC: {eval_results['auroc']:.4f}")
    print(f"SVM Accuracy: {eval_results['classification_report']['accuracy']:.4f}")
