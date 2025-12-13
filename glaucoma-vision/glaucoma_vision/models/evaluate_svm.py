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


def evaluate_svm(model_path: str, csv_path: str, val_dir: str, save_dir: str, show_plots: bool = True):
    """
    Evaluate SVM model for glaucoma binary classification
    Args:
        model_path: Path to the trained SVM model (.pkl file)
        csv_path: Path to glaucoma.csv file (contains ExpCDR values)
        val_dir: Path to validation dataset directory (Glaucoma_Negative/Positive subfolders)
        save_dir: Path to save evaluation plots
        show_plots: Whether to display AUROC/AUPRC plots (default: True)
    Returns:
        dict: Evaluation metrics including accuracy, AUROC, AUPRC, F1 scores
    """
    # Create save directory if not exists (add path check)
    if save_dir and not os.path.isdir(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create save directory {save_dir}: {e}")
    
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
        
        # Print dataset info (unified format)
        print(f"Validation Classes Mapping: {class_map}")
        
        for class_name, label in class_map.items():
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found, skipping...")
                continue
                
            files = os.listdir(class_dir)
            
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
                    print(f"Error processing image {filename}: {e}")
                    
        return np.array(features), np.array(labels)

    # Read CSV file
    print(f"Loading CSV metadata from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Load validation features and labels
    X_val, y_true = load_data_from_directory(val_dir, df)
    print(f"Validation Samples: {len(X_val)}")
    print("Running inference on validation set...")

    # ================= 2. Load SVM Model =================
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    svm_pipeline = joblib.load(model_path)
    print("Model loaded successfully.")

    # ================= 3. Inference =================
    # Hard classification results (0/1)
    y_pred = svm_pipeline.predict(X_val)
    
    # Probability/decision scores for positive class (1)
    if hasattr(svm_pipeline, "predict_proba"):
        y_probs = svm_pipeline.predict_proba(X_val)
        y_scores = y_probs[:, 1]  # Positive class probability
    else:
        # Fallback to decision function if no probability estimation
        print("Warning: Model does not support predict_proba, using decision_function.")
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
        digits=3,
        output_dict=True
    )
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3), end='')

    # ================= 5. Summary Metrics =================
    # Calculate core metrics
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = auc(*roc_curve(y_true, y_scores)[:2])
    pr_auc = average_precision_score(y_true, y_scores)

    # Print summary metrics (unified format)
    print("="*40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {roc_auc:.4f}")
    print(f"AUPRC Score: {pr_auc:.4f}")
    print("="*40)

    # ================= 6. Plot AUROC/AUPRC (One row two columns) =================
    if show_plots and len(y_true) > 0:
        # Create one row two columns figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Plot ROC Curve (unified style) ---
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC)')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # --- Plot Precision-Recall Curve (unified style) ---
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve (PRC)')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        # Adjust layout and save plot
        plt.tight_layout()
        if save_dir:
            try:
                plt.savefig(os.path.join(save_dir, 'svm_evaluation_curves.png'), dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Warning: Failed to save plot to {save_dir}: {e}")
        plt.show()
        plt.close()

    # ================= 7. Prepare Metrics Dictionary =================
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(roc_auc),
        "auprc": float(pr_auc),
        "negative_f1": float(report[target_names[0]]['f1-score']),
        "positive_f1": float(report[target_names[1]]['f1-score']),
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores
    }

    print("\n✅ SVM evaluation completed successfully!")
    return metrics


def svm_integrate(MODEL_PATH, VAL_DIR, CSV_PATH, SAVE_DIR):
    """
    Evaluate SVM integrated model (Image stats + ExpCDR) for glaucoma classification
    
    Args:
        MODEL_PATH: Path to trained SVM model (.pkl file)
        VAL_DIR: Root directory of validation images (class-separated folders)
        CSV_PATH: Path to CSV file containing ExpCDR values
        SAVE_DIR: Path to save evaluation plots
        
    Returns:
        dict: Evaluation results including metrics and predictions
    """
    # Create save directory if not exists (add path check)
    if SAVE_DIR and not os.path.isdir(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create save directory {SAVE_DIR}: {e}")
    
    # ================= 1. Data Loading Function =================
    def load_integrated_data(directory, csv_df):
        """Load validation data: image statistical features + ExpCDR from CSV"""
        features = []
        labels = []
        
        # Create mapping for fast lookup
        cdr_map = dict(zip(csv_df['Filename'], csv_df['ExpCDR']))
        class_map = {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1}
        
        # Print dataset info (unified format)
        print(f"Validation Classes Mapping: {class_map}")
        
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
                    
        print(f"Validation Samples: {count}")
        return np.array(features), np.array(labels)

    # ================= 2. Load Data & Model =================
    # Read CSV metadata
    print(f"Loading CSV metadata from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Load validation set
    X_val, y_true = load_integrated_data(VAL_DIR, df)

    # Load SVM model
    print(f"Loading integrated model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    svm_pipeline = joblib.load(MODEL_PATH)
    print("Model weights loaded successfully.")

    # ================= 3. Inference =================
    print("Running inference on validation set...")

    # Get hard predictions (0 or 1)
    y_pred = svm_pipeline.predict(X_val)

    # Get probability predictions (for ROC/PR curves)
    # predict_proba returns [N, 2] array - second column = Positive (1) probability
    if hasattr(svm_pipeline, "predict_proba"):
        y_probs = svm_pipeline.predict_proba(X_val)
        y_scores = y_probs[:, 1] 
    else:
        print("Warning: Model does not support predict_proba, using decision_function.")
        y_scores = svm_pipeline.decision_function(X_val)
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    # ================= 4. Evaluation Metrics =================
    print("\n" + "="*40)
    print("Integrated Model Classification Report")
    print("="*40)
    target_names = ['Glaucoma_Negative', 'Glaucoma_Positive']
    cls_report = classification_report(
        y_true, y_pred, 
        target_names=target_names,
        digits=3,
        output_dict=True
    )
    
    # Print classification report (unified format)
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3), end='')

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = auc(*roc_curve(y_true, y_scores)[:2])
    pr_auc = average_precision_score(y_true, y_scores)

    # Print summary metrics (unified format)
    print("="*40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {roc_auc:.4f}")
    print(f"AUPRC Score: {pr_auc:.4f}")
    print("="*40)

    # ================= 5. Plot Curves (Save to SAVE_DIR) =================
    plt.figure(figsize=(14, 6))

    # ROC Curve (unified style)
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Precision-Recall Curve (unified style)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PRC)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save plot to specified directory
    if SAVE_DIR:
        try:
            plt.savefig(os.path.join(SAVE_DIR, 'svm_integrated_curves.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save plot to {SAVE_DIR}: {e}")
    plt.show()
    plt.close()

    # Compile results (unified format)
    results = {
        "class_to_idx": {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1},
        "num_samples": len(X_val),
        "accuracy": float(accuracy),
        "auroc": float(roc_auc),
        "auprc": float(pr_auc),
        "negative_f1": float(cls_report[target_names[0]]['f1-score']),
        "positive_f1": float(cls_report[target_names[1]]['f1-score']),
        "classification_report": cls_report,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores
    }

    print("\n✅ Integrated SVM evaluation completed successfully!")
    return results

# ================= Example Usage =================
if __name__ == '__main__':
    # Configuration parameters (unified path)
    SAVE_DIR = '/Users/apple/Desktop/BIA 4/ICA'  # Save plot directory
    MODEL_PATH = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/svm.pkl'
    MODEL_PATH_INTEGRATED = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/svm_integrated.pkl'
    VAL_DIR = '/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation'
    CSV_PATH = '/Users/apple/Desktop/BIA 4/glaucoma.csv.xls'
    
    # Run standard SVM evaluation
    # metrics = evaluate_svm(MODEL_PATH, CSV_PATH, VAL_DIR, SAVE_DIR)
    
    # Run integrated SVM evaluation (correct parameter order)
    eval_results = svm_integrate(MODEL_PATH_INTEGRATED, VAL_DIR, CSV_PATH, SAVE_DIR)
    
    # Optional: Access specific metrics
    print(f"\nSVM AUROC: {eval_results['auroc']:.4f}")
    print(f"SVM Accuracy: {eval_results['accuracy']:.4f}")
