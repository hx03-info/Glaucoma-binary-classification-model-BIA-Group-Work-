import os
import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
)

def extract_img_features(path):
    try:
        img = cv2.imread(path)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        f = {}
        # Color features
        f['Mean_R'] = np.mean(img[:,:,0]); f['Mean_G'] = np.mean(img[:,:,1]); f['Mean_B'] = np.mean(img[:,:,2])
        f['Std_R'] = np.std(img[:,:,0])
        
        # Center region features
        c=60; h,w,_=img.shape; center=img[h//2-c:h//2+c, w//2-c:w//2+c]
        f['Center_R'] = np.mean(center[:,:,0])
        f['Center_Bright_Ratio'] = np.mean(center) / (np.mean(img) + 1e-5)
        
        # Texture features (Variance/Entropy)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f['Tex_Variance'] = np.var(gray)
        f['Tex_Entropy'] = -np.sum((gray/255.0) * np.log2((gray/255.0) + 1e-7))
        return f
    except Exception as e:
        print(f"Feature extraction failed {path}: {e}")
        return None

def evaluate_xgb(
    model_path: str,
    val_dir: str,  
    csv_path: str,
    save_dir: str
):
    """
    Evaluate XGBoost model for glaucoma binary classification
    Args:
        model_path: Path to XGBoost model (.json)
        val_dir: Path to validation dataset directory (contains Glaucoma_Negative/Positive subfolders)
        csv_path: Path to glaucoma.csv file (contains ExpCDR values)
        save_dir: Path to save evaluation plots
    Returns:
        dict: Evaluation metrics including accuracy, AUROC, AUPRC, F1 scores
    """
    # Create save directory if not exists (add path check)
    if save_dir and not os.path.isdir(save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create save directory {save_dir}: {e}")
    
    df = pd.read_csv(csv_path)
    folder_map = {0: 'Glaucoma_Negative', 1: 'Glaucoma_Positive'}    
    data_list, labels = [], []
    
    # Print dataset info (unified format)
    class_to_idx = {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1}
    print(f"Validation Classes Mapping: {class_to_idx}")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        img_path = os.path.join(val_dir, folder_map[row['Glaucoma']], row['Filename'])
        if os.path.exists(img_path):
            feats = extract_img_features(img_path)
            if feats:
                data_list.append(feats)
                labels.append(row['Glaucoma'])

    # Model inference
    X_test = pd.DataFrame(data_list)
    y_test = np.array(labels)
    print(f"Validation Samples: {len(X_test)} (Negative: {np.sum(y_test==0)}, Positive: {np.sum(y_test==1)})")

    # Load model
    print(f"Loading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print("Model loaded successfully.")

    # Prediction
    print("Running inference on validation set...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Calculate metrics
    target_names = ['Glaucoma_Negative', 'Glaucoma_Positive']
    report = classification_report(
        y_test, y_pred, 
        target_names=target_names,
        digits=3,
        output_dict=True
    )
    
    # Core metrics (unified format)
    accuracy = float(accuracy_score(y_test, y_pred))
    auroc = float(roc_auc_score(y_test, y_prob))
    auprc = float(average_precision_score(y_test, y_prob))
    negative_f1 = float(report['Glaucoma_Negative']['f1-score'])
    positive_f1 = float(report['Glaucoma_Positive']['f1-score'])

    # Print classification report (unified format)
    print("\n" + "="*40)
    print("XGBoost Classification Report")
    print("="*40)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3), end='')

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
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # PR Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    baseline = np.sum(y_test == 1) / len(y_test)
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
            plt.savefig(os.path.join(save_dir, 'xgboost_evaluation_curves.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save plot to {save_dir}: {e}")
    plt.show()
    plt.close()

    # Prepare metrics dictionary (unified format)
    metrics = {
        "class_to_idx": class_to_idx,
        "num_samples": len(X_test),
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "negative_f1": negative_f1,
        "positive_f1": positive_f1,
        "classification_report": report
    }

    print("\n✅ XGBoost evaluation completed successfully!")
    return metrics


def xgb_integrate(MODEL_PATH, VAL_DIR, CSV_PATH, SAVE_DIR):
    """
    Evaluate XGBoost integrated model (Image + ExpCDR) for glaucoma classification
    Args:
        MODEL_PATH: Path to XGBoost model (.json)
        VAL_DIR: Root directory of validation images (class-separated folders)
        CSV_PATH: Path to CSV file containing ExpCDR values
        SAVE_DIR: Path to save evaluation plots
    Returns:
        dict: Comprehensive evaluation results
    """
    # Create save directory if not exists (add path check)
    if SAVE_DIR and not os.path.isdir(SAVE_DIR):
        try:
            os.makedirs(SAVE_DIR, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create save directory {SAVE_DIR}: {e}")
    
    # ================= Path Validation =================
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file missing: {CSV_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")
    if not os.path.exists(VAL_DIR):
        raise FileNotFoundError(f"Validation directory missing: {VAL_DIR}")
    
    print(f"Validation set directory: {VAL_DIR}")
    print(f"Class folders: {[d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))]}")
    
    # ================= Feature Extraction Function =================
    def extract_img_features(path):
        """Extract standardized image features (RGB, texture, center region)"""
        try:
            img = cv2.imread(path)
            if img is None:
                return None
            
            # Convert to RGB and resize to 224x224 (consistent with training)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            
            features = {}
            # Color channel statistics
            features['Mean_R'] = np.mean(img[:, :, 0])
            features['Mean_G'] = np.mean(img[:, :, 1])
            features['Mean_B'] = np.mean(img[:, :, 2])
            features['Std_R'] = np.std(img[:, :, 0])
            
            # Central region features (60x60 center crop)
            c = 60
            h, w, _ = img.shape
            center = img[h//2 - c : h//2 + c, w//2 - c : w//2 + c]
            features['Center_R'] = np.mean(center[:, :, 0])
            features['Center_Bright_Ratio'] = np.mean(center) / (np.mean(img) + 1e-5)
            
            # Texture features from grayscale image
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            features['Tex_Variance'] = np.var(gray)
            # Fix entropy calculation (avoid log(0))
            gray_norm = gray / 255.0
            gray_norm = np.clip(gray_norm, 1e-7, 1.0)
            features['Tex_Entropy'] = -np.sum(gray_norm * np.log2(gray_norm))
            
            return features
        
        except Exception as e:
            return None

    # ================= Data Loading (Full Validation Set) =================
    print("\nLoading validation set data...")
     
    # Load and clean CSV data
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['Filename', 'ExpCDR', 'Glaucoma'])
    df['eye_code'] = df['Eye'].map({'OD': 0, 'OS': 1}).fillna(0)
    df['set_code'] = df['Set'].map({'A': 0, 'B': 1}).fillna(0)
    
    # Map glaucoma labels to directory paths
    class_dirs = {
        0: os.path.join(VAL_DIR, 'Glaucoma_Negative'),
        1: os.path.join(VAL_DIR, 'Glaucoma_Positive')
    }
    
    # Collect all valid images from validation directories
    valid_images = {}
    total_val_images = 0
    for label, dir_path in class_dirs.items():
        if os.path.exists(dir_path):
            # Filter for image files only
            img_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            valid_images[label] = img_files
            total_val_images += len(img_files)
    
    # Print dataset info (unified format)
    class_to_idx = {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1}
    print(f"Validation Classes Mapping: {class_to_idx}")
    print(f"Total validation images found: {total_val_images}")
    print(f"Negative samples: {len(valid_images.get(0, []))}")
    print(f"Positive samples: {len(valid_images.get(1, []))}")
    
    # Match CSV entries with validation images
    data_list, labels = [], []
    matched_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing sample data"):
        filename = row['Filename']
        csv_label = row['Glaucoma']
        
        # Skip if image not found in validation set
        if csv_label not in valid_images or filename not in valid_images[csv_label]:
            continue
        
        # Get full image path and extract features
        img_path = os.path.join(class_dirs[csv_label], filename)
        img_feats = extract_img_features(img_path)
        
        if img_feats is None:
            continue
        
        # Merge image features with clinical/metadata features
        img_feats['ExpCDR'] = row['ExpCDR']
        img_feats['eye'] = row['eye_code']
        img_feats['set'] = row['set_code']
        
        data_list.append(img_feats)
        labels.append(csv_label)
        matched_count += 1
    
    # Critical check for valid data
    if matched_count == 0:
        raise ValueError("No matching samples between CSV file and validation set images! Please check if filenames and labels match.")
    
    print(f"\nValidation Samples: {matched_count}")
    
    # Convert to DataFrame/array for model input
    X_val = pd.DataFrame(data_list)
    y_val = np.array(labels)
    
    # ================= Model Inference =================
    print("\nLoading model and performing inference...")
    # Load pre-trained XGBoost model
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # Generate predictions on full validation set
    y_prob = model.predict_proba(X_val)[:, 1]  # Positive class probabilities
    y_pred = model.predict(X_val)              # Hard predictions (0/1)
    
    # ================= Evaluation Metrics Calculation =================
    # Generate classification report (unified format)
    target_names = ['Glaucoma_Negative', 'Glaucoma_Positive']
    report = classification_report(
        y_val, y_pred,
        target_names=target_names,
        digits=3,
        output_dict=True
    )
    
    # Calculate core metrics
    accuracy = accuracy_score(y_val, y_pred)
    auroc = roc_auc_score(y_val, y_prob)
    auprc = average_precision_score(y_val, y_prob)
    
    # Print classification report (unified format)
    print("\n" + "="*40)
    print("XGBoost (Hybrid Features) Classification Report")
    print("="*40)
    print(classification_report(y_val, y_pred, target_names=target_names, digits=3), end='')

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
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # PR Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    baseline = np.sum(y_val == 1) / len(y_val)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {auprc:.3f})')
    plt.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline (AP = {baseline:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PRC)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save plot to specified directory
    if SAVE_DIR:
        try:
            plt.savefig(os.path.join(SAVE_DIR, 'xgboost_integrated_curves.png'), dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Warning: Failed to save plot to {SAVE_DIR}: {e}")
    plt.show()
    plt.close()
    
    # ================= Compile Results =================
    results = {
        "class_to_idx": class_to_idx,
        "total_validation_images": total_val_images,
        "matched_samples": matched_count,
        "num_evaluation_samples": len(y_val),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "negative_f1": float(report['Glaucoma_Negative']['f1-score']),
        "positive_f1": float(report['Glaucoma_Positive']['f1-score']),
        "classification_report": report,
        "y_true": y_val,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "feature_names": X_val.columns.tolist()
    }

    print("\n✅ XGBoost (Hybrid Features) evaluation completed successfully!")
    return results

# ================= Example Usage =================
if __name__ == '__main__':
    # Configuration parameters (unified path)
    SAVE_DIR = '/Users/apple/Desktop/BIA 4/ICA'  # Save plot directory
    MODEL_PATH = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/xgb.json'
    MODEL_PATH_INTEGRATED = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/xgb_integrated.json'
    VAL_DIR = '/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation'
    CSV_PATH = '/Users/apple/Desktop/BIA 4/glaucoma.csv.xls'
    
    # Run standard XGBoost evaluation
    # metrics = evaluate_xgb(MODEL_PATH, VAL_DIR, CSV_PATH, SAVE_DIR)
    
    # Run integrated XGBoost evaluation (correct parameter order)
    eval_results = xgb_evaluate(MODEL_PATH_INTEGRATED, VAL_DIR, CSV_PATH, SAVE_DIR)
    
    # Optional: Access specific metrics
    print(f"\nXGBoost AUROC: {eval_results['auroc']:.4f}")
    print(f"XGBoost Accuracy: {eval_results['accuracy']:.4f}")



