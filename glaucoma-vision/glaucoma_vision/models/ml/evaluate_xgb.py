import os
import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, average_precision_score
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
    csv_path: str
):
    df = pd.read_csv(csv_path)
    folder_map = {0: 'Glaucoma_Negative', 1: 'Glaucoma_Positive'}    
    data_list, labels = [], []
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
    print(f"✅ Loaded {len(X_test)} valid samples (Negative: {np.sum(y_test==0)}, Positive: {np.sum(y_test==1)})")

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Prediction
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Calculate metrics
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "auprc": float(average_precision_score(y_test, y_prob)),
        "negative_f1": float(report['Negative']['f1-score']),
        "positive_f1": float(report['Positive']['f1-score']),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
    }

    # Print results
    print("\n" + "="*50)
    print("XGBOOST (IMAGE ONLY) - VALIDATION SET METRICS")
    print("="*50)
    print(f"Glaucoma_Negative F1 score : {metrics['negative_f1']:.4f}")
    print(f"Glaucoma_Positive F1 score : {metrics['positive_f1']:.4f}")
    print(f"Accuracy                     : {metrics['accuracy']:.4f}")
    print(f"AUROC Score                  : {metrics['auroc']:.4f}")
    print(f"AUPRC Score                  : {metrics['auprc']:.4f}")
    
    # Formatted confusion matrix
    print("\n" + "-"*50)
    print("XGBoost Confusion Matrix (TN, FP, FN, TP)")
    print("-"*50)
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {tn:<10}           {fp:<10}")
    print(f"Actual Positive        {fn:<10}           {tp:<10}")
    print(f"\nConfusion Matrix Values -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*50 + "\n")

    return metrics


import os
import cv2
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, average_precision_score
)

def xgb_evaluate(MODEL_PATH, VAL_DIR, CSV_PATH):
    """
    Final XGBoost evaluation with two SHAP plots side by side (one row)
    Uses all 130 validation samples with English output and comments
    
    Args:
        MODEL_PATH: Path to XGBoost model (.json)
        VAL_DIR: Direct path to validation images root
        CSV_PATH: Path to glaucoma CSV with ExpCDR data
        
    Returns:
        dict: Comprehensive evaluation results with SHAP analysis
    """
    # Initialize SHAP JavaScript for notebook visualization
    shap.initjs()
    
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
            # Silent fail for feature extraction errors
            return None

    # ================= Data Loading (Full Validation Set) =================
    print("\n[XGB Evaluator] Loading validation set data...")
    
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
    
    # Print image count statistics
    print(f"Total validation images found: {total_val_images}")
    print(f"Negative samples (Glaucoma_Negative): {len(valid_images.get(0, []))}")
    print(f"Positive samples (Glaucoma_Positive): {len(valid_images.get(1, []))}")
    
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
    
    print(f"\nMatched validation samples: {matched_count} (full evaluation set)")
    
    # Convert to DataFrame/array for model input
    X_val = pd.DataFrame(data_list)
    y_val = np.array(labels)
    
    # ================= Model Inference =================
    print("\n[XGB Evaluator] Loading model and performing inference...")
    # Load pre-trained XGBoost model
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    
    # Generate predictions on full validation set
    y_prob = model.predict_proba(X_val)[:, 1]  # Positive class probabilities
    y_pred = model.predict(X_val)              # Hard predictions (0/1)
    
    # ================= Evaluation Metrics Calculation =================
    # Generate classification report
    report = classification_report(
        y_val, y_pred,
        target_names=['Glaucoma Negative', 'Glaucoma Positive'],
        output_dict=True
    )
    
    # Calculate confusion matrix and derived metrics
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size >=4 else (0,0,0,0)
    accuracy = accuracy_score(y_val, y_pred)
    auroc = roc_auc_score(y_val, y_prob)
    auprc = average_precision_score(y_val, y_prob)
    
    # Clinical metrics (sensitivity/specificity)
    sensitivity = tp/(tp+fn) if (tp+fn) >0 else 0  # Sensitivity/Recall
    specificity = tn/(tn+fp) if (tn+fp) >0 else 0  # Specificity
    
    # Print comprehensive evaluation report
    print("\n" + "="*60)
    print("XGBoost (Hybrid Features) Validation Set Evaluation Report (130 samples)")
    print("="*60)
    print(classification_report(y_val, y_pred, target_names=['Glaucoma Negative', 'Glaucoma Positive']))
    print(f"Core Evaluation Metrics:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  AUROC:       {auroc:.4f}")
    print(f"  AUPRC:       {auprc:.4f}")
    print(f"  Sensitivity/Recall: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  Confusion Matrix: True Negative(TN)={tn}, False Positive(FP)={fp}, False Negative(FN)={fn}, True Positive(TP)={tp}")
    print("="*60)
    
    # ================= SHAP Analysis (Two Plots Side by Side) =================
    print("\n[XGB Evaluator] Generating SHAP feature analysis plots (two plots side by side)...")
    
    # Compute SHAP values using TreeExplainer (optimal for XGBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    # 优化：增大图形尺寸 + 加宽子图间距
    fig = plt.figure(figsize=(22, 15))  # 增大整体尺寸，提供更多显示空间
    
    # 1. SHAP Bar Plot (Feature Importance) - Left Plot
    ax1 = plt.subplot(1, 2, 1)
    shap.summary_plot(
        shap_values, X_val,
        plot_type="bar",
        feature_names=X_val.columns,
        show=False
    )
    # 设置标题和标签，优化字体和间距

    plt.gca().set_xlabel("SHAP Feature Importance Value", fontsize=12)
    # 调整Y轴标签间距，避免重叠
    plt.gca().tick_params(axis='y', labelsize=10, pad=8)
    
    # 2. SHAP Dot Plot (Feature Effect) - Right Plot
    ax2 = plt.subplot(1, 2, 2)
    shap.summary_plot(
        shap_values, X_val,
        feature_names=X_val.columns,
        show=False
    )
    # 设置标题，优化字体和间距
    # 调整Y轴标签间距，避免重叠
    plt.gca().tick_params(axis='y', labelsize=10, pad=8)
    
    # 优化：加宽子图之间的水平间距（从0.3增加到0.5）
    plt.subplots_adjust(wspace=4, hspace=4)
    plt.tight_layout()
    plt.show()
    
    # ================= Compile Results =================
    results = {
        "total_validation_images": total_val_images,
        "matched_samples": matched_count,
        "num_evaluation_samples": len(y_val),
        "classification_report": report,
        "confusion_matrix": cm,
        "accuracy": accuracy,
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "y_true": y_val,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "shap_values": shap_values,
        "shap_explainer": explainer,
        "feature_names": X_val.columns.tolist()
    }
    
    return results



