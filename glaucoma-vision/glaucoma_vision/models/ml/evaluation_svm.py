import os
import pandas as pd
import numpy as np
import joblib
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix,
    accuracy_score
)

def load_data_from_directory(val_dir, csv_df):
    features = []
    labels = []
    cdr_map = dict(zip(csv_df['Filename'], csv_df['ExpCDR']))    
    class_map = {
        'Glaucoma_Negative': 0, 
        'Glaucoma_Positive': 1
    }
    
    print(f"Loading validation data from: {val_dir} ...")
    
    for class_name, label in class_map.items():
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.exists(class_dir):
            continue            
        files = os.listdir(class_dir)
        print(f"   Processing class '{class_name}': {len(files)} images")        
        for filename in tqdm(files, desc=f"Extracting {class_name} features"):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue            
            # Skip if no ExpCDR for the file
            if filename not in cdr_map:
                continue                
            cdr = float(cdr_map[filename])
            file_path = os.path.join(class_dir, filename)
            
            try:
                # Image preprocessing (64x64)
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
                
                # Concatenate features
                feat_vec = np.concatenate(([cdr], img_mean, img_std))
                
                features.append(feat_vec)
                labels.append(label)
                
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
                
    return np.array(features), np.array(labels)

def evaluate_svm(
    model_path: str,
    val_dir: str,  
    csv_path: str
):

    print("[SVM Evaluator] Loading validation data...")
    
    # 1. Load CSV and validation data
    df = pd.read_csv(csv_path)
    X_val, y_true = load_data_from_directory(val_dir, df)
    print(f"✅ Validation Set Shape: {X_val.shape} (Samples: {len(X_val)}, Features: {X_val.shape[1]})")


    print(f"Loading SVM model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    svm_pipeline = joblib.load(model_path)
    print("✅ Model loaded successfully.")

  
    print("Running inference...")
    y_pred = svm_pipeline.predict(X_val)
    
    # Get scores (handle predict_proba/decision_function)
    if hasattr(svm_pipeline, "predict_proba"):
        y_scores = svm_pipeline.predict_proba(X_val)[:, 1] # Positive class probability
    else:
        print("Warning: Model does not support predict_proba, using decision_function.")
        y_scores = svm_pipeline.decision_function(X_val)
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min()) # Normalize to 0-1

    # 4. Calculate metrics
    report = classification_report(y_true, y_pred, target_names=['Glaucoma_Negative', 'Glaucoma_Positive'], output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Package metrics
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(roc_auc),
        "auprc": float(pr_auc),
        "negative_f1": float(report['Glaucoma_Negative']['f1-score']),
        "positive_f1": float(report['Glaucoma_Positive']['f1-score']),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
    }

    # 5. Print results (unified format with XGB/ResNet18)
    print("\n" + "="*50)
    print("SVM (ExpCDR + Image Features) - VALIDATION SET METRICS")
    print("="*50)
    print(f"Glaucoma_Negative F1 score : {metrics['negative_f1']:.4f}")
    print(f"Glaucoma_Positive F1 score : {metrics['positive_f1']:.4f}")
    print(f"Accuracy                     : {metrics['accuracy']:.4f}")
    print(f"AUROC Score                  : {metrics['auroc']:.4f}")
    print(f"AUPRC Score                  : {metrics['auprc']:.4f}")
    
    # Formatted confusion matrix (same as XGB/ResNet18)
    print("\n" + "-"*50)
    print("SVM Confusion Matrix (TN, FP, FN, TP)")
    print("-"*50)
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {tn:<10}           {fp:<10}")
    print(f"Actual Positive        {fn:<10}           {tp:<10}")
    print(f"\nConfusion Matrix Values -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*50 + "\n")

    return metrics
