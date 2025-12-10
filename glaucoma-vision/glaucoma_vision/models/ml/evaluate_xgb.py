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
    """Extract image features (color/center/texture)"""
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
    base_path: str,
    csv_path: str
):
    df = pd.read_csv(csv_path)
    folder_map = {0: 'Glaucoma_Negative', 1: 'Glaucoma_Positive'}
    val_root = os.path.join(base_path, 'Validation')
    
    data_list, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        img_path = os.path.join(val_root, folder_map[row['Glaucoma']], row['Filename'])
        if os.path.exists(img_path):
            feats = extract_img_features(img_path)
            if feats:
                data_list.append(feats)
                labels.append(row['Glaucoma'])

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


    print("\n" + "="*50)
    print("XGBOOST (IMAGE ONLY) - VALIDATION SET METRICS")
    print("="*50)
    print(f"Glaucoma_Negative F1 score : {metrics['negative_f1']:.4f}")
    print(f"Glaucoma_Positive F1 score : {metrics['positive_f1']:.4f}")
    print(f"Accuracy                     : {metrics['accuracy']:.4f}")
    print(f"AUROC Score                  : {metrics['auroc']:.4f}")
    print(f"AUPRC Score                  : {metrics['auprc']:.4f}")
    

    print("\n" + "-"*50)
    print("XGBoost Confusion Matrix (TN, FP, FN, TP)")
    print("-"*50)
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {tn:<10}           {fp:<10}")
    print(f"Actual Positive        {fn:<10}           {tp:<10}")
    print(f"\nConfusion Matrix Values -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*50 + "\n")

    return metrics
