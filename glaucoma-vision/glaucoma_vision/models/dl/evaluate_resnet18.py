import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, average_precision_score
)
from glaucoma_vision.utils.dl_utils import (
    get_device,
    load_dl_model,
    dl_model_inference
)

def evaluate_resnet18(
    model_path: str,
    val_dir: str,
    csv_path: str,
    img_size: tuple = (224, 224),
    batch_size: int = 8
):
    """Evaluate ResNet18 model for glaucoma detection (unified format with ML models)"""
    print("[ResNet18 Evaluator] Starting evaluation...")
    
    # 1. Load CSV (same as XGB/SVM)
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded CSV file: {csv_path} (Samples: {len(df)})")
    
    # 2. Load ResNet18 model
    model = load_dl_model("resnet18", model_path)
    
    # 3. Run DL inference (replace load_dl_data/collect_dl_predictions)
    y_true, y_scores = dl_model_inference(
        model=model,
        val_dir=val_dir,
        csv_df=df,
        img_size=img_size,
        batch_size=batch_size
    )
    
    # 4. Convert scores to hard predictions (threshold=0.5)
    y_pred = (y_scores >= 0.5).astype(int)
    
    # 5. Calculate metrics (same as XGB/SVM)
    report = classification_report(
        y_true, y_pred,
        target_names=['Glaucoma_Negative', 'Glaucoma_Positive'],
        output_dict=True
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_scores)),
        "auprc": float(average_precision_score(y_true, y_scores)),
        "negative_f1": float(report['Glaucoma_Negative']['f1-score']),
        "positive_f1": float(report['Glaucoma_Positive']['f1-score']),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
    }

    # 6. Print results (unified format with XGB/SVM)
    print("\n" + "="*50)
    print("RESNET18 - VALIDATION SET METRICS")
    print("="*50)
    print(f"Glaucoma_Negative F1 score : {metrics['negative_f1']:.4f}")
    print(f"Glaucoma_Positive F1 score : {metrics['positive_f1']:.4f}")
    print(f"Accuracy                     : {metrics['accuracy']:.4f}")
    print(f"AUROC Score                  : {metrics['auroc']:.4f}")
    print(f"AUPRC Score                  : {metrics['auprc']:.4f}")
    
    # Formatted confusion matrix
    print("\n" + "-"*50)
    print("ResNet18 Confusion Matrix (TN, FP, FN, TP)")
    print("-"*50)
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {tn:<10}           {fp:<10}")
    print(f"Actual Positive        {fn:<10}           {tp:<10}")
    print(f"\nConfusion Matrix Values -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*50 + "\n")

    return metrics

