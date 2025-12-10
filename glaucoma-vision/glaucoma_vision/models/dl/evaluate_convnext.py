import os
from glaucoma_vision.utils.dl_utils import load_dl_data, load_dl_model, DEVICE
from glaucoma_vision.utils.evaluation import (
    calculate_metrics, collect_dl_predictions
)

def evaluate_convnext(
    model_path: str,
    data_dir: str
):
    val_loader = load_dl_data(data_dir, model_type="convnext")
    print(f"✅ ConvNeXt Validation Data Set：{len(val_loader.dataset)}Sample")
    model = load_dl_model(model_path, model_type="convnext")
    y_true, y_pred, y_scores = collect_dl_predictions(model, val_loader, model_type="convnext")
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    print("\n" + "="*40)
    print("ConvNeXt Classification Report")
    print("="*40)
    from sklearn.metrics import classification_report
    target_names = ['Glaucoma_Negative', 'Glaucoma_Positive']
    print(classification_report(y_true, y_pred, target_names=target_names))
    print("\n" + "-"*40)
    print("ConvNeXt Confusion Matrix (TN, FP, FN, TP)")
    print("-"*40)
    tn = metrics['confusion_matrix']['TN']
    fp = metrics['confusion_matrix']['FP']
    fn = metrics['confusion_matrix']['FN']
    tp = metrics['confusion_matrix']['TP']
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {tn:<10}           {fp:<10}")
    print(f"Actual Positive        {fn:<10}           {tp:<10}")
    print(f"\nConfusion Matrix Values -> TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"\nSummary Metrics (ConvNeXt):")
    print(f"AUROC Score: {metrics['auroc']:.4f}")
    print(f"AUPRC Score: {metrics['auprc']:.4f}")
    print("="*40)
    
    return metrics
