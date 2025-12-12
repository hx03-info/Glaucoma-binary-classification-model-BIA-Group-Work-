import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score
)

def evaluate_rf(model_dir: str, val_dir: str):
    """
    Evaluate Random Forest model for glaucoma binary classification
    Args:
        model_dir: Path to the directory containing RF model files (pkl files)
        val_dir: Path to the validation dataset directory (contains Glaucoma_Negative/Positive subfolders)
    Returns:
        dict: Evaluation metrics including accuracy, AUROC, AUPRC, F1 scores and confusion matrix
    """
    # ===================== 1. 配置参数（与训练脚本一致） =====================
    IMAGE_SIZE = (64, 64)
    FEATURE_NAMES = ['R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std']
    TARGET_NAMES = ['Non-Glaucoma', 'Glaucoma']
    
    print("[RF Evaluator] Starting Random Forest Evaluation Pipeline")
    print("=" * 60)

    # ===================== 2. 加载模型相关文件 =====================
    print("[RF Evaluator] Loading model and dependencies...")
    # 定义文件路径
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
    feature_info_path = os.path.join(model_dir, 'feature_info.pkl')

    # 加载模型
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # 加载标准化器
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler loaded from: {scaler_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {e}")

    # 加载特征信息（可选，用于兼容）
    try:
        with open(feature_info_path, 'rb') as f:
            feature_info = pickle.load(f)
        print(f"✓ Feature info loaded from: {feature_info_path}")
    except:
        feature_info = {'feature_names': FEATURE_NAMES}
        print(f"⚠️ Feature info not found, using default names")

    # ===================== 3. 定义特征提取函数（与训练脚本完全一致） =====================
    def get_features(image):
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[1] > image.shape[0]:
            image = np.rot90(image)
        image = resize(image, IMAGE_SIZE)
        img_mean = np.mean(image, axis=(0, 1))
        img_std = np.std(image, axis=(0, 1))
        return np.concatenate((img_mean, img_std))

    # ===================== 4. 加载验证集图像并提取特征 =====================
    print("\n[RF Evaluator] Loading validation images and extracting features...")
    # 定义类别路径
    val_positive_dir = os.path.join(val_dir, "Glaucoma_Positive")
    val_negative_dir = os.path.join(val_dir, "Glaucoma_Negative")

    # 读取图像路径和标签
    val_negative_files = [os.path.join(val_negative_dir, f) for f in os.listdir(val_negative_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    val_positive_files = [os.path.join(val_positive_dir, f) for f in os.listdir(val_positive_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 合并路径和标签
    filenames_val = val_negative_files + val_positive_files
    y_true = np.array([0] * len(val_negative_files) + [1] * len(val_positive_files))

    print(f"Validation set: {len(filenames_val)} images ({len(val_negative_files)} negative, {len(val_positive_files)} positive)")

    # 提取特征
    X_val = []
    for idx, filepath in enumerate(filenames_val):
        try:
            img = imread(filepath)
            feat = get_features(img)
            X_val.append(feat)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            X_val.append(np.zeros(6))  # 失败时填充0特征

    X_val = np.array(X_val)
    # 特征标准化（必须用训练好的scaler）
    X_val_scaled = scaler.transform(X_val)
    print(f"✓ Validation features shape: {X_val_scaled.shape}")

    # ===================== 5. 模型推理 =====================
    print("\n[RF Evaluator] Running inference on validation set...")
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]  # 正类概率
    print("✓ Prediction completed!")

    # ===================== 6. 计算评估指标 =====================
    print("\n" + "=" * 40)
    print("Classification Report")
    print("=" * 40)
    report = classification_report(y_true, y_pred, target_names=TARGET_NAMES, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES))

    # 核心指标
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = auc(*roc_curve(y_true, y_pred_proba)[:2])
    pr_auc = average_precision_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 打印汇总指标
    print("=" * 40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {roc_auc:.4f}")
    print(f"AUPRC Score: {pr_auc:.4f}")
    print(f"Confusion Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("=" * 40)

    # ===================== 7. 可视化（ROC + PR 曲线） =====================
    plt.figure(figsize=(14, 6))

    # ROC 曲线
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # PR 曲线
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    baseline = np.sum(y_true == 1) / len(y_true)
    plt.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline (AP = {baseline:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PRC)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ===================== 8. 整理返回指标 =====================
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(roc_auc),
        "auprc": float(pr_auc),
        "negative_f1": float(report[TARGET_NAMES[0]]['f1-score']),
        "positive_f1": float(report[TARGET_NAMES[1]]['f1-score']),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "classification_report": report,
        "feature_names": feature_info['feature_names'],
        "feature_importance": model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    }

    print("\n✅ [RF Evaluator] Evaluation completed successfully!")
    return metrics

# ===================== 示例调用 =====================
if __name__ == "__main__":
    # 替换为你的实际路径
    MODEL_DIR = "/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/RF"  # 存放3个pkl文件的文件夹
    VAL_DIR = "/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation"

    # 执行评估
    rf_metrics = evaluate_rf(model_dir=MODEL_DIR, val_dir=VAL_DIR)


import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
)

def rf_integrate(model_dir: str, val_dir: str, csv_path: str = None):
    """
    Evaluate Random Forest model for glaucoma binary classification
    (Supports both image-only and image+ExpCDR from CSV features)
    Args:
        model_dir: Path to the directory containing RF model files (pkl files)
        val_dir: Path to the validation dataset directory (contains Glaucoma_Negative/Positive subfolders)
        csv_path: Optional path to glaucoma.csv (if provided, integrates ExpCDR feature)
    Returns:
        dict: Evaluation metrics including accuracy, AUROC, AUPRC, F1 scores and confusion matrix
    """
    # ===================== 1. 配置参数（与训练脚本完全对齐） =====================
    IMAGE_SIZE = (64, 64)
    # 基础图像特征 + 可选ExpCDR
    BASE_FEATURE_NAMES = ['R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std']
    FULL_FEATURE_NAMES = ['ExpCDR'] + BASE_FEATURE_NAMES if csv_path else BASE_FEATURE_NAMES
    TARGET_NAMES = ['Non-Glaucoma', 'Glaucoma']
    
    print("[RF Evaluator] Starting Random Forest Evaluation Pipeline (with CSV ExpCDR support)")
    print("=" * 60)

    # ===================== 2. 加载CSV ExpCDR映射（如果提供CSV） =====================
    filename_to_cdr = None
    if csv_path:
        print(f"[RF Evaluator] Loading CSV metadata from: {csv_path}")
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            # 构建文件名到ExpCDR的映射
            filename_to_cdr = {}
            for idx, row in df.iterrows():
                filename_to_cdr[row['Filename']] = row['ExpCDR']
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV: {e}")

    # ===================== 3. 加载模型相关文件 =====================
    print("\n[RF Evaluator] Loading model and dependencies...")
    # 定义模型文件路径（兼容训练脚本的命名）
    model_filename = 'random_forest_csv_model.pkl' if csv_path else 'random_forest_model.pkl'
    scaler_filename = 'feature_scaler_csv.pkl' if csv_path else 'feature_scaler.pkl'
    feature_info_filename = 'feature_info_csv.pkl' if csv_path else 'feature_info.pkl'

    model_path = os.path.join(model_dir, model_filename)
    scaler_path = os.path.join(model_dir, scaler_filename)
    feature_info_path = os.path.join(model_dir, feature_info_filename)

    # 加载模型
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # 加载标准化器
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler loaded from: {scaler_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {e}")

    # 加载特征信息（兼容两种模式）
    try:
        with open(feature_info_path, 'rb') as f:
            feature_info = pickle.load(f)
        print(f"✓ Feature info loaded from: {feature_info_path}")
    except:
        feature_info = {'feature_names': FULL_FEATURE_NAMES}
        print(f"⚠️ Feature info not found, using default names: {FULL_FEATURE_NAMES}")

    # ===================== 4. 定义特征提取函数（匹配训练脚本） =====================
    def get_features(image, exp_cdr=None):
        """
        Extract features (image stats + optional ExpCDR)
        Args:
            image: Input fundus image
            exp_cdr: ExpCDR value from CSV (optional)
        Returns:
            Combined feature vector
        """
        # 图像特征提取（和训练脚本完全一致）
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[1] > image.shape[0]:
            image = np.rot90(image)
        image = resize(image, IMAGE_SIZE)
        img_mean = np.mean(image, axis=(0, 1))  # R, G, B mean
        img_std = np.std(image, axis=(0, 1))    # R, G, B std
        img_features = np.concatenate((img_mean, img_std))
        
        # 集成ExpCDR特征（如果提供）
        if exp_cdr is not None:
            return np.concatenate(([exp_cdr], img_features))  # [ExpCDR, R_mean, G_mean, B_mean, R_std, G_std, B_std]
        return img_features  # 仅图像特征

    # ===================== 5. 加载验证集并提取特征 =====================
    print("\n[RF Evaluator] Loading validation images and extracting features...")
    # 定义验证集类别路径
    val_positive_dir = os.path.join(val_dir, "Glaucoma_Positive")
    val_negative_dir = os.path.join(val_dir, "Glaucoma_Negative")

    # 读取所有验证集图像路径
    val_negative_files = [os.path.join(val_negative_dir, f) for f in os.listdir(val_negative_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    val_positive_files = [os.path.join(val_positive_dir, f) for f in os.listdir(val_positive_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 合并路径和标签
    filenames_val = val_negative_files + val_positive_files
    y_true = np.array([0] * len(val_negative_files) + [1] * len(val_positive_files))

    print(f"Validation set: {len(filenames_val)} images "
          f"({len(val_negative_files)} negative, {len(val_positive_files)} positive)")

    # 提取特征（图像 + 可选ExpCDR）
    X_val = []
    for idx, filepath in enumerate(filenames_val):
        try:
            # 读取图像
            img = imread(filepath)
            # 获取文件名（用于匹配CSV的ExpCDR）
            filename = os.path.basename(filepath)
            # 获取ExpCDR（默认0.5，匹配训练脚本的容错逻辑）
            exp_cdr = filename_to_cdr.get(filename, 0.5) if filename_to_cdr else None
            # 提取特征
            feat = get_features(img, exp_cdr)
            X_val.append(feat)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            # 失败时填充0特征（维度匹配：6=仅图像，7=图像+ExpCDR）
            fill_dim = 7 if csv_path else 6
            X_val.append(np.zeros(fill_dim))

    X_val = np.array(X_val)
    # 特征标准化（必须用训练好的scaler）
    X_val_scaled = scaler.transform(X_val)
    print(f"✓ Validation features shape: {X_val_scaled.shape} "
          f"(features: {FULL_FEATURE_NAMES})")

    # ===================== 6. 模型推理 =====================
    print("\n[RF Evaluator] Running inference on validation set...")
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]  # 正类概率
    print("✓ Prediction completed!")

    # ===================== 7. 计算评估指标 =====================
    print("\n" + "=" * 40)
    print("Classification Report")
    print("=" * 40)
    report = classification_report(y_true, y_pred, target_names=TARGET_NAMES, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES))

    # 核心指标计算
    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # 衍生指标
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率/敏感度
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异度
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # 精确率
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0


    # ===================== 8. 可视化（ROC + PR + 混淆矩阵） =====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. 混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES,
                annot_kws={'size': 12, 'weight': 'bold'})
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)

    # 2. ROC 曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auroc:.3f}')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    # 3. PR 曲线
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    baseline = np.sum(y_true == 1) / len(y_true)  # 随机分类器基线
    axes[2].plot(recall_curve, precision_curve, color='green', lw=2, label=f'AP = {auprc:.3f}')
    axes[2].axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline (AP = {baseline:.3f})')
    axes[2].set_xlabel('Recall', fontsize=12)
    axes[2].set_ylabel('Precision', fontsize=12)
    axes[2].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[2].legend(loc="lower left")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'RF Evaluation Results ({"Image + ExpCDR" if csv_path else "Image Only"})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ===================== 9. 特征重要性可视化（可选） =====================
    #if hasattr(model, 'feature_importances_'):
        #plt.figure(figsize=(10, 6))
        #importance = model.feature_importances_
        #indices = np.argsort(importance)[::-1]
        # 高亮ExpCDR特征
        #colors = ['crimson' if feat == 'ExpCDR' else 'steelblue' for feat in [FULL_FEATURE_NAMES[i] for i in indices]]
        
        #plt.bar(range(len(importance)), importance[indices], color=colors, alpha=0.8)
        #plt.xticks(range(len(importance)), [FULL_FEATURE_NAMES[i] for i in indices], rotation=45)
        #plt.xlabel('Features', fontsize=12, fontweight='bold')
        #plt.ylabel('Importance', fontsize=12, fontweight='bold')
        #plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        #plt.grid(True, alpha=0.3, axis='y')
        
        # 图例区分ExpCDR和图像特征
        #from matplotlib.patches import Patch
        #legend_elements = [
            #Patch(facecolor='crimson', alpha=0.8, label='ExpCDR (Clinical)'),
            #Patch(facecolor='steelblue', alpha=0.8, label='Image Features')
        #] if csv_path else None
        #if legend_elements:
            #plt.legend(handles=legend_elements, loc='upper right')
        
        #plt.tight_layout()
        #plt.show()

    # ===================== 10. 整理返回指标 =====================
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "negative_f1": float(report[TARGET_NAMES[0]]['f1-score']),
        "positive_f1": float(report[TARGET_NAMES[1]]['f1-score']),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_overall": float(f1),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "classification_report": report,
        "feature_names": FULL_FEATURE_NAMES,
        "feature_importance": model.feature_importances_ if hasattr(model, 'feature_importances_') else None,
        "used_csv": True if csv_path else False  # 标记是否使用了CSV特征
    }

    print("\n✅ [RF Evaluator] Evaluation completed successfully!")
    return metrics


if __name__ == "__main__":
    # 替换为你的实际路径
    MODEL_DIR = "/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/RF_integrated"
    VAL_DIR = "/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation"
    CSV_PATH = '/Users/apple/Desktop/BIA 4/glaucoma.csv.xls'

