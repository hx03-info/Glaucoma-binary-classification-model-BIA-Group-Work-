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
    accuracy_score,
    roc_auc_score
)

def evaluate_rf(model_dir: str, val_dir: str, save_dir: str):
    """
    Evaluate Random Forest model for glaucoma binary classification
    Args:
        model_dir: Path to the directory containing RF model files (pkl files)
        val_dir: Path to the validation dataset directory (contains Glaucoma_Negative/Positive subfolders)
        save_dir: Path to save evaluation plots
    Returns:
        dict: Evaluation metrics including accuracy, AUROC, AUPRC, F1 scores
    """
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # ===================== 1. 配置参数（与训练脚本一致） =====================
    IMAGE_SIZE = (64, 64)
    FEATURE_NAMES = ['R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std']
    TARGET_NAMES = ['Glaucoma_Negative', 'Glaucoma_Positive']  # 统一类别名称
    
    # ===================== 2. 加载模型相关文件 =====================
    print(f"Loading model from {model_dir}...")
    # 定义文件路径
    model_path = os.path.join(model_dir, 'random_forest_model.pkl')
    scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
    feature_info_path = os.path.join(model_dir, 'feature_info.pkl')

    # 加载模型
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # 加载标准化器
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {e}")

    # 加载特征信息（可选，用于兼容）
    try:
        with open(feature_info_path, 'rb') as f:
            feature_info = pickle.load(f)
    except:
        feature_info = {'feature_names': FEATURE_NAMES}

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

    # 打印数据集信息（统一格式）
    class_to_idx = {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1}
    print(f"Validation Classes Mapping: {class_to_idx}")
    print(f"Validation Samples: {len(filenames_val)}")

    # 提取特征
    print("Running inference on validation set...")
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

    # ===================== 5. 模型推理 =====================
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]  # 正类概率

    # ===================== 6. 计算评估指标 =====================
    print("\n" + "="*40)
    print("Random Forest Classification Report")
    print("="*40)
    report = classification_report(
        y_true, y_pred,
        target_names=TARGET_NAMES,
        digits=3,
        output_dict=True
    )
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=3), end='')

    # 核心指标
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = auc(*roc_curve(y_true, y_pred_proba)[:2])
    pr_auc = average_precision_score(y_true, y_pred_proba)

    # 打印汇总指标（统一格式）
    print("="*40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {roc_auc:.4f}")
    print(f"AUPRC Score: {pr_auc:.4f}")
    print("="*40)

    # ===================== 7. 可视化（仅ROC + PR 曲线） =====================
    plt.figure(figsize=(14, 6))

    # ROC 曲线（统一样式）
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # PR 曲线（统一样式）
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    baseline = np.sum(y_true == 1) / len(y_true)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline (AP = {baseline:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PRC)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # 保存图片到指定目录
    plt.savefig(os.path.join(save_dir, 'rf_evaluation_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # ===================== 8. 整理返回指标 =====================
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(roc_auc),
        "auprc": float(pr_auc),
        "negative_f1": float(report[TARGET_NAMES[0]]['f1-score']),
        "positive_f1": float(report[TARGET_NAMES[1]]['f1-score']),
        "classification_report": report,
        "feature_names": feature_info['feature_names'],
        "feature_importance": model.feature_importances_ if hasattr(model, 'feature_importances_') else None
    }

    print("\n✅ Random Forest evaluation completed successfully!")
    return metrics


def rf_integrate(model_dir: str, val_dir: str, save_dir: str, csv_path: str = None):
    """
    Evaluate Random Forest model for glaucoma binary classification
    (Supports both image-only and image+ExpCDR from CSV features)
    Args:
        model_dir: Path to the directory containing RF model files (pkl files)
        val_dir: Path to the validation dataset directory (contains Glaucoma_Negative/Positive subfolders)
        save_dir: Path to save evaluation plots
        csv_path: Optional path to glaucoma.csv (if provided, integrates ExpCDR feature)
    Returns:
        dict: Evaluation metrics including accuracy, AUROC, AUPRC, F1 scores
    """
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    # ===================== 1. 配置参数（统一格式） =====================
    IMAGE_SIZE = (64, 64)
    # 基础图像特征 + 可选ExpCDR
    BASE_FEATURE_NAMES = ['R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std']
    FULL_FEATURE_NAMES = ['ExpCDR'] + BASE_FEATURE_NAMES if csv_path else BASE_FEATURE_NAMES
    TARGET_NAMES = ['Glaucoma_Negative', 'Glaucoma_Positive']  # 统一类别名称

    # ===================== 2. 加载CSV ExpCDR映射（如果提供CSV） =====================
    filename_to_cdr = None
    if csv_path:
        print(f"Loading CSV metadata from: {csv_path}")
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
    print(f"Loading integrated model from {model_dir}...")
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
        print("Model weights loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # 加载标准化器
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler: {e}")

    # 加载特征信息（兼容两种模式）
    try:
        with open(feature_info_path, 'rb') as f:
            feature_info = pickle.load(f)
    except:
        feature_info = {'feature_names': FULL_FEATURE_NAMES}

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

    # 打印数据集信息（统一格式）
    class_to_idx = {'Glaucoma_Negative': 0, 'Glaucoma_Positive': 1}
    print(f"Validation Classes Mapping: {class_to_idx}")
    print(f"Validation Samples: {len(filenames_val)}")

    # 提取特征（图像 + 可选ExpCDR）
    print("Running inference on validation set...")
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

    # ===================== 6. 模型推理 =====================
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]  # 正类概率

    # ===================== 7. 计算评估指标 =====================
    print("\n" + "="*40)
    print("Integrated Model Classification Report")
    print("="*40)
    report = classification_report(
        y_true, y_pred,
        target_names=TARGET_NAMES,
        digits=3,
        output_dict=True
    )
    print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=3), end='')

    # 核心指标计算
    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)

    # 打印汇总指标（统一格式）
    print("="*40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {auroc:.4f}")
    print(f"AUPRC Score: {auprc:.4f}")
    print("="*40)

    # ===================== 8. 可视化（仅ROC + PR 曲线） =====================
    plt.figure(figsize=(14, 6))

    # ROC 曲线（统一样式）
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # PR 曲线（统一样式）
    plt.subplot(1, 2, 2)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
    baseline = np.sum(y_true == 1) / len(y_true)  # 随机分类器基线
    plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (AP = {auprc:.3f})')
    plt.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline (AP = {baseline:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PRC)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # 保存图片到指定目录
    plt.savefig(os.path.join(save_dir, 'rf_integrated_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # ===================== 9. 整理返回指标 =====================
    metrics = {
        "class_to_idx": class_to_idx,
        "num_samples": len(filenames_val),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "negative_f1": float(report[TARGET_NAMES[0]]['f1-score']),
        "positive_f1": float(report[TARGET_NAMES[1]]['f1-score']),
        "classification_report": report,
        "feature_names": FULL_FEATURE_NAMES,
        "feature_importance": model.feature_importances_ if hasattr(model, 'feature_importances_') else None,
        "used_csv": True if csv_path else False  # 标记是否使用了CSV特征
    }

    print("\n✅ Integrated Random Forest evaluation completed successfully!")
    return metrics


# ===================== 示例调用 =====================
if __name__ == "__main__":
    # 配置参数（统一路径）
    SAVE_DIR = '/Users/apple/Desktop/BIA 4/ICA'
    MODEL_DIR = "/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/RF"
    MODEL_DIR_INTEGRATED = "/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/RF_integrated"
    VAL_DIR = "/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation"
    CSV_PATH = '/Users/apple/Desktop/BIA 4/glaucoma.csv.xls'

    # 执行普通RF评估
    # rf_metrics = evaluate_rf(model_dir=MODEL_DIR, val_dir=VAL_DIR, save_dir=SAVE_DIR)
    
    # 执行集成RF评估
    rf_integrated_metrics = rf_integrate(
        model_dir=MODEL_DIR_INTEGRATED,
        val_dir=VAL_DIR,
        save_dir=SAVE_DIR,
        csv_path=CSV_PATH
    )

