import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score
)
import matplotlib.pyplot as plt

def evaluate_mobilenet(model_path: str, val_dir: str):
    """
    Evaluate MobileNetV2 model for glaucoma binary classification
    Args:
        model_path: Path to the trained MobileNetV2 model weights
        val_dir: Path to the validation dataset directory (contains Glaucoma_Negative/Positive subfolders)
    Returns:
        dict: Evaluation metrics including accuracy, AUROC, AUPRC, F1 scores and confusion matrix
    """
    # Configuration (对齐ConvNeXt的配置风格)
    IMG_SIZE = 224
    BATCH_SIZE = 32
    

    print("[MobileNetV2 Evaluator] Loading validation dataset...")
    
    # 构建数据生成器（模拟PyTorch的transforms逻辑）
    val_datagen = ImageDataGenerator(
        rescale=1./255,  # 对应ToTensor()的归一化
        featurewise_center=False,
        featurewise_std_normalization=False
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,  # 保持顺序一致
        classes=None,   # 自动识别子文件夹
        seed=42
    )
    
    # 对齐ConvNeXt的输出格式
    class_to_idx = val_generator.class_indices
    print(f"Validation Classes Mapping: {class_to_idx}")
    print(f"Validation Samples: {val_generator.samples}")

    # Step 2: Model Loading and Initialization (对齐ConvNeXt的加载逻辑)
    # 设备配置（模拟PyTorch的device逻辑）
    device = tf.device("cuda" if tf.config.list_physical_devices('GPU') else "cpu")
    print(f"[MobileNetV2 Evaluator] Loading model architecture and weights from {model_path}...")

    # Build MobileNetV2 backbone (对齐ConvNeXt的模型构建逻辑)
    def build_mobilenetv2_backbone():
        base_model = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights=None  
        )

        x = layers.Flatten()(base_model.output)
        outputs = layers.Dense(1, activation='sigmoid')(x)  # 二分类sigmoid输出
        model = Model(inputs=base_model.input, outputs=outputs)
        return model

    # 初始化模型并加载权重
    model = build_mobilenetv2_backbone()
    try:
        model.load_weights(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")

    # 切换到评估模式（模拟model.eval()）
    model.compile(optimizer='adam', loss='binary_crossentropy')  # 仅为评估编译
    model.trainable = False  # 冻结所有层
    print("[MobileNetV2 Evaluator] Model loaded successfully.")

    # Step 3: Inference on Validation Set (完全对齐ConvNeXt的推理逻辑)
    print("[MobileNetV2 Evaluator] Running inference on validation set...")
    y_true = []      # True labels
    y_pred = []      # Predicted classes (0/1)
    y_scores = []    # Probability of positive class (1)

    # 批量推理（模拟PyTorch的DataLoader遍历）
    y_true = val_generator.classes  # 获取真实标签
    y_scores = model.predict(val_generator, verbose=0).flatten()  # 预测概率
    y_pred = (y_scores >= 0.5).astype(int)  # 二分类结果

    # 转换为numpy数组（对齐ConvNeXt的格式）
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    # Step 4: Calculate Evaluation Metrics (完全对齐ConvNeXt的指标计算)
    print("\n" + "="*40)
    print("Classification Report")
    print("="*40)
    # 对齐class_to_idx的键值顺序
    target_names = list(class_to_idx.keys())
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Calculate additional metrics (完全对齐ConvNeXt的指标)
    roc_auc = auc(*roc_curve(y_true, y_scores)[:2])
    pr_auc = average_precision_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Step 5: Print Summary Metrics (完全对齐ConvNeXt的输出格式)
    print("="*40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {roc_auc:.4f}")
    print(f"AUPRC Score: {pr_auc:.4f}")
    print(f"Confusion Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*40)
    
    # ================= 5. 绘制并保存 AUROC 和 AUPRC (完全对齐ConvNeXt的可视化) =================
    plt.figure(figsize=(14, 6))

    # --- 绘制 ROC 曲线 (完全对齐ConvNeXt的样式) ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # --- 绘制 Precision-Recall 曲线 (完全对齐ConvNeXt的样式) ---
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PRC)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Prepare metrics dictionary for return (完全对齐ConvNeXt的返回字段)
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(roc_auc),
        "auprc": float(pr_auc),
        "negative_f1": float(report[target_names[0]]['f1-score']),
        "positive_f1": float(report[target_names[1]]['f1-score']),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "classification_report": report
    }

    
    return metrics

if __name__ == "__main__":
    # 替换为你的实际路径
    MODEL_PATH = "/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/mobilenet_70.h5"
    VAL_DIR = "/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation"
    
    # 执行评估（输入仅model_path和val_dir，与ConvNeXt完全一致）
    metrics = evaluate_mobilenet(model_path=MODEL_PATH, val_dir=VAL_DIR)
