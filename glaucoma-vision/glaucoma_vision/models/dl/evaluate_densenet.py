import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
)

def evaluate_densenet(model_path: str, test_dir: str, img_size: int = 224, show_plots: bool = True):
    """
    Evaluate DenseNet121 model for glaucoma binary classification
    Args:
        model_path: Path to the trained DenseNet model weights
        test_dir: Path to the test dataset directory (contains Glaucoma_Negative/Positive subfolders)
        img_size: Image size for preprocessing (default: 224)
        show_plots: Whether to display AUROC/AUPRC plots (default: True)
    Returns:
        dict: Evaluation metrics including accuracy, F1, AUROC, AUPRC and confusion matrix
    """
    # ===================== 1. Device Configuration =====================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DenseNet Evaluator] Using device: {DEVICE}")

    # ===================== 2. Model Loading =====================
    print(f"[DenseNet Evaluator] Loading model from {model_path}...")
    # Initialize DenseNet121 (match training architecture)
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1)
    )
    
    # Load weights and set to evaluation mode
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("[DenseNet Evaluator] Model loaded successfully.")

    # ===================== 3. Image Preprocessing =====================
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ===================== 4. Inference on Test Set =====================
    y_true = []
    y_prob = []
    class_names = ['Glaucoma_Negative', 'Glaucoma_Positive']
    print(f"[DenseNet Evaluator] Running inference on test set: {test_dir}")

    # Iterate through negative/positive folders
    for label, folder_name in enumerate(class_names):
        folder_path = os.path.join(test_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found, skipping...")
            continue
        
        # Iterate through all images in folder
        for img_name in os.listdir(folder_path):
            if not img_name.lower().endswith('.jpg'):
                continue
            
            # Read and preprocess image
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            # Inference
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.sigmoid(output).item()  # Probability of positive class
            
            # Collect results
            y_true.append(label)
            y_prob.append(prob)

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)

    # ===================== 5. Generate Classification Report =====================
    print("\n" + "="*40)
    print("Classification Report")
    print("="*40)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4  # 保持小数位数一致
    )
    print(report)

    # ===================== 6. Calculate Summary Metrics =====================
    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Print summary metrics (match ConvNeXt format)
    print("="*40)
    print(f"Summary Metrics:")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"AUROC Score: {auroc:.4f}")
    print(f"AUPRC Score: {auprc:.4f}")
    print(f"Confusion Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*40)

    # ===================== 7. Plot AUROC and AUPRC Curves (一行两列) =====================
    if show_plots and len(y_true) > 0:
        # 创建一行两列的画布，两张图并排展示
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- 左图：ROC Curve ---
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC)')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # --- 右图：Precision-Recall Curve ---
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax2.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve (PRC)')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        # 调整布局，避免重叠
        plt.tight_layout()
        plt.show()

    # ===================== 8. Prepare Metrics Dictionary =====================
    metrics = {
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred
    }

    return metrics

# Example usage (uncomment to test)
if __name__ == "__main__":
    MODEL_PATH = '/Users/apple/Desktop/Glaucoma-binary-classification-model-BIA-Group-Work-/glaucoma-vision/glaucoma_vision/models/weights/densenet.pth'
    TEST_DIR = '/Users/apple/Desktop/BIA 4/Fundus_Scanes_Sorted/Validation'
    metrics = evaluate_densenet(MODEL_PATH, TEST_DIR)


import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score, average_precision_score, precision_recall_curve

def densenet_integrate(MODEL_PATH, BASE_PATH, CSV_PATH):
    """
    封装后的DenseNet混合模型评估函数
    参数:
        MODEL_PATH: 模型权重文件路径
        BASE_PATH: 验证集根目录 (Fundus_Scanes_Sorted/Validation)
        CSV_PATH: 青光眼CSV数据文件路径
    返回:
        dict: 评估结果字典
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hybrid Evaluator] Loading Model from {MODEL_PATH}...")
    print(f"[Hybrid Evaluator] Validation Path: {BASE_PATH}")

    # 定义文件夹映射（修复self关键字问题）
    folder_map = {0: 'Glaucoma_Negative', 1: 'Glaucoma_Positive'}

    # 1. Dataset Class (适配BASE_PATH为Validation目录)
    class HybridDataset(Dataset):
        def __init__(self, df, val_path, transform=None):
            self.df = df.reset_index(drop=True)  # 重置索引避免越界
            self.val_path = val_path  # 直接使用Validation根目录
            self.transform = transform
            self.folder_map = folder_map  # 使用外层定义的映射
            self.df['eye_code'] = self.df['Eye'].map({'OD': 0, 'OS': 1}).fillna(0)
            self.df['set_code'] = self.df['Set'].map({'A': 0, 'B': 1}).fillna(0)
            self.tabular_cols = ['ExpCDR', 'eye_code', 'set_code']

        def __len__(self): return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            label = row['Glaucoma']
            # 直接从Validation目录拼接路径 (适配BASE_PATH=Validation)
            path = os.path.join(self.val_path, self.folder_map[label], row['Filename'])
            
            # 路径检查 & 图片读取
            if not os.path.exists(path):
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.imread(path)
                if img is None:
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                img = self.transform(img)
            
            tab = torch.tensor(row[self.tabular_cols].values.astype(float), dtype=torch.float32)
            return img, tab, torch.tensor(label, dtype=torch.float32)

    # 2. Model Class (保持不变)
    class HybridDenseNet(nn.Module):
        def __init__(self):
            super(HybridDenseNet, self).__init__()
            self.cnn = models.densenet121(weights=None).features
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.tab_net = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
            self.classifier = nn.Sequential(
                nn.Linear(1024 + 16, 64), 
                nn.ReLU(), 
                nn.Dropout(0.3), 
                nn.Linear(64, 1)
            )

        def forward(self, img, tab):
            x1 = torch.flatten(self.global_pool(self.cnn(img)), 1)
            x2 = self.tab_net(tab)
            return self.classifier(torch.cat((x1, x2), dim=1))

    # 3. Hybrid Grad-CAM Helper (仅修复梯度计算)
    class HybridGradCAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target = target_layer
            self.grads = None
            self.acts = None
            self.target.register_forward_hook(self.save_act)
            self.target.register_full_backward_hook(self.save_grad)

        def save_act(self, m, i, o): self.acts = o
        def save_grad(self, m, gi, go): self.grads = go[0]

        def __call__(self, img, tab):
            self.model.eval()
            with torch.enable_grad():  # 启用梯度计算
                out = self.model(img, tab)
                self.model.zero_grad()
                out.backward()  # 修复反向传播

            w = torch.mean(self.grads[0], dim=(1, 2), keepdim=True)
            cam = torch.sum(w * self.acts[0], dim=0)
            cam = torch.nn.functional.relu(cam)
            cam = cam.detach().cpu().numpy()
            return cam / cam.max() if cam.max() > 0 else cam, torch.sigmoid(out).item()

    # ==================== 数据加载 ====================
    # 加载并清洗CSV数据
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['Filename', 'ExpCDR', 'Glaucoma'])
    
    # 筛选仅属于验证集的样本 (匹配BASE_PATH=Validation)
    # 过滤出CSV中存在于Validation目录的样本
    valid_filenames = []
    for label in [0, 1]:
        # 修复：使用外层定义的folder_map，而非self.folder_map
        folder = os.path.join(BASE_PATH, folder_map[label])
        if os.path.exists(folder):
            valid_filenames.extend([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    df = df[df['Filename'].isin(valid_filenames)].reset_index(drop=True)
    print(f"[Hybrid Evaluator] Valid samples in Validation set: {len(df)}")

    # 数据变换
    tfms = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集和加载器 (直接使用Validation目录)
    test_dataset = HybridDataset(df, BASE_PATH, tfms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ==================== 模型加载 ====================
    model = HybridDenseNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("Model not found!")
        return None

    model.eval()

    # ==================== 模型推理 ====================
    y_true, y_prob = [], []
    with torch.no_grad():
        for img, tab, lbl in test_loader:
            img, tab = img.to(DEVICE), tab.to(DEVICE)
            out = model(img, tab)
            y_prob.append(torch.sigmoid(out).item())
            y_true.append(lbl.item())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)

    # ==================== 评估指标计算 ====================
    # 处理空数据情况
    if len(y_true) == 0:
        print("No valid samples for evaluation!")
        return None
    
    report = classification_report(
        y_true, y_pred, 
        target_names=['Negative', 'Positive'], 
        output_dict=True, 
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size >=4 else (0,0,0,0)

    # 核心指标计算
    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_prob)

    # 打印评估报告
    print("\n" + "="*50)
    print("HYBRID MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Glaucoma_Negative F1 score : {report['Negative']['f1-score']:.4f}")
    print(f"Glaucoma_Positive F1 score : {report['Positive']['f1-score']:.4f}")
    print(f"Accuracy                     : {accuracy:.4f}")
    print(f"AUROC Score                  : {auroc:.4f}")
    print(f"AUPRC Score                  : {auprc:.4f}")
    print("-" * 50)
    print(f"Confusion Matrix -> TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print("="*50 + "\n")

    # ==================== 可视化 ====================
    # 1. 混淆矩阵 + ROC + PR曲线
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title('Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('True')

    # ROC曲线
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax[1].plot(fpr, tpr, color='orange', lw=2, label=f'AUC={auroc:.4f}')
    ax[1].plot([0,1],[0,1],'k--')
    ax[1].legend()
    ax[1].set_title('ROC Curve')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')

    # PR曲线
    if len(np.unique(y_true)) > 1:
        p, r, _ = precision_recall_curve(y_true, y_prob)
        ax[2].plot(r, p, color='green', lw=2)
    ax[2].set_title('PR Curve')
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
    plt.tight_layout()
    plt.show()

    # 2. Grad-CAM可视化
    print("\n[Evaluator] Generating Hybrid Grad-CAM...")
    target_layer = model.cnn.denseblock4.denselayer16.conv2
    cam_tool = HybridGradCAM(model, target_layer)

    # 选择TP样本可视化
    tp_idx = [i for i, (t,p) in enumerate(zip(y_true, y_pred)) if t==1 and p==1][:3]

    if tp_idx:
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(tp_idx):
            ds = test_loader.dataset
            img, tab, lbl = ds[idx]

            # 生成Grad-CAM
            heatmap, prob = cam_tool(img.unsqueeze(0).to(DEVICE), tab.unsqueeze(0).to(DEVICE))

            # 反归一化图片
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img.permute(1, 2, 0).numpy() * std + mean
            img_np = np.clip(img_np, 0, 1)
            img_u8 = (img_np * 255).astype(np.uint8)

            # 叠加热力图
            hm = cv2.resize(heatmap, (224, 224))
            hm = cv2.applyColorMap(np.uint8(255*hm), cv2.COLORMAP_JET)
            over = cv2.addWeighted(img_u8, 0.6, hm, 0.4, 0)

            plt.subplot(1, 3, i+1)
            plt.imshow(over)
            plt.axis('off')
            plt.title(f"ExpCDR: {tab[0]:.2f} | Prob: {prob:.2f}")
        
        plt.suptitle("Hybrid Grad-CAM (Combines Image & Tabular Influence)")
        plt.tight_layout()
        plt.show()

    # ==================== 返回评估结果 ====================
    results = {
        'accuracy': accuracy,
        'auroc': auroc,
        'auprc': auprc,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    return results

