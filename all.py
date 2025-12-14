import warnings
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import joblib
import cv2
import pandas as pd
import xgboost as xgb
from PIL import Image
from torchvision import models, transforms
from skimage.io import imread
from skimage.transform import resize
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import StandardScaler

# ===================== 全局配置与警告屏蔽 =====================
# 精准屏蔽sklearn版本不匹配警告
warnings.filterwarnings('ignore', category=InconsistentVersionWarning, module='sklearn.base')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.base')

# 全局阈值配置
UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.65
IMG_SIZE = 224
EYE_CODE = 0
SET_CODE = 0

# ===================== 通用工具函数 =====================
def create_dir_if_not_exists(dir_path):
    """Create directory (with exception handling)"""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {dir_path}: {e}")

def get_pred_class(prob):
    """Get prediction class based on probability threshold"""
    if prob < UNCERTAIN_LOW:
        return "Glaucoma Negative"
    elif prob > UNCERTAIN_HIGH:
        return "Glaucoma Positive"
    else:
        return "Uncertain"

# ===================== 第一个版本模型 (原始版) =====================
class CNNModel_V1:
    def __init__(self, model_name, weights_root_dir):
        self.model_name = model_name
        self.weights_root_dir = weights_root_dir
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.model = self._build_model()

    def _build_model(self):
        """Build CNN model (V1 原始版)"""
        if self.model_name == "mobilenet":
            model = models.mobilenet_v2(weights=None)
            try:
                model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1))
            except:
                model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 2))
        
        elif self.model_name == "convnext":
            model = models.convnext_tiny(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        
        elif self.model_name == "densenet":
            model = models.densenet121(weights=None)
            try:
                model.classifier = nn.Linear(model.classifier.in_features, 1)
            except:
                model.classifier = nn.Linear(model.classifier.in_features, 2)
        
        elif self.model_name == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
        
        else:
            raise ValueError(f"Unsupported CNN model: {self.model_name}")
        return model.to(self.device)

    def predict_single(self, img_path):
        """Predict single image (V1 原始版)"""
        # Load weights
        model_path = os.path.join(self.weights_root_dir, f"{self.model_name}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CNN weights not found: {model_path}")
        
        # Image preprocessing
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        # Load model
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        # Prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            if output.shape[-1] == 1:
                prob = torch.sigmoid(output).item()
            else:
                prob = torch.softmax(output, dim=1)[0, 1].item()
        
        pred_class = get_pred_class(prob)
        return {"model": f"{self.model_name}_cnn", "prob": prob, "class": pred_class}

class RFModel_V1:
    def __init__(self, weights_root_dir):
        self.rf_model_dir = os.path.join(weights_root_dir, "RF")
        self.IMAGE_SIZE = (64, 64)

    def _extract_features(self, image):
        """RF feature extraction (V1 原始版)"""
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[1] > image.shape[0]:
            image = np.rot90(image)
        image = resize(image, self.IMAGE_SIZE)
        img_mean = np.mean(image, axis=(0, 1))
        img_std = np.std(image, axis=(0, 1))
        return np.concatenate((img_mean, img_std))

    def predict_single(self, img_path):
        """Predict single image (V1 原始版)"""
        # Load model and scaler
        model_path = os.path.join(self.rf_model_dir, "random_forest_model.pkl")
        scaler_path = os.path.join(self.rf_model_dir, "feature_scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"RF model files missing: {model_path} / {scaler_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Process image
        img = imread(img_path)
        features = self._extract_features(img).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        # Prediction
        prob = self.model.predict_proba(features_scaled)[:, 1][0]
        pred_class = get_pred_class(prob)
        return {"model": "rf", "prob": prob, "class": pred_class}

class SVMModel_V1:
    def __init__(self, weights_root_dir, fixed_expcdr):
        self.model_path = os.path.join(weights_root_dir, "svm.pkl")
        self.fixed_expcdr = fixed_expcdr
        self.IMAGE_SIZE = (64, 64)

    def _extract_features(self, image):
        """SVM feature extraction (V1 原始版)"""
        if len(image.shape) == 2:
            image = np.stack((image,)*3, axis=-1)
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = resize(image, self.IMAGE_SIZE)
        
        img_mean = np.mean(image, axis=(0, 1))
        img_std = np.std(image, axis=(0, 1))
        return np.concatenate(([self.fixed_expcdr], img_mean, img_std)).reshape(1, -1)

    def predict_single(self, img_path):
        """Predict single image (V1 原始版)"""
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SVM model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        
        # Process image
        img = imread(img_path)
        features = self._extract_features(img)
        
        # Prediction
        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(features)[:, 1][0]
        else:
            score = self.model.decision_function(features)[0]
            prob = (score - score.min()) / (score.max() - score.min()) if score.max() != score.min() else 0.0
        
        pred_class = get_pred_class(prob)
        return {"model": "svm", "prob": prob, "class": pred_class, "exp_cdr": self.fixed_expcdr}

class XGBModel_V1:
    def __init__(self, weights_root_dir):
        self.model_path = os.path.join(weights_root_dir, "xgb.json")

    def _extract_features(self, img_path):
        """XGB feature extraction (V1 原始版)"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            feats = {}
            # Color features
            feats['Mean_R'] = np.mean(img[:,:,0])
            feats['Mean_G'] = np.mean(img[:,:,1])
            feats['Mean_B'] = np.mean(img[:,:,2])
            feats['Std_R'] = np.std(img[:,:,0])
            
            # Central region features
            c = 60
            h, w, _ = img.shape
            center = img[h//2 - c : h//2 + c, w//2 - c : w//2 + c]
            feats['Center_R'] = np.mean(center[:,:,0])
            feats['Center_Bright_Ratio'] = np.mean(center) / (np.mean(img) + 1e-5)
            
            # Texture features
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            feats['Tex_Variance'] = np.var(gray)
            
            # Entropy calculation
            gray_norm = gray / 255.0
            gray_norm = np.clip(gray_norm, 1e-7, 1.0)
            feats['Tex_Entropy'] = -np.sum(gray_norm * np.log2(gray_norm))
            
            return feats
        except Exception as e:
            print(f"XGB feature extraction failed: {e}")
            return None

    def predict_single(self, img_path):
        """Predict single image (V1 原始版)"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"XGB model not found: {self.model_path}")
        
        # Extract features
        feats = self._extract_features(img_path)
        if feats is None:
            raise RuntimeError("XGB feature extraction failed")
        
        # Load model
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)
        
        # Prediction
        X = pd.DataFrame([feats])
        prob = self.model.predict_proba(X)[:, 1][0]
        pred_class = get_pred_class(prob)
        return {"model": "xgb", "prob": prob, "class": pred_class}

def batch_predict_single_image_v1(img_path, weights_root_dir, fixed_expcdr):
    """V1 原始版模型批量预测"""
    all_results = {}
    
    # 1. CNN models (mobilenet/convnext/densenet/resnet18)
    cnn_models = ["mobilenet", "convnext", "densenet", "resnet18"]
    for model_name in cnn_models:
        try:
            cnn = CNNModel_V1(model_name, weights_root_dir)
            res = cnn.predict_single(img_path)
            all_results[res["model"]] = res
        except Exception as e:
            print(f"[V1] {model_name} prediction failed: {str(e)}")
    
    # 2. RF model
    try:
        rf = RFModel_V1(weights_root_dir)
        res = rf.predict_single(img_path)
        all_results[res["model"]] = res
    except Exception as e:
        print(f"[V1] RF prediction failed: {str(e)}")
    
    # 3. SVM model
    try:
        svm = SVMModel_V1(weights_root_dir, fixed_expcdr)
        res = svm.predict_single(img_path)
        all_results[res["model"]] = res
    except Exception as e:
        print(f"[V1] SVM prediction failed: {str(e)}")
    
    # 4. XGB model
    try:
        xgb_clf = XGBModel_V1(weights_root_dir)
        res = xgb_clf.predict_single(img_path)
        all_results[res["model"]] = res
    except Exception as e:
        print(f"[V1] XGB prediction failed: {str(e)}")
    
    # Print V1 summary
    print("\n" + "="*80)
    print("V1 - Original Models Prediction Summary")
    print("="*80)
    model_order = [
        "mobilenet_cnn", "convnext_cnn", "densenet_cnn", 
        "resnet18_cnn", "rf", "svm", "xgb"
    ]
    for model_name in model_order:
        if model_name in all_results:
            res = all_results[model_name]
            print(f"{model_name:20s} | Probability: {res['prob']:.4f} | Result: {res['class']}")
    print("="*80)
    
    return all_results

# ===================== 第二个版本模型 (集成版) =====================
def predict_glaucoma_v2(img_path, weights_root_dir, fixed_expcdr):
    """V2 集成版模型预测"""
    # ===================== 自动拼接所有模型权重路径 =====================
    CONVNEXT_MODEL_PATH = os.path.join(weights_root_dir, "convnext_integrated.pth")
    DENSENET_MODEL_PATH = os.path.join(weights_root_dir, "densenet_integrated.pth")
    MOBILENET_MODEL_PATH = os.path.join(weights_root_dir, "mobilenet_integrated.pth")
    RESNET18_MODEL_PATH = os.path.join(weights_root_dir, "resnet18_integrated.pth")
    RF_MODEL_DIR = os.path.join(weights_root_dir, "RF_integrated")
    SVM_MODEL_PATH = os.path.join(weights_root_dir, "svm_integrated.pkl")
    XGB_MODEL_PATH = os.path.join(weights_root_dir, "xgb_integrated.json")

    # ===================== 设备配置 =====================
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # ===================== 图像预处理 =====================
    cnn_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert("RGB")
    cnn_image_tensor = cnn_transform(image).unsqueeze(0).to(device)

    # ===================== 1. ConvNeXt + ExpCDR =====================
    class FusionConvNext(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = models.convnext_tiny(weights=None)
            n_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()
            self.fusion_head = nn.Sequential(
                nn.Linear(n_features + 1, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 2)
            )

        def forward(self, img, cdr):
            feat = self.backbone(img)
            return self.fusion_head(torch.cat((feat, cdr), dim=1))

    convnext = FusionConvNext().to(device)
    convnext.load_state_dict(torch.load(CONVNEXT_MODEL_PATH, map_location=device))
    convnext.eval()

    cdr_tensor = torch.tensor([[fixed_expcdr]], dtype=torch.float32).to(device)
    with torch.no_grad():
        out = convnext(cnn_image_tensor, cdr_tensor)
        p_cn = torch.softmax(out, dim=1)[0, 1].item()

    # ===================== 2. DenseNet + ExpCDR =====================
    class HybridDenseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = models.densenet121(weights=None).features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.tab_net = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
            self.classifier = nn.Sequential(
                nn.Linear(1024 + 16, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )

        def forward(self, img, tab):
            x1 = torch.flatten(self.pool(self.cnn(img)), 1)
            x2 = self.tab_net(tab)
            return self.classifier(torch.cat((x1, x2), dim=1))

    densenet = HybridDenseNet().to(device)
    densenet.load_state_dict(torch.load(DENSENET_MODEL_PATH, map_location=device))
    densenet.eval()

    tab_tensor = torch.tensor([[fixed_expcdr, 0, 0]], dtype=torch.float32).to(device)
    with torch.no_grad():
        out = densenet(cnn_image_tensor, tab_tensor)
        p_dn = torch.sigmoid(out).item()

    # ===================== 3. MobileNet + ExpCDR =====================
    class HybridMobileNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = models.mobilenet_v2(weights=None)
            img_feat_dim = self.cnn.last_channel
            self.tab_net = nn.Sequential(nn.Linear(3, 16), nn.ReLU())
            self.classifier = nn.Sequential(
                nn.Linear(img_feat_dim + 16, 150),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(150, 50),
                nn.ReLU(),
                nn.Linear(50, 1)
            )

        def forward(self, img, tab):
            x_img = self.cnn.features(img).mean([2, 3])
            x_tab = self.tab_net(tab)
            x = torch.cat((x_img, x_tab), dim=1)
            return self.classifier(x)

    mobilenet = HybridMobileNet().to(device)
    mobilenet.load_state_dict(torch.load(MOBILENET_MODEL_PATH, map_location=device))
    mobilenet.eval()
    
    tab_tensor_mb = torch.tensor([[fixed_expcdr, 0.0, 0.0]], dtype=torch.float32).to(device)
    with torch.no_grad():
        out = mobilenet(cnn_image_tensor, tab_tensor_mb)
        p_mb = torch.sigmoid(out).item()

    # ===================== 4. ResNet18 + ExpCDR =====================
    class FusionResNet18(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.backbone = models.resnet18(weights=None)
            n_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.fusion_head = nn.Sequential(
                nn.Linear(n_features + 1, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

        def forward(self, image, cdr):
            img_feat = self.backbone(image)
            combined_feat = torch.cat((img_feat, cdr), dim=1)
            return self.fusion_head(combined_feat)

    resnet18 = FusionResNet18(num_classes=2).to(device)
    state_dict = torch.load(RESNET18_MODEL_PATH, map_location=device)
    resnet18.load_state_dict(state_dict)
    resnet18.eval()

    cdr_tensor_rn = torch.tensor([[fixed_expcdr]], dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs_rn = resnet18(cnn_image_tensor, cdr_tensor_rn)
        probs_rn = torch.softmax(outputs_rn, dim=1)
        p_rn = probs_rn[:, 1].cpu().numpy()[0]

    # ===================== 5. Random Forest =====================
    RF_IMAGE_SIZE = (64, 64)
    def get_rf_features(image_path, exp_cdr):
        img = imread(image_path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[1] > img.shape[0]:
            img = np.rot90(img)
        img = resize(img, RF_IMAGE_SIZE)
        img_mean = np.mean(img, axis=(0, 1))
        img_std = np.std(img, axis=(0, 1))
        img_features = np.concatenate((img_mean, img_std))
        return np.concatenate(([exp_cdr], img_features))

    rf_model_path = os.path.join(RF_MODEL_DIR, "random_forest_csv_model.pkl")
    rf_scaler_path = os.path.join(RF_MODEL_DIR, "feature_scaler_csv.pkl")

    with open(rf_model_path, 'rb') as f:
        rf_model = pickle.load(f)
    with open(rf_scaler_path, 'rb') as f:
        rf_scaler = pickle.load(f)

    rf_feat = get_rf_features(img_path, fixed_expcdr)
    rf_feat = np.expand_dims(rf_feat, axis=0)
    rf_feat_scaled = rf_scaler.transform(rf_feat)
    p_rf = float(rf_model.predict_proba(rf_feat_scaled)[:, 1][0])

    # ===================== 6. SVM =====================
    def get_svm_features(image_path, exp_cdr):
        img = imread(image_path)
        img = resize(img, (64, 64))
        if len(img.shape) == 2:
            img = np.stack((img,)*3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        means = np.mean(img, axis=(0, 1))
        stds = np.std(img, axis=(0, 1))
        return np.concatenate(([exp_cdr], means, stds))

    svm_pipeline = joblib.load(SVM_MODEL_PATH)
    svm_feat = get_svm_features(img_path, fixed_expcdr)
    svm_feat = np.expand_dims(svm_feat, axis=0)

    with np.errstate(all='ignore'):
        if hasattr(svm_pipeline, "predict_proba"):
            p_svm = float(svm_pipeline.predict_proba(svm_feat)[:, 1][0])
        else:
            svm_decision = svm_pipeline.decision_function(svm_feat)[0]
            p_svm = (svm_decision - svm_decision.min()) / (svm_decision.max() - svm_decision.min()) if svm_decision.max() != svm_decision.min() else 0.0

    # ===================== 7. XGBoost =====================
    def extract_xgb_features(image_path, exp_cdr):
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        features = {}
        features['Mean_R'] = np.mean(img[:, :, 0])
        features['Mean_G'] = np.mean(img[:, :, 1])
        features['Mean_B'] = np.mean(img[:, :, 2])
        features['Std_R'] = np.std(img[:, :, 0])
        
        c = 60
        h, w, _ = img.shape
        center = img[h//2 - c : h//2 + c, w//2 - c : w//2 + c]
        features['Center_R'] = np.mean(center[:, :, 0])
        features['Center_Bright_Ratio'] = np.mean(center) / (np.mean(img) + 1e-5)
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        features['Tex_Variance'] = np.var(gray)
        gray_norm = gray / 255.0
        gray_norm = np.clip(gray_norm, 1e-7, 1.0)
        features['Tex_Entropy'] = -np.sum(gray_norm * np.log2(gray_norm))
        
        features['ExpCDR'] = exp_cdr
        features['eye'] = EYE_CODE
        features['set'] = SET_CODE
        
        feature_order = [
            'ExpCDR', 'eye', 'set', 'Mean_R', 'Mean_G', 'Mean_B', 
            'Std_R', 'Center_R', 'Center_Bright_Ratio', 'Tex_Variance', 'Tex_Entropy'
        ]
        return np.array([features[fn] for fn in feature_order])

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)

    xgb_feat = extract_xgb_features(img_path, fixed_expcdr)
    xgb_feat = np.expand_dims(xgb_feat, axis=0)
    p_xgb = float(xgb_model.predict_proba(xgb_feat)[:, 1][0])

    # ===================== 整理所有模型结果 =====================
    all_results = {
        "mobilenet_integrated": {"prob": p_mb, "result": get_pred_class(p_mb)},
        "convnext_integrated": {"prob": p_cn, "result": get_pred_class(p_cn)},
        "densenet_integrated": {"prob": p_dn, "result": get_pred_class(p_dn)},
        "resnet18_integrated": {"prob": p_rn, "result": get_pred_class(p_rn)},
        "rf_integrated": {"prob": p_rf, "result": get_pred_class(p_rf)},
        "svm_integrated": {"prob": p_svm, "result": get_pred_class(p_svm)},
        "xgb_integrated": {"prob": p_xgb, "result": get_pred_class(p_xgb)}
    }

    # ===================== 格式化输出 =====================
    print("\n" + "="*80)
    print("V2 - Integrated Models Prediction Summary")
    print("="*80)
    for model_name, data in all_results.items():
        print(f"{model_name:<20} | Probability: {data['prob']:.4f} | Result: {data['result']}")
    print("="*80)

    return all_results

# ===================== 主函数 (命令行入口) =====================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Glaucoma Prediction with Dual Version Models')
    parser.add_argument('--expcdr', type=float, required=True, help='ExpCDR value (e.g., 0.4803)')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--weights_dir', type=str, required=True, help='Root directory of model weights')
    
    args = parser.parse_args()

    # 验证输入参数
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    if not os.path.isdir(args.weights_dir):
        raise NotADirectoryError(f"Weights directory not found: {args.weights_dir}")

    # 运行V1版本预测
    print("Running V1 (Original) Models Prediction...")
    v1_results = batch_predict_single_image_v1(args.image_path, args.weights_dir, args.expcdr)
    
    # 运行V2版本预测
    print("\nRunning V2 (Integrated) Models Prediction...")
    v2_results = predict_glaucoma_v2(args.image_path, args.weights_dir, args.expcdr)

    # 返回合并结果
    final_results = {
        "v1_original": v1_results,
        "v2_integrated": v2_results,
        "input_params": {
            "expcdr": args.expcdr,
            "image_path": args.image_path,
            "weights_dir": args.weights_dir
        }
    }

    return final_results

if __name__ == "__main__":
    main()