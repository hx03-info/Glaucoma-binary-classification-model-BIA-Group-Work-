"""
Deep Learning Model Evaluation Utilities
For Glaucoma Detection (ResNet18/DenseNet121/ConvNeXt-Tiny)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2

def get_device() -> torch.device:
    """Get available device (GPU first, then CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_dl_image(img_path: str, img_size: tuple = (224, 224)) -> np.ndarray | None:
    try:
        # Read and convert color space
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        
        # Normalize to 0-1 and transpose to CHW (PyTorch format)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        
        return img
    except Exception as e:
        print(f"DL Image preprocessing failed for {img_path}: {str(e)}")
        return None


def load_dl_model(
    model_name: str,
    weights_path: str,
    num_classes: int = 2
) -> nn.Module:
    device = get_device()
    
    if model_name.lower() == "resnet18":
        model = models.resnet18(pretrained=False)
        # Replace final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name.lower() == "densenet121":
        model = models.densenet121(pretrained=False)
        # Replace classifier layer
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name.lower() == "convnext_tiny":
        model = models.convnext_tiny(pretrained=False)
        # Replace final classifier layer
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Use resnet18/densenet121/convnext_tiny")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location=device)
    # Handle both state_dict and direct model weights
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded {model_name} model from {weights_path} (Device: {device})")
    return model

# ==================== Custom DL Dataset ====================
class GlaucomaDLDataset(Dataset):
    """Custom Dataset for Glaucoma DL Model Evaluation"""
    def __init__(self, val_dir: str, csv_df: pd.DataFrame, img_size: tuple = (224, 224)):
        self.val_dir = val_dir
        self.img_size = img_size
        self.class_map = {0: 'Glaucoma_Negative', 1: 'Glaucoma_Positive'}
        
        # Build image path + label list
        self.data = []
        for _, row in csv_df.iterrows():
            class_name = self.class_map[row['Glaucoma']]
            img_path = os.path.join(val_dir, class_name, row['Filename'])
            if os.path.exists(img_path):
                self.data.append((img_path, row['Glaucoma']))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.data[idx]
        
        # Preprocess image
        img = preprocess_dl_image(img_path, self.img_size)
        if img is None:
            raise ValueError(f"Invalid image at index {idx}: {img_path}")
        
        # Convert to torch tensor
        img_tensor = torch.tensor(img, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return img_tensor, label_tensor

# ==================== DL Model Inference ====================
def dl_model_inference(
    model: nn.Module,
    val_dir: str,
    csv_df: pd.DataFrame,
    img_size: tuple = (224, 224),
    batch_size: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run batch inference with DL model
    :param model: Loaded DL model (eval mode)
    :param val_dir: Validation directory path
    :param csv_df: Loaded CSV DataFrame (Filename/Glaucoma columns)
    :param img_size: Model input size
    :param batch_size: Inference batch size
    :return: (y_true: np.ndarray, y_scores: np.ndarray)
             - y_true: True labels
             - y_scores: Probability of positive class (glaucoma)
    """
    device = get_device()
    
    # Create dataset and dataloader
    dataset = GlaucomaDLDataset(val_dir, csv_df, img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if device.type == "cuda" else False
    )
    
    print(f"📊 Running DL model inference (Batch size: {batch_size}, Samples: {len(dataset)})")
    
    # Collect predictions
    y_true = []
    y_scores = []
    
    # Inference with no gradient computation
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="DL Inference Progress"):
            # Move data to device
            imgs = imgs.to(device)
            
            # Forward pass
            outputs = model(imgs)
            
            # Get positive class probability (softmax)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            # Collect results (convert to numpy)
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    print(f"✅ Inference complete - Collected {len(y_true)} samples")
    return y_true, y_scores
