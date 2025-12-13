import sys
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

# ===============================
# Parse arguments
# ===============================
parser = argparse.ArgumentParser(
    description="Glaucoma model evaluation (external weights supported)"
)

parser.add_argument(
    "--val_dir",
    required=True,
    help="Path to validation image directory"
)

parser.add_argument(
    "--weights_dir",
    required=True,
    help="Directory containing model weights"
)

parser.add_argument(
    "--model",
    required=True,
    choices=["convnext", "densenet", "resnet18", "mobilenet", "rf", "svm", "xgb"],
    help="Model name"
)

parser.add_argument(
    "--integrated",
    action="store_true",
    help="Use ExpCDR integrated model"
)

args = parser.parse_args()

# ===============================
# Auto path inference
# ===============================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

REPO_PATH = os.path.join(PROJECT_ROOT, "glaucoma-vision")
CSV_PATH = os.path.join(PROJECT_ROOT, "glaucoma.csv")
SAVE_DIR = os.path.join(PROJECT_ROOT, "ICA")

os.makedirs(SAVE_DIR, exist_ok=True)
sys.path.append(REPO_PATH)

# ===============================
# Imports
# ===============================
from glaucoma_vision.models.evaluate_convnext import evaluate_convnext, convnext_integrate
from glaucoma_vision.models.evaluate_densenet import evaluate_densenet, densenet_integrate
from glaucoma_vision.models.evaluate_resnet18 import evaluate_resnet18, resnet18_integrate
from glaucoma_vision.models.evaluate_mobilenet import evaluate_mobilenet, mobilenet_integrate
from glaucoma_vision.models.evaluate_rf import evaluate_rf, rf_integrate
from glaucoma_vision.models.evaluate_svm import evaluate_svm, svm_integrate
from glaucoma_vision.models.evaluate_xgb import evaluate_xgb, xgb_integrate

# ===============================
# Dispatcher
# ===============================
def run():
    W = args.weights_dir
    V = args.val_dir
    C = CSV_PATH
    S = SAVE_DIR

    print("\n==============================")
    print(f"Model      : {args.model}")
    print(f"Integrated : {args.integrated}")
    print(f"Weights    : {W}")
    print("==============================\n")

    if args.model == "convnext":
        return convnext_integrate(os.path.join(W, "convnext_integrated.pth"), V, C, S) \
            if args.integrated else \
            evaluate_convnext(os.path.join(W, "convnext.pth"), V, S)

    if args.model == "densenet":
        return densenet_integrate(os.path.join(W, "densenet_integrated.pth"), V, C, S) \
            if args.integrated else \
            evaluate_densenet(os.path.join(W, "densenet.pth"), V, S)

    if args.model == "resnet18":
        return resnet18_integrate(os.path.join(W, "resnet18_integrated.pth"), V, C, S) \
            if args.integrated else \
            evaluate_resnet18(os.path.join(W, "resnet18.pth"), V, S)

    if args.model == "mobilenet":
        return mobilenet_integrate(os.path.join(W, "mobilenet_integrated.pth"), V, C, S) \
            if args.integrated else \
            evaluate_mobilenet(os.path.join(W, "mobilenet.pth"), V, S)

    if args.model == "rf":
        return rf_integrate(os.path.join(W, "RF_integrated"), V, S, C) \
            if args.integrated else \
            evaluate_rf(os.path.join(W, "RF"), V, S)

    if args.model == "svm":
        return svm_integrate(os.path.join(W, "svm_integrated.pkl"), V, C, S) \
            if args.integrated else \
            evaluate_svm(os.path.join(W, "svm.pkl"), C, V, S)

    if args.model == "xgb":
        return xgb_integrate(os.path.join(W, "xgb_integrated.json"), V, C, S) \
            if args.integrated else \
            evaluate_xgb(os.path.join(W, "xgb.json"), V, C, S)


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    results = run()


