from pathlib import Path
import torch
import typing as t

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "img_data"
MODEL_WEIGHT_DIR = ROOT_DIR / "model_weights"

# DEVICE
DEVICE_LITERAL = t.Literal["cuda", "mps", "cpu"]
DEVICE_TORCH_STR: DEVICE_LITERAL
if torch.cuda.is_available():
    DEVICE_TORCH_STR = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_TORCH_STR = "mps"
else:
    DEVICE_TORCH_STR = "cpu"
DEVICE = torch.device(DEVICE_TORCH_STR)
