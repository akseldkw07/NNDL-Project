import typing as t
from pathlib import Path

import torch

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "img_data"
MODEL_WEIGHT_DIR = ROOT_DIR / "model_weights"

# WANDB
WANDB_PROJECT_NAME = "nndl-project"

# DEVICE
DEVICE_LITERAL = t.Literal["cuda", "mps", "xpu", "cpu"]  # extend to include "xla", "xpu" if needed


def pick_device() -> DEVICE_LITERAL:
    if torch.cuda.is_available():
        return "cuda"
    # If you plan to use TPUs:
    if torch.backends.mps.is_available():
        return "mps"
    # If using Intel GPUs:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


DEVICE_TORCH_STR: DEVICE_LITERAL = pick_device()
DEVICE = torch.device(DEVICE_TORCH_STR)


# IMAGES
DEF_IMAGE_SIZE = 64  # 64x64 images
MEAN_IMG = (0.4170, 0.3801, 0.3132)  # computed over training set - notebooks/Exploration.ipynb
STD_IMG = (0.2131, 0.2014, 0.1924)  # computed over training set - notebooks/Exploration.ipynb
