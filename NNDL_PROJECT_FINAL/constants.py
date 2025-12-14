import os
import typing as t
from pathlib import Path

import torch

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "img_data"
MODEL_WEIGHT_DIR = ROOT_DIR / "model_weights"

# WANDB
WANDB_TEAM_NAME = "nndl-project-F25"
WANDB_PROJECT_NAME = "Multihead-Classification-Competition"

# DEVICE
DEVICE_LITERAL = t.Literal["cuda", "mps", "xpu", "cpu"]  # extend to include "xla", "xpu" if needed


def pick_device() -> DEVICE_LITERAL:
    if torch.cuda.is_available():
        return "cuda"
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
MEAN_IMG = [0.485, 0.456, 0.406]
STD_IMG = [0.229, 0.224, 0.225]  # computed over training set - notebooks/Exploration.ipynb

# =============================================================
# Christian params
# =============================================================

DATA_ROOT = "/content/drive/MyDrive/NNDL-Project/Project Data"

# Local scratch space on the VM / Colab
LOCAL_DATA_ROOT = "/content/local_data"


TRAIN_CSV = os.path.join(DATA_ROOT, "train_data.csv")
SUPER_CSV = os.path.join(DATA_ROOT, "superclass_mapping.csv")
SUB_CSV = os.path.join(DATA_ROOT, "subclass_mapping.csv")

BATCH_SIZE = 64
NUM_WORKERS = 2  # can set to 0 if we hit dataloader issues

VAL_SPLIT = 0.1  # 10% validation
IMG_SIZE = 64  # our image dimensions

PROJECT_NAME = "coms4776-transfer-learning"  # TBD update
APPROACH = "two_models"  # "two_heads" or "two_models"
DATA_AUGMENT = True

# Indices for "novel" classes (per provided data)
NOVEL_SUPER_IDX = 3  # superclass index for novel
NOVEL_SUB_IDX = 87  # subclass index for novel

# Number of times run full-batch
EPOCHS = 25

# Learning rates
LR = 1e-4  # overall learning rate
LR_HEAD = 1e-2  # head learning rate, used when freezing backbone
WEIGHT_DECAY = 1e-4  # seems standard
BACKBONE = "resnet18"  # "resnet18" or "resnet50"

# Novel-super CIFAR integration (more images)
# Options: "none", "small" (~1000 samples), "large" (~5000 samples)
CIFAR_NOVEL_MODE = "large"  # "large" or "small" or "none"

# Path to store metadata about CIFAR novel images
CIFAR_NOVEL_CSV_PATH = os.path.join(LOCAL_DATA_ROOT, "cifar_novel_data.csv")

# Fine-tuning mode for ResNet backbone
# "full"   = train all layers (what you're currently doing)
# "frozen" = freeze backbone, train only the heads on top
FINE_TUNE_MODE = "frozen"  # or "frozen"

# Initial novelty thresholds (starting points, will tune further)
TAU_SUPER = (
    0.99  # NOTE: per calibration with validation data. if max superclass prob < TAU_SUPER -> predict novel superclass
)
TAU_SUB = 0.73  # NOTE: per calibration with validation data. if max subclass prob < TAU_SUB  -> predict novel subclass
SUPER_HEAD_TYPE = "linear"  # "linear" or "cosine"
SUB_HEAD_TYPE = "cosine"  # "linear" or "cosine"
ALPHA_SUPER_CONSISTENCY = 0.2

########### MAKE SURE USE_PSEDUO_NOVEL IS FALSE BEFORE LEADERBOARD SUBMISSION ##################################################
USE_PSEUDO_NOVEL = True  # to validate on held-out subclasses from training. Used to fine-tune TAU_SUB
PSEUDO_NOVEL_FRACTION = 0.15
PSEUDO_NOVEL_SEED = 123


####################################################################################################################################
ALLOWED_FINE_NAMES = {
    # aquatic mammals
    "beaver",
    "dolphin",
    "otter",
    "seal",
    "whale",
    # fish
    "aquarium_fish",
    "flatfish",
    "ray",
    "shark",
    "trout",
    # flowers
    "orchid",
    "poppy",
    "rose",
    "sunflower",
    "tulip",
    # food containers
    "bottle",
    "bowl",
    "can",
    "cup",
    "plate",
    # fruit and vegetables
    "apple",
    "mushroom",
    "orange",
    "pear",
    "sweet_pepper",
    # household electrical devices
    "clock",
    "keyboard",
    "lamp",
    "telephone",
    "television",
    # household furniture
    "bed",
    "chair",
    "couch",
    "table",
    "wardrobe",
    # insects
    "bee",
    "beetle",
    "butterfly",
    "caterpillar",
    "cockroach",
    # large carnivores
    "bear",
    "leopard",
    "lion",
    "tiger",
    "wolf",
    # large man-made outdoor things
    "bridge",
    "castle",
    "house",
    "road",
    "skyscraper",
    # large natural outdoor scenes
    "cloud",
    "forest",
    "mountain",
    "plain",
    "sea",
    # large omnivores and herbivores
    "camel",
    "cattle",
    "chimpanzee",
    "elephant",
    "kangaroo",
    # medium-sized mammals
    "fox",
    "porcupine",
    "possum",
    "raccoon",
    "skunk",
    # non-insect invertebrates
    "crab",
    "lobster",
    "snail",
    "spider",
    "worm",
    # people
    "baby",
    "boy",
    "girl",
    "man",
    "woman",
    # small mammals
    "hamster",
    "mouse",
    "rabbit",
    "shrew",
    "squirrel",
    # trees
    "maple",
    "oak",
    "palm",
    "pine",
    "willow",
    # vehicles 1
    "bicycle",
    "bus",
    "motorcycle",
    "pickup_truck",
    "train",
    # vehicles 2
    "lawn_mower",
    "rocket",
    "streetcar",
    "tank",
    "tractor",
    # NOTE: reptiles group ("crocodile", "dinosaur", "lizard", "snake", "turtle") is *excluded* on purpose
}
