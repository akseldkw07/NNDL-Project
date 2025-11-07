from __future__ import annotations
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
from torchvision import transforms as T


# ----------------------------
# Configurable column names
# ----------------------------
COL_IMAGE = "image"  # e.g., "123.jpg"
COL_SUPER = "superclass"  # integer id in [0..S-1]
COL_SUB = "subclass"  # integer id in [0..K-1]


class HierImageDataset(Dataset):
    """
    CSV-driven image dataset that returns:
      (image_tensor, {'super': y_super, 'sub': y_sub})   for train/val
      (image_tensor, image_id)                           for test (when labels_csv is None)
    """

    def __init__(
        self,
        images_dir: Path,
        labels_csv: Path | None,
        transform: T.Compose | None = None,
        is_test: bool = False,
        id_column: str = COL_IMAGE,
        col_super: str = COL_SUPER,
        col_sub: str = COL_SUB,
    ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.is_test = is_test
        self.id_column = id_column
        self.col_super = col_super
        self.col_sub = col_sub

        if labels_csv is None:
            # Test set: filenames are all files in images_dir
            self.df = pd.DataFrame({self.id_column: sorted(os.listdir(self.images_dir))})
            self.has_labels = False
        else:
            self.df = pd.read_csv(labels_csv)
            if not {self.id_column, self.col_sub}.issubset(self.df.columns):
                missing = {self.id_column, self.col_sub} - set(self.df.columns)
                raise ValueError(f"Missing required columns in {labels_csv}: {missing}")
            self.has_labels = self.col_super in self.df.columns and self.col_sub in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.images_dir / str(row[self.id_column])
        # Pillow is robust to RGB/LA/etc; enforce RGB
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        x = self.transform(im) if self.transform is not None else T.ToTensor()(im)

        if self.is_test or not self.has_labels:
            # Return id for writing predictions later
            return x, str(row[self.id_column])

        y_sup = int(row[self.col_super])
        y_sub = int(row[self.col_sub])
        labels = {"super": torch.tensor(y_sup, dtype=torch.long), "sub": torch.tensor(y_sub, dtype=torch.long)}
        return x, labels


def default_transforms(image_size: int = 224) -> dict[str, T.Compose]:
    # ImageNet-ish recipe; change size if you like
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tf = T.Compose(
        [
            T.Resize(int(image_size * 1.14)),  # keep aspect ratio
            T.RandomResizedCrop(image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    eval_tf = T.Compose(
        [
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return {"train": train_tf, "eval": eval_tf}


def build_M(superclass_csv: Path, subclass_csv: Path, df_train: pd.DataFrame) -> torch.Tensor:
    """
    Build an [S, K] mapping matrix M, using any rows present in train CSV.
    We assume df_train has integer columns COL_SUPER, COL_SUB.
    """
    S = len(pd.read_csv(superclass_csv))
    K = len(pd.read_csv(subclass_csv))
    M = torch.zeros(S, K, dtype=torch.float32)
    # If you want every possible mapping, you can precompute from a master file;
    # here we fill based on observed pairs in training CSV.
    for s, k in df_train[[COL_SUPER, COL_SUB]].drop_duplicates().itertuples(index=False):
        if 0 <= s < S and 0 <= k < K:
            M[s, k] = 1.0
    return M


def make_dataloaders(
    root_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    image_size: int = 224,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, int, int]:
    """
    Returns:
      train_loader, val_loader, test_loader, M[S,K], S, K
    """
    data_dir = root_dir / "img_data"
    train_images = data_dir / "train_images"
    test_images = data_dir / "test_images"
    train_csv = data_dir / "train_data.csv"
    sup_csv = data_dir / "superclass_mapping.csv"
    sub_csv = data_dir / "subclass_mapping.csv"

    tfs = default_transforms(image_size)

    # Build a temporary df to compute M and for splits
    df = pd.read_csv(train_csv)
    # Basic sanity checks
    for col in (COL_IMAGE, COL_SUB):
        if col not in df.columns:
            raise ValueError(f"{train_csv} must contain column '{col}'")
    if COL_SUPER not in df.columns:
        raise ValueError(f"{train_csv} must contain column '{COL_SUPER}' (needed for hierarchical training).")

    # Split train/val by rows
    full_ds = HierImageDataset(train_images, train_csv, transform=tfs["train"], is_test=False)
    n_total = len(full_ds)
    n_val = max(1, int(round(val_fraction * n_total)))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(1337))
    # Use eval transforms for val dataset
    val_ds.dataset.transform = tfs["eval"]

    # Test dataset (no labels)
    test_ds = HierImageDataset(test_images, labels_csv=None, transform=tfs["eval"], is_test=True)

    # Build M mapping for BaseModel.configure_hierarchy
    M = build_M(sup_csv, sub_csv, df)
    S, K = M.shape

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, M, S, K
