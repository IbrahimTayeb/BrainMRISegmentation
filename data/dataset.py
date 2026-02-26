"""
BraTS 2020 Dataset loading and preprocessing.

Expected directory structure:
    data/
        BraTS2020_TrainingData/
            MICCAI_BraTS2020_TrainingData/
                BraTS20_Training_001/
                    BraTS20_Training_001_flair.nii
                    BraTS20_Training_001_seg.nii
                    ...
                BraTS20_Training_002/
                ...

Download from Kaggle:
    kaggle datasets download -d awsaf49/brats20-dataset-training-validation
"""

import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalize_slice(slc: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2-D slice to [0, 1]."""
    min_val, max_val = slc.min(), slc.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(slc, dtype=np.float32)
    return ((slc - min_val) / (max_val - min_val)).astype(np.float32)


def load_volume(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()


def extract_slices(flair_path: str, seg_path: str):
    """
    Extract all 2-D axial slices that contain at least one tumor voxel.

    Returns
    -------
    list of (flair_slice, binary_mask) tuples, each shape (H, W).
    """
    flair_vol = load_volume(flair_path)   # (H, W, D)
    seg_vol   = load_volume(seg_path)     # (H, W, D)

    slices = []
    for i in range(flair_vol.shape[2]):
        mask = (seg_vol[:, :, i] > 0).astype(np.uint8)
        if mask.sum() == 0:
            continue
        img = normalize_slice(flair_vol[:, :, i])
        slices.append((img, mask))
    return slices


def build_slice_index(data_root: str):
    """
    Walk the BraTS directory tree and return a flat list of
    (flair_path, seg_path, slice_idx) tuples for all tumor-bearing slices.
    """
    pattern = os.path.join(data_root, "**", "*_flair.nii")
    flair_files = sorted(glob.glob(pattern, recursive=True))

    if not flair_files:
        # also try .nii.gz
        pattern = os.path.join(data_root, "**", "*_flair.nii.gz")
        flair_files = sorted(glob.glob(pattern, recursive=True))

    records = []
    for flair_path in flair_files:
        seg_path = flair_path.replace("_flair.nii", "_seg.nii")
        if not os.path.exists(seg_path):
            seg_path = flair_path.replace("_flair.nii.gz", "_seg.nii.gz")
        if not os.path.exists(seg_path):
            print(f"[WARN] No seg file for {flair_path}, skipping.")
            continue

        flair_vol = load_volume(flair_path)
        seg_vol   = load_volume(seg_path)
        for i in range(flair_vol.shape[2]):
            if (seg_vol[:, :, i] > 0).any():
                records.append((flair_path, seg_path, i))

    return records


# ---------------------------------------------------------------------------
# Albumentations transforms
# ---------------------------------------------------------------------------

def get_train_transforms(img_size: int = 256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                           rotate_limit=15, p=0.5,
                           border_mode=0),
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.4),
        A.GaussNoise(p=0.3),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 256):
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class BraTSDataset(Dataset):
    """
    Lazy-loading dataset.  Volumes are opened on the fly to keep
    RAM footprint small.
    """

    def __init__(self, records, transform=None):
        """
        Parameters
        ----------
        records   : list of (flair_path, seg_path, slice_idx)
        transform : albumentations Compose or None
        """
        self.records   = records
        self.transform = transform

        # cache open volumes to avoid re-loading the same file repeatedly
        self._flair_cache: dict = {}
        self._seg_cache:   dict = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        flair_path, seg_path, slc_idx = self.records[idx]

        if flair_path not in self._flair_cache:
            self._flair_cache[flair_path] = load_volume(flair_path)
        if seg_path not in self._seg_cache:
            self._seg_cache[seg_path] = load_volume(seg_path)

        flair_vol = self._flair_cache[flair_path]
        seg_vol   = self._seg_cache[seg_path]

        img  = normalize_slice(flair_vol[:, :, slc_idx])   # (H, W) float32
        mask = (seg_vol[:, :, slc_idx] > 0).astype(np.uint8)  # binary

        # albumentations expects HWC for image
        img_hwc = img[:, :, np.newaxis]  # (H, W, 1)

        if self.transform:
            augmented = self.transform(image=img_hwc, mask=mask)
            img_t  = augmented["image"].float()          # (1, H, W)
            mask_t = augmented["mask"].unsqueeze(0).float()  # (1, H, W)
        else:
            img_t  = torch.from_numpy(img).unsqueeze(0).float()
            mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        return img_t, mask_t


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_dataloaders(data_root: str,
                    img_size: int = 256,
                    batch_size: int = 16,
                    val_split: float = 0.15,
                    test_split: float = 0.10,
                    num_workers: int = 4,
                    seed: int = 42):
    """
    Build train / val / test DataLoaders from the BraTS directory.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    print("Building slice index (this may take a minute)…")
    records = build_slice_index(data_root)
    print(f"  Found {len(records):,} tumor-bearing slices.")

    train_val, test = train_test_split(records, test_size=test_split,
                                       random_state=seed)
    train, val = train_test_split(train_val,
                                  test_size=val_split / (1 - test_split),
                                  random_state=seed)

    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    train_ds = BraTSDataset(train, transform=get_train_transforms(img_size))
    val_ds   = BraTSDataset(val,   transform=get_val_transforms(img_size))
    test_ds  = BraTSDataset(test,  transform=get_val_transforms(img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
