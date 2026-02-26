"""
notebooks/eda.py
Exploratory Data Analysis for BraTS 2020.

Run this after downloading the dataset to understand:
  - Dataset statistics
  - Class imbalance
  - Sample visualisations

Usage
-----
    python notebooks/eda.py --data_root /path/to/BraTS2020
"""

import argparse
import random
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob

from data.dataset import build_slice_index, load_volume, normalize_slice


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_dataset_stats(records, n_sample: int = 500):
    """Sample slices and return foreground pixel ratio statistics."""
    sample = random.sample(records, min(n_sample, len(records)))
    fg_ratios = []
    for flair_path, seg_path, slc_idx in sample:
        seg_vol = load_volume(seg_path)
        mask    = (seg_vol[:, :, slc_idx] > 0)
        fg_ratios.append(mask.mean())
    return np.array(fg_ratios)


def plot_class_imbalance(fg_ratios, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(fg_ratios * 100, bins=50, color="#6c5ce7", edgecolor="white")
    axes[0].set_xlabel("Foreground pixel %")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Tumour Area per Slice")

    labels = ["Background", "Tumour"]
    sizes  = [1 - fg_ratios.mean(), fg_ratios.mean()]
    axes[1].pie(sizes, labels=labels, autopct="%1.2f%%",
                colors=["#636e72", "#6c5ce7"])
    axes[1].set_title("Mean Pixel-Level Class Balance")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved class imbalance plot → {out_path}")


def plot_sample_slices(records, out_path: str, n: int = 6):
    """Plot n random FLAIR slices with their segmentation masks."""
    sample = random.sample(records, min(n, len(records)))

    fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n))
    fig.suptitle("Sample FLAIR Slices", fontsize=14, fontweight="bold")

    for row, (flair_path, seg_path, slc_idx) in enumerate(sample):
        flair_vol = load_volume(flair_path)
        seg_vol   = load_volume(seg_path)

        img  = normalize_slice(flair_vol[:, :, slc_idx])
        mask = (seg_vol[:, :, slc_idx] > 0).astype(np.uint8)

        # Overlay
        rgb     = np.stack([img, img, img], axis=-1)
        overlay = rgb.copy()
        overlay[mask == 1] = [0.9, 0.2, 0.2]

        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(f"FLAIR — slice {slc_idx}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask, cmap="hot")
        axes[row, 1].set_title("Ground Truth Mask")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved sample slices → {out_path}")


def plot_modality_comparison(data_root: str, out_path: str):
    """Show all 4 BraTS modalities for a single patient."""
    pattern  = data_root + "/**/*_flair.nii"
    flair_files = glob.glob(pattern, recursive=True)
    if not flair_files:
        pattern = data_root + "/**/*_flair.nii.gz"
        flair_files = glob.glob(pattern, recursive=True)
    if not flair_files:
        print("[WARN] No FLAIR files found for modality comparison.")
        return

    sample_dir = Path(flair_files[0]).parent
    modalities  = ["flair", "t1", "t1ce", "t2"]

    # pick a mid-slice
    flair = nib.load(flair_files[0]).get_fdata()
    mid   = flair.shape[2] // 2

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    titles = ["FLAIR", "T1", "T1ce", "T2", "Segmentation"]

    for col, (mod, title) in enumerate(zip(modalities, titles)):
        candidates = list(sample_dir.glob(f"*_{mod}.nii*"))
        if not candidates:
            axes[col].set_visible(False)
            continue
        vol = nib.load(str(candidates[0])).get_fdata()
        slc = normalize_slice(vol[:, :, mid])
        axes[col].imshow(slc, cmap="gray")
        axes[col].set_title(title, fontsize=12)
        axes[col].axis("off")

    # Segmentation column
    seg_candidates = list(sample_dir.glob("*_seg.nii*"))
    if seg_candidates:
        seg  = nib.load(str(seg_candidates[0])).get_fdata()
        axes[4].imshow(seg[:, :, mid], cmap="tab10")
        axes[4].set_title("Segmentation", fontsize=12)
        axes[4].axis("off")

    plt.suptitle("BraTS 2020 — All Modalities", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved modality comparison → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir",   type=str, default="results/eda")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building slice index…")
    records = build_slice_index(args.data_root)
    print(f"  Total tumor-bearing slices: {len(records):,}")

    print("\nComputing class imbalance…")
    fg_ratios = compute_dataset_stats(records)
    print(f"  Mean foreground ratio: {fg_ratios.mean()*100:.2f}%  "
          f"(std {fg_ratios.std()*100:.2f}%)")
    plot_class_imbalance(fg_ratios, str(out_dir / "class_imbalance.png"))

    print("\nPlotting sample slices…")
    plot_sample_slices(records, str(out_dir / "sample_slices.png"))

    print("\nPlotting modality comparison…")
    plot_modality_comparison(args.data_root,
                              str(out_dir / "modality_comparison.png"))

    print("\nEDA complete. All figures saved to:", out_dir)


if __name__ == "__main__":
    main()
