"""
notebooks/plot_history.py
Plot training/validation curves from the JSON history files
produced by train.py.

Usage
-----
    python notebooks/plot_history.py \
        --unet_history checkpoints/unet/history.json \
        --sam_history  checkpoints/sam/history.json \
        --out_dir      results
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STYLE = {
    "unet_train": dict(color="#6c5ce7", linestyle="-",  linewidth=2,
                       label="U-Net train"),
    "unet_val":   dict(color="#6c5ce7", linestyle="--", linewidth=2,
                       label="U-Net val",  alpha=0.8),
    "sam_train":  dict(color="#00b894", linestyle="-",  linewidth=2,
                       label="SAM train"),
    "sam_val":    dict(color="#00b894", linestyle="--", linewidth=2,
                       label="SAM val",   alpha=0.8),
}


def plot_metric(unet_h, sam_h, metric: str, title: str, ax, ylabel: str):
    def epoch_vals(history, split):
        return [ep[metric] for ep in history[split]]

    if unet_h:
        ax.plot(epoch_vals(unet_h, "train"), **STYLE["unet_train"])
        ax.plot(epoch_vals(unet_h, "val"),   **STYLE["unet_val"])
    if sam_h:
        ax.plot(epoch_vals(sam_h,  "train"), **STYLE["sam_train"])
        ax.plot(epoch_vals(sam_h,  "val"),   **STYLE["sam_val"])

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet_history", type=str, default=None)
    p.add_argument("--sam_history",  type=str, default=None)
    p.add_argument("--out_dir",      type=str, default="results")
    args = p.parse_args()

    unet_h = None
    sam_h  = None

    if args.unet_history and Path(args.unet_history).exists():
        with open(args.unet_history) as f:
            unet_h = json.load(f)
    if args.sam_history and Path(args.sam_history).exists():
        with open(args.sam_history) as f:
            sam_h = json.load(f)

    if unet_h is None and sam_h is None:
        print("No history files found — pass --unet_history or --sam_history.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training Curves — Brain MRI Segmentation",
                 fontsize=15, fontweight="bold")

    plot_metric(unet_h, sam_h, "loss",  "Loss",      axes[0], "Loss")
    plot_metric(unet_h, sam_h, "dice",  "Dice Score", axes[1], "Dice")
    plot_metric(unet_h, sam_h, "iou",   "IoU Score",  axes[2], "IoU")

    plt.tight_layout()
    out_path = Path(args.out_dir) / "training_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves → {out_path}")


if __name__ == "__main__":
    main()
