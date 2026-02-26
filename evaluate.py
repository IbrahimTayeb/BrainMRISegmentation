"""
evaluate.py — Evaluate trained models on the test set and produce a comparison report.

Usage
-----
python evaluate.py \
    --data_root /path/to/BraTS2020 \
    --unet_checkpoint checkpoints/unet/best_model.pth \
    --sam_checkpoint_pth sam_vit_b_01ec64.pth \
    --sam_finetuned checkpoints/sam/best_model.pth \
    --output_dir results
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

from data.dataset import get_dataloaders
from models.unet import UNet
from models.sam_finetune import build_sam_segmenter
from models.losses import (CombinedLoss, dice_score, iou_score,
                            precision_recall, MetricTracker)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_unet(checkpoint_path: str, device) -> UNet:
    model = UNet(in_channels=1, out_channels=1).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_sam(sam_pth: str, finetuned_path: str, device):
    model = build_sam_segmenter(sam_pth).to(device)
    ckpt  = torch.load(finetuned_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def full_eval(model, loader, device, model_name: str = "") -> dict:
    criterion = CombinedLoss()
    tracker   = MetricTracker()
    times     = []

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        t0 = time.perf_counter()
        with autocast():
            logits = model(imgs)
        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append((time.perf_counter() - t0) / imgs.size(0) * 1000)

        loss        = criterion(logits, masks)
        ds          = dice_score(logits, masks)
        io          = iou_score(logits, masks)
        prec, rec   = precision_recall(logits, masks)

        tracker.update(loss=loss.item(), dice=ds, iou=io,
                       precision=prec, recall=rec)

    avgs = tracker.averages()
    avgs["inference_ms_per_image"] = float(np.mean(times))
    return avgs


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_comparison_figure(imgs, masks, unet_logits, sam_logits,
                            out_path: str, n: int = 4):
    """Save a side-by-side grid: Image | GT | U-Net | SAM."""
    n = min(n, imgs.size(0))
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))

    unet_preds = (torch.sigmoid(unet_logits) >= 0.5).float()
    sam_preds  = (torch.sigmoid(sam_logits)  >= 0.5).float()

    col_titles = ["FLAIR", "Ground Truth", "U-Net", "SAM (fine-tuned)"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight="bold")

    for i in range(n):
        img  = imgs[i, 0].cpu().numpy()
        gt   = masks[i, 0].cpu().numpy()
        un   = unet_preds[i, 0].cpu().numpy()
        sm   = sam_preds[i, 0].cpu().numpy()

        dice_u = dice_score(unet_logits[i:i+1], masks[i:i+1])
        dice_s = dice_score(sam_logits[i:i+1],  masks[i:i+1])

        for ax, data, cmap, title in zip(
            axes[i],
            [img, gt, un, sm],
            ["gray", "hot", "hot", "hot"],
            ["", "", f"Dice={dice_u:.3f}", f"Dice={dice_s:.3f}"],
        ):
            ax.imshow(data, cmap=cmap)
            ax.axis("off")
            if title:
                ax.set_xlabel(title, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison figure → {out_path}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(unet_metrics: dict, sam_metrics: dict, out_path: str):
    lines = [
        "# Brain MRI Segmentation — Evaluation Report\n",
        "## Quantitative Comparison\n",
        "| Metric | U-Net | SAM (fine-tuned) |",
        "|---|---|---|",
    ]
    all_keys = sorted(set(unet_metrics) | set(sam_metrics))
    for k in all_keys:
        u = unet_metrics.get(k, float("nan"))
        s = sam_metrics.get(k, float("nan"))
        lines.append(f"| {k} | {u:.4f} | {s:.4f} |")

    lines += [
        "",
        "## Analysis\n",
        "- **U-Net** is purpose-built for dense medical segmentation and typically achieves higher Dice scores on domain-specific datasets.",
        "- **SAM** (fine-tuned) brings zero-shot generalisation but requires more inference compute due to the ViT backbone.",
        "- For production deployment on embedded hardware, U-Net is the clear choice given its speed and smaller footprint.",
        "",
        "## Sample Predictions\n",
        "![Comparison](comparison.png)",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved report → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",       type=str, required=True)
    p.add_argument("--unet_checkpoint", type=str,
                   default="checkpoints/unet/best_model.pth")
    p.add_argument("--sam_checkpoint_pth", type=str,
                   default="sam_vit_b_01ec64.pth",
                   help="Original SAM weights")
    p.add_argument("--sam_finetuned",   type=str,
                   default="checkpoints/sam/best_model.pth",
                   help="Fine-tuned SAM checkpoint")
    p.add_argument("--output_dir",      type=str, default="results")
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--img_size",        type=int, default=256)
    p.add_argument("--num_workers",     type=int, default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("\nLoading U-Net…")
    unet = load_unet(args.unet_checkpoint, device)
    print("Evaluating U-Net…")
    unet_metrics = full_eval(unet, test_loader, device, "unet")
    print(f"  {unet_metrics}")

    print("\nLoading SAM…")
    sam = load_sam(args.sam_checkpoint_pth, args.sam_finetuned, device)
    print("Evaluating SAM…")
    sam_metrics = full_eval(sam, test_loader, device, "sam")
    print(f"  {sam_metrics}")

    # Save raw metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"unet": unet_metrics, "sam": sam_metrics}, f, indent=2)

    # Visual comparison on first batch
    imgs, masks = next(iter(test_loader))
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad():
        unet_logits = unet(imgs)
        sam_logits  = sam(imgs)

    save_comparison_figure(
        imgs, masks, unet_logits, sam_logits,
        out_path=str(out_dir / "comparison.png"),
    )

    write_report(unet_metrics, sam_metrics,
                 out_path=str(out_dir / "report.md"))

    print("\n=== Test Set Results ===")
    print(f"  U-Net  — Dice: {unet_metrics['dice']:.4f}  "
          f"IoU: {unet_metrics['iou']:.4f}  "
          f"Speed: {unet_metrics['inference_ms_per_image']:.1f} ms/img")
    print(f"  SAM    — Dice: {sam_metrics['dice']:.4f}  "
          f"IoU: {sam_metrics['iou']:.4f}  "
          f"Speed: {sam_metrics['inference_ms_per_image']:.1f} ms/img")


if __name__ == "__main__":
    main()
