"""
train.py — Train U-Net and/or SAM on BraTS 2020.

Usage
-----
# Train U-Net only
python train.py --model unet --data_root /path/to/BraTS2020

# Train SAM only
python train.py --model sam --data_root /path/to/BraTS2020 \
                --sam_checkpoint sam_vit_b_01ec64.pth

# Train both sequentially
python train.py --model both --data_root /path/to/BraTS2020 \
                --sam_checkpoint sam_vit_b_01ec64.pth
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from data.dataset import get_dataloaders
from models.unet import UNet
from models.sam_finetune import build_sam_segmenter
from models.losses import CombinedLoss, dice_score, iou_score, MetricTracker


# ---------------------------------------------------------------------------
# Training loop (shared by both models)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device) -> dict:
    model.train()
    tracker = MetricTracker()

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            ds = dice_score(logits.detach(), masks)
            io = iou_score(logits.detach(), masks)

        tracker.update(loss=loss.item(), dice=ds, iou=io)

    return tracker.averages()


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    tracker = MetricTracker()

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, masks)
        ds = dice_score(logits, masks)
        io = iou_score(logits, masks)
        tracker.update(loss=loss.item(), dice=ds, iou=io)

    return tracker.averages()


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train(model, model_name: str, train_loader, val_loader,
          args, device):

    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    criterion = CombinedLoss(focal_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler()

    best_val_dice = 0.0
    history = {"train": [], "val": []}

    print(f"\n{'='*60}")
    print(f"  Training {model_name.upper()}  ({args.epochs} epochs)")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer,
                                        criterion, scaler, device)
        val_metrics   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        elapsed = time.time() - t0
        print(f"[{model_name}] Epoch {epoch:03d}/{args.epochs}  "
              f"({elapsed:.1f}s)  "
              f"train loss={train_metrics['loss']:.4f}  "
              f"train dice={train_metrics['dice']:.4f}  |  "
              f"val loss={val_metrics['loss']:.4f}  "
              f"val dice={val_metrics['dice']:.4f}")

        # Checkpoint best model
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            ckpt_path = out_dir / "best_model.pth"
            torch.save({
                "epoch":      epoch,
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "val_dice":   best_val_dice,
                "val_iou":    val_metrics["iou"],
            }, ckpt_path)
            print(f"  ✓ New best val dice={best_val_dice:.4f} — saved to {ckpt_path}")

    # Save training history
    import json
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[{model_name}] Training complete. Best val Dice: {best_val_dice:.4f}")
    return best_val_dice


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          type=str, default="unet",
                   choices=["unet", "sam", "both"],
                   help="Which model to train")
    p.add_argument("--data_root",      type=str, required=True,
                   help="Path to BraTS2020 training directory")
    p.add_argument("--sam_checkpoint", type=str, default="sam_vit_b_01ec64.pth",
                   help="Path to SAM ViT-B checkpoint")
    p.add_argument("--output_dir",     type=str, default="checkpoints",
                   help="Directory to save checkpoints")
    p.add_argument("--epochs",         type=int, default=30)
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--img_size",       type=int, default=256)
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    results = {}

    if args.model in ("unet", "both"):
        model = UNet(in_channels=1, out_channels=1).to(device)
        dice  = train(model, "unet", train_loader, val_loader, args, device)
        results["unet"] = dice

    if args.model in ("sam", "both"):
        model = build_sam_segmenter(args.sam_checkpoint).to(device)
        dice  = train(model, "sam", train_loader, val_loader, args, device)
        results["sam"] = dice

    print("\n=== Final Results ===")
    for name, dice in results.items():
        print(f"  {name}: best val Dice = {dice:.4f}")


if __name__ == "__main__":
    main()
