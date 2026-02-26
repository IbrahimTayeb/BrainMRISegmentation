"""
app.py — Interactive Gradio demo for Brain MRI Segmentation.

Runs both U-Net and SAM side-by-side so visitors can upload an MRI slice
and instantly compare both models' predictions.

Usage
-----
python app.py \
    --unet_checkpoint checkpoints/unet/best_model.pth \
    --sam_checkpoint_pth sam_vit_b_01ec64.pth \
    --sam_finetuned checkpoints/sam/best_model.pth

Deploy to Hugging Face Spaces
------------------------------
1. Create a new Space (Gradio SDK).
2. Push this repo to the Space.
3. Add your checkpoint files or download them in a setup script.
4. The Space URL becomes your live demo link for the resume!
"""

import argparse
import time
from io import BytesIO

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import gradio as gr

from models.unet import UNet
from models.sam_finetune import build_sam_segmenter
from models.losses import dice_score, iou_score


# ---------------------------------------------------------------------------
# Global model state (loaded once)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_MODEL = None
SAM_MODEL  = None


def load_models(unet_ckpt: str, sam_pth: str, sam_finetuned: str):
    global UNET_MODEL, SAM_MODEL

    print(f"Loading U-Net from {unet_ckpt}…")
    UNET_MODEL = UNet(in_channels=1, out_channels=1).to(DEVICE)
    ckpt = torch.load(unet_ckpt, map_location=DEVICE)
    UNET_MODEL.load_state_dict(ckpt["state_dict"])
    UNET_MODEL.eval()

    print(f"Loading SAM from {sam_pth} + {sam_finetuned}…")
    SAM_MODEL = build_sam_segmenter(sam_pth).to(DEVICE)
    ckpt = torch.load(sam_finetuned, map_location=DEVICE)
    SAM_MODEL.load_state_dict(ckpt["state_dict"])
    SAM_MODEL.eval()

    print("Models ready.")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray) -> torch.Tensor:
    """
    Accept a numpy image (H, W) or (H, W, 3) from Gradio and return
    a (1, 1, 256, 256) float tensor normalised to [0, 1].
    """
    if image.ndim == 3:
        image = image.mean(axis=2)  # convert RGB to grayscale
    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    import torch.nn.functional as F
    t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False)
    return t.to(DEVICE)


def run_inference(img_tensor: torch.Tensor):
    """Run both models and return logits, predictions, and timing."""
    with torch.no_grad():
        t0 = time.perf_counter()
        unet_logits = UNET_MODEL(img_tensor)
        unet_time   = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sam_logits  = SAM_MODEL(img_tensor)
        sam_time    = (time.perf_counter() - t0) * 1000

    unet_pred = (torch.sigmoid(unet_logits) >= 0.5).float()
    sam_pred  = (torch.sigmoid(sam_logits)  >= 0.5).float()

    return (unet_logits, unet_pred, unet_time,
            sam_logits,  sam_pred,  sam_time)


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def overlay_mask(img: np.ndarray, mask: np.ndarray,
                 color=(1.0, 0.2, 0.2), alpha=0.45) -> np.ndarray:
    """Overlay a binary mask on a grayscale image as an RGBA composite."""
    rgb = np.stack([img, img, img], axis=-1)  # (H, W, 3)
    overlay = rgb.copy()
    overlay[mask == 1] = (np.array(color) * 0.6 +
                          rgb[mask == 1] * 0.4)
    # Blend
    blended = (1 - alpha) * rgb + alpha * overlay
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)


def make_comparison_figure(img_np, unet_pred_np, sam_pred_np,
                            unet_metrics, sam_metrics,
                            unet_time, sam_time) -> Image.Image:
    """Create a 4-panel matplotlib figure and return as PIL Image."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("#0f0f0f")

    panels = [
        ("Input FLAIR", img_np, "gray"),
        ("U-Net Prediction",
         overlay_mask(img_np, unet_pred_np, color=(1.0, 0.3, 0.3)), None),
        ("SAM Prediction",
         overlay_mask(img_np, sam_pred_np,  color=(0.3, 0.8, 0.3)), None),
        ("Overlay Comparison", None, None),
    ]

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0f0f0f")

    # Panel 0 — original
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Input FLAIR", color="white", fontsize=13, pad=8)

    # Panel 1 — U-Net
    axes[1].imshow(panels[1][1])
    dice_u = unet_metrics.get("dice", 0)
    axes[1].set_title(
        f"U-Net  |  Dice: {dice_u:.3f}  |  {unet_time:.0f} ms",
        color="#ff6b6b", fontsize=12, pad=8
    )

    # Panel 2 — SAM
    axes[2].imshow(panels[2][1])
    dice_s = sam_metrics.get("dice", 0)
    axes[2].set_title(
        f"SAM  |  Dice: {dice_s:.3f}  |  {sam_time:.0f} ms",
        color="#69db7c", fontsize=12, pad=8
    )

    # Panel 3 — both overlaid for visual comparison
    rgb = np.stack([img_np, img_np, img_np], axis=-1)
    comp = rgb.copy()
    comp[unet_pred_np == 1] = [0.9, 0.2, 0.2]   # red = U-Net
    comp[sam_pred_np  == 1] = [0.2, 0.8, 0.2]   # green = SAM
    # Agreement in yellow
    both = (unet_pred_np == 1) & (sam_pred_np == 1)
    comp[both] = [0.9, 0.9, 0.1]
    axes[3].imshow(np.clip(comp, 0, 1))
    legend = [
        mpatches.Patch(color="#e63946", label="U-Net only"),
        mpatches.Patch(color="#2dc653", label="SAM only"),
        mpatches.Patch(color="#e9c46a", label="Both agree"),
    ]
    axes[3].legend(handles=legend, loc="lower right",
                   fontsize=9, framealpha=0.6)
    axes[3].set_title("Agreement Map", color="white", fontsize=13, pad=8)

    plt.tight_layout(pad=1.5)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=130,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Gradio callback
# ---------------------------------------------------------------------------

def segment_image(image):
    """Main Gradio callback."""
    if image is None:
        return None, "Please upload an MRI slice."

    if UNET_MODEL is None or SAM_MODEL is None:
        return None, "⚠️ Models not loaded. Run app.py with checkpoint paths."

    img_tensor = preprocess(image)
    img_np     = img_tensor[0, 0].cpu().numpy()   # (256, 256)

    (unet_logits, unet_pred, unet_time,
     sam_logits,  sam_pred,  sam_time) = run_inference(img_tensor)

    unet_pred_np = unet_pred[0, 0].cpu().numpy()
    sam_pred_np  = sam_pred[0, 0].cpu().numpy()

    # Compute metrics (no GT available at demo time, so just show coverage)
    unet_coverage = unet_pred_np.mean() * 100
    sam_coverage  = sam_pred_np.mean()  * 100

    unet_metrics = {"dice": 0.0, "coverage_pct": unet_coverage}
    sam_metrics  = {"dice": 0.0, "coverage_pct": sam_coverage}

    fig_img = make_comparison_figure(
        img_np, unet_pred_np, sam_pred_np,
        unet_metrics, sam_metrics, unet_time, sam_time
    )

    stats = (
        f"**U-Net**\n"
        f"- Inference: {unet_time:.1f} ms\n"
        f"- Tumor coverage: {unet_coverage:.2f}% of slice\n\n"
        f"**SAM (fine-tuned)**\n"
        f"- Inference: {sam_time:.1f} ms\n"
        f"- Tumor coverage: {sam_coverage:.2f}% of slice\n\n"
        f"*Upload a labeled slice to see Dice scores.*"
    )

    return fig_img, stats


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Brain MRI Tumor Segmentation",
        theme=gr.themes.Base(primary_hue="violet"),
        css="""
        .gradio-container { max-width: 1100px; margin: auto; }
        footer { display: none !important; }
        """
    ) as demo:

        gr.Markdown("""
# 🧠 Brain MRI Tumor Segmentation
### U-Net vs SAM (Segment Anything Model)

Upload a 2-D **FLAIR MRI slice** (grayscale or RGB) to compare both models' predictions side-by-side.
Trained on **BraTS 2020** using PyTorch.  [GitHub →](https://github.com/yourname/brain-mri-segmentation)
""")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    label="Upload MRI Slice",
                    type="numpy",
                    image_mode="L",
                    height=300,
                )
                run_btn = gr.Button("🔬 Run Segmentation", variant="primary",
                                    size="lg")

            with gr.Column(scale=2):
                output_img   = gr.Image(label="Segmentation Results",
                                        type="pil", height=380)
                output_stats = gr.Markdown(label="Statistics")

        gr.Examples(
            examples=[["examples/sample_mri_1.png"],
                       ["examples/sample_mri_2.png"]],
            inputs=input_img,
            label="Example Slices",
        )

        gr.Markdown("""
---
**Model Details**
| | U-Net | SAM (ViT-B fine-tuned) |
|---|---|---|
| Architecture | Custom encoder-decoder | Meta's ViT-B backbone |
| Trainable params | ~31 M | ~4 M (decoder only) |
| Training data | BraTS 2020 FLAIR slices | Same |
| Loss | Focal + Dice | Focal + Dice |
""")

        run_btn.click(
            fn=segment_image,
            inputs=input_img,
            outputs=[output_img, output_stats],
        )
        input_img.change(
            fn=segment_image,
            inputs=input_img,
            outputs=[output_img, output_stats],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--unet_checkpoint",    type=str,
                   default="checkpoints/unet/best_model.pth")
    p.add_argument("--sam_checkpoint_pth", type=str,
                   default="sam_vit_b_01ec64.pth")
    p.add_argument("--sam_finetuned",      type=str,
                   default="checkpoints/sam/best_model.pth")
    p.add_argument("--share",              action="store_true",
                   help="Create a public Gradio link")
    p.add_argument("--port",               type=int, default=7860)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_models(args.unet_checkpoint,
                args.sam_checkpoint_pth,
                args.sam_finetuned)
    demo = build_demo()
    demo.launch(share=args.share, server_port=args.port)
