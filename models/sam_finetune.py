"""
Fine-tuning Meta's Segment Anything Model (SAM) for brain MRI segmentation.

Strategy
--------
- Freeze the heavy ViT image encoder.
- Keep prompt encoder frozen (we use a fixed "no prompt" / empty prompt).
- Only train the lightweight mask decoder (~4 M params).

This makes fine-tuning feasible on a single consumer GPU (8 GB VRAM).

Requirements
------------
    pip install segment-anything
    # Download ViT-B checkpoint (~375 MB):
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from segment_anything import sam_model_registry, SamPredictor
    from segment_anything.modeling import Sam
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[WARN] segment-anything not installed. SAM model unavailable.")


# ---------------------------------------------------------------------------
# SAM wrapper for dense binary segmentation
# ---------------------------------------------------------------------------

class SAMSegmenter(nn.Module):
    """
    Wraps a SAM model for dense (prompt-free) binary segmentation.

    The image encoder maps (B, 3, 1024, 1024) → (B, 256, 64, 64).
    We upsample the mask decoder output to match the input resolution.

    Input images are single-channel (grayscale); we replicate to 3 channels
    and resize to 1024×1024 as required by SAM's encoder.
    """

    SAM_INPUT_SIZE = 1024

    def __init__(self, checkpoint: str, model_type: str = "vit_b"):
        super().__init__()
        assert SAM_AVAILABLE, "Install segment-anything: pip install segment-anything"

        sam: Sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.image_encoder  = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder   = sam.mask_decoder

        # Freeze encoder and prompt encoder — only decoder is trainable
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        print(f"[SAM] Trainable params: "
              f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    # ------------------------------------------------------------------
    def _prepare_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, H, W) normalised [0, 1]
        returns (B, 3, 1024, 1024) in SAM's expected pixel range [0, 255]
        """
        x3 = x.repeat(1, 3, 1, 1)                          # (B, 3, H, W)
        x3 = F.interpolate(x3,
                           size=(self.SAM_INPUT_SIZE, self.SAM_INPUT_SIZE),
                           mode="bilinear", align_corners=False)
        x3 = x3 * 255.0                                     # SAM expects 0-255
        return x3

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W)  — normalised grayscale MRI slice

        Returns
        -------
        logits : (B, 1, H, W)  — raw (un-sigmoided) predictions
        """
        B, _, H, W = x.shape
        img = self._prepare_image(x)

        # Image embeddings — no grad for frozen encoder
        with torch.no_grad():
            image_embeddings = self.image_encoder(img)   # (B, 256, 64, 64)

        # Empty / no-op prompts
        sparse_emb, dense_emb = self.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        # prompt_encoder returns embeddings for a *single* image; expand to batch
        sparse_emb = sparse_emb.expand(B, -1, -1)
        dense_emb  = dense_emb.expand(B, -1, -1, -1)

        # Mask decoder
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,       # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_emb,     # (B, N, 256)
            dense_prompt_embeddings=dense_emb,       # (B, 256, 64, 64)
            multimask_output=False,                  # single mask
        )
        # low_res_masks : (B, 1, 256, 256) — upsample back to input size
        logits = F.interpolate(low_res_masks,
                               size=(H, W),
                               mode="bilinear", align_corners=False)
        return logits   # (B, 1, H, W)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_sam_segmenter(checkpoint: str,
                        model_type: str = "vit_b") -> SAMSegmenter:
    """Load and configure SAM for fine-tuning."""
    model = SAMSegmenter(checkpoint=checkpoint, model_type=model_type)
    return model


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sam_finetune.py <path/to/sam_vit_b.pth>")
        sys.exit(0)

    ckpt  = sys.argv[1]
    model = build_sam_segmenter(ckpt)
    dummy = torch.randn(2, 1, 256, 256)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")   # (2, 1, 256, 256)
