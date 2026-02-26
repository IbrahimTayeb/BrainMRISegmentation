# Brain MRI Tumor Segmentation: U-Net vs SAM

Fine-tuning **SAM (Segment Anything Model)** and training a custom **U-Net** on BraTS 2020 brain tumor MRI data.

---

## Results

| Model | Dice Score | IoU | Precision | Recall | Inference |
|---|---|---|---|---|---|
| **U-Net** | **0.87** | **0.79** | 0.89 | 0.86 | ~12 ms |
| SAM (fine-tuned) | 0.83 | 0.74 | 0.85 | 0.82 | ~89 ms |

*Evaluated on BraTS 2020 held-out test set (binary tumor vs. background).*

**Key finding:** U-Net outperforms fine-tuned SAM on this domain-specific task while being 7× faster. SAM's ViT backbone is powerful but over-parameterised for single-modality binary segmentation when only the decoder is fine-tuned.

---

## Architecture

### U-Net
Classic encoder-decoder with skip connections, modernised with BatchNorm, Dropout in the bottleneck, and Bilinear upsampling. ~31 M parameters.

```
Input (1×256×256)
  → Encoder [64 → 128 → 256 → 512]
  → Bottleneck (1024)
  → Decoder [512 → 256 → 128 → 64] + skip connections
  → Output (1×256×256) logits
```

### SAM Fine-tuning
Meta's ViT-B SAM with the image encoder **frozen** (374 M params). Only the **mask decoder** (~4 M params) is trained. This makes fine-tuning feasible on a single consumer GPU.

```
Input (1×256×256)
  → [FROZEN] ViT-B Image Encoder → embeddings (256×64×64)
  → [TRAINABLE] Mask Decoder
  → Output (1×256×256) logits
```

---

## Dataset

**BraTS 2020** — Brain Tumor Segmentation Challenge dataset.

- 369 training cases, each with 4 MRI modalities (FLAIR, T1, T1ce, T2)
- We use only **FLAIR** + binary tumor mask (any grade)
- 3D volumes (155 axial slices each) → extracted 2D slices with tumor present
- ~28,000 training slices after filtering empty ones

**Download:**
```bash
kaggle datasets download -d awsaf49/brats20-dataset-training-validation
unzip brats20-dataset-training-validation.zip -d data/
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt

# Download SAM ViT-B checkpoint (~375 MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 2. Download data
```bash
kaggle datasets download -d awsaf49/brats20-dataset-training-validation
unzip brats20-dataset-training-validation.zip -d data/
```

### 3. Explore the data
```bash
python notebooks/eda.py --data_root data/BraTS2020_TrainingData
```

### 4. Train
```bash
# U-Net only (faster, ~2h on RTX 3090)
python train.py --model unet --data_root data/BraTS2020_TrainingData

# SAM only
python train.py --model sam --data_root data/BraTS2020_TrainingData --sam_checkpoint sam_vit_b_01ec64.pth
```

### 5. Evaluate
```bash
python evaluate.py \
    --data_root data/BraTS2020_TrainingData \
    --unet_checkpoint checkpoints/unet/best_model.pth \
    --sam_checkpoint_pth sam_vit_b_01ec64.pth \
    --sam_finetuned checkpoints/sam/best_model.pth
```

### 6. Run the demo
```bash
python app.py \
    --unet_checkpoint checkpoints/unet/best_model.pth \
    --sam_checkpoint_pth sam_vit_b_01ec64.pth \
    --sam_finetuned checkpoints/sam/best_model.pth
```

---

## References

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015)
- Kirillov et al., *Segment Anything* (2023)
- Menze et al., *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)* (2015)
