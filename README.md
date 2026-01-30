# Texture-Aware Conditional Diffusion for DFU Synthetic Progression (Reproducible Code)

This repository implements the method described in the manuscript **"Texture-Aware Diffusion Modelling for Synthetic Progression Prediction of Diabetic Foot Ulcers from Single Images"**:
- Preprocessing (256×256, background suppression, multi-scale texture enhancement, histogram matching, elastic deformation)
- Hybrid texture encoding (LBP + VGG-19 perceptual features) used as conditioning
- Conditional DDPM with a U-Net backbone and **cross-attention** integrating texture conditioning
- Multi-objective training (diffusion objective + reconstruction + perceptual + adversarial + feature-consistency)
- Baselines: PatchGAN-style conditional GAN, and a VAE
- Ablations: (i) no texture conditioning, (ii) no cross-attention, (iii) neither

> **Dataset**: Kaggle *diabetic-foot-ulcer-dfu* (laithjj). Download using Kaggle API or from the Kaggle webpage and unzip locally.

---

## 1) Environment

**Python**: 3.10+  
**PyTorch**: 2.0+ (CUDA recommended)

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 2) Dataset layout

Point `--data_root` to the unzipped Kaggle dataset folder.

The Kaggle dataset contains multiple folders (e.g., “Original Images”, “Patches”, etc.).  
This code **recursively discovers** image files under `data_root` (jpg/png/jpeg/webp).

---

## 3) Reproducible split + pseudo-stages

The paper uses a fixed 70/15/15 split (image-level) and a pseudo-progression paradigm (sorted into ordinal stages by severity proxies).

Create splits + pseudo-stages:

```bash
python -m scripts.prepare_splits \
  --data_root "/path/to/kaggle/diabetic-foot-ulcer-dfu" \
  --out_dir "./artifacts/splits" \
  --seed 2026 \
  --num_stages 4
```

This writes:
- `train.csv`, `val.csv`, `test.csv` with columns: `path,stage,score`
- `reference_hist.npy` used for histogram matching (built from a small subset of training images)

---

## 4) Train diffusion model (full model)

```bash
python -m scripts.train_diffusion \
  --data_root "/path/to/kaggle/diabetic-foot-ulcer-dfu" \
  --splits_dir "./artifacts/splits" \
  --out_dir "./artifacts/runs/diffusion_full" \
  --epochs 300 \
  --batch_size 8 \
  --lr 1e-4 \
  --timesteps 1000 \
  --ema_decay 0.999 \
  --use_texture 1 \
  --use_cross_attention 1
```

Ablations:
```bash
# no texture conditioning
python -m scripts.train_diffusion ... --use_texture 0 --use_cross_attention 1 --out_dir ./artifacts/runs/diffusion_no_texture

# no cross-attention
python -m scripts.train_diffusion ... --use_texture 1 --use_cross_attention 0 --out_dir ./artifacts/runs/diffusion_no_ca

# no both
python -m scripts.train_diffusion ... --use_texture 0 --use_cross_attention 0 --out_dir ./artifacts/runs/diffusion_plain
```

---

## 5) Train baselines

### GAN baseline (conditional PatchGAN)
```bash
python -m scripts.train_gan_baseline \
  --data_root "/path/to/kaggle/diabetic-foot-ulcer-dfu" \
  --splits_dir "./artifacts/splits" \
  --out_dir "./artifacts/runs/gan"
```

### VAE baseline
```bash
python -m scripts.train_vae_baseline \
  --data_root "/path/to/kaggle/diabetic-foot-ulcer-dfu" \
  --splits_dir "./artifacts/splits" \
  --out_dir "./artifacts/runs/vae"
```

---

## 6) Evaluation (SSIM / PSNR / FID / TCS / GTFS)

```bash
python -m scripts.evaluate \
  --data_root "/path/to/kaggle/diabetic-foot-ulcer-dfu" \
  --splits_dir "./artifacts/splits" \
  --ckpt "./artifacts/runs/diffusion_full/checkpoints/ema_last.pt" \
  --out_dir "./artifacts/eval/diffusion_full" \
  --model diffusion
```

**GTFS note**: If you have expert granulation masks, place them in `--granulation_masks_dir` with filenames matching image stems.
If not provided, the code computes a **proxy** granulation mask via color heuristics (documented in `src/metrics/gtfs.py`).

---

## 7) Inference: generate a progression montage for one image

```bash
python -m scripts.infer_progression \
  --ckpt "./artifacts/runs/diffusion_full/checkpoints/ema_last.pt" \
  --image "/path/to/some_dfu_image.jpg" \
  --out_dir "./artifacts/infer"
```

---

## Citation

If you use this code, cite your associated manuscript and the Kaggle dataset (laithjj).
