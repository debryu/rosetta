# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of **multi-view nonlinear ICA** from *"The Incomplete Rosetta Stone Problem"* (Gresele et al., ICML 2020). A contrastive discriminator is trained to distinguish same-source image pairs from different-source pairs on the shapes3d dataset. The encoder inside the discriminator learns to invert the nonlinear mixing and recover the 5 latent generative factors (floor/wall/object hue, scale, shape), using orientation as the per-view noise.

## Environment

Managed with `uv` — no conda.

```bash
uv sync                                          # install all deps into .venv
uv run python train.py --epochs 100              # full training run
uv run jupyter notebook notebooks/tutorial.ipynb # interactive tutorial
```

## Common Commands

```bash
# Download dataset (~2 GB, one-time)
uv run python -c "from data import download_shapes3d; download_shapes3d()"

# Train (checkpoints saved to checkpoints/)
uv run python train.py --epochs 100 --batch_size 256

# Quick smoke-test (1 epoch, small batch)
uv run python train.py --epochs 1 --batch_size 64 --num_workers 0 --eval_samples 500

# Evaluate a saved checkpoint
uv run python -c "
from training.trainer import load_checkpoint, evaluate_representations
from data import Shapes3DPairDataset
from config import Config
cfg = Config()
model = load_checkpoint('checkpoints/best.pt', cfg=cfg)
ds = Shapes3DPairDataset(cfg.hdf5_path, split='val')
results = evaluate_representations(model, ds, cfg.device)
print('MCC:', results['mcc'])
"
```

## Architecture

```
data/dataset.py          Shapes3DPairDataset — returns (x1, x2, x2_neg, s_labels) triplets
                          x1, x2 : same scene, different orientations (positive pair)
                          x2_neg : different scene (negative pair)
                          s_labels: 5 ground-truth factors (evaluation only)

models/encoder.py        Encoder — 4-layer CNN → FC, maps 64x64x3 → latent_dim
models/discriminator.py  MultiViewDiscriminator — weight-tied encoder + decomposable head
                          implements r(x1, x2) = Σ_i ψ_i(h_i(x1), h_i(x2)) from Theorem 4
                          each ψ_i is a separate MLP seeing only (z1_i, z2_i) — required for identifiability

training/trainer.py      train_epoch, evaluate_representations, train
                          loss: BCEWithLogitsLoss on concat([pos_logits, neg_logits])
                          eval: MCC via Hungarian assignment on correlation matrix

evaluation/metrics.py    compute_mcc, linear_probe_r2
evaluation/visualize.py  plot_tsne, plot_factor_scatter, plot_latent_traversal, plot_training_curves

config.py                Config dataclass (latent_dim=5, batch_size=256, lr=3e-4, epochs=100)
train.py                 argparse CLI wrapping Config → trainer.train()
notebooks/tutorial.ipynb 7-section guided walkthrough
```

## shapes3d Factor Structure

- **480,000 images** (64×64 RGB), stored as HDF5 at `data/shapes3d/3dshapes.h5`
- Factor ordering in the HDF5: `[floor_hue(10), wall_hue(10), object_hue(10), scale(8), shape(4), orientation(15)]`
- Flat index formula: `scene_idx * 15 + orientation_idx` (orientation is fastest-changing)
- Source factors (shared): indices 0–4; Noise (orientation): index 5

## Expected Training Behaviour

| Metric | Epoch 1 | Epoch 20 | Epoch 100 |
|--------|---------|----------|-----------|
| BCE loss | ~0.69 | <0.30 | <0.10 |
| Accuracy | ~50% | >85% | >98% |
| MCC | ~0.1 | ~0.4 | ~0.7–0.9 |

MCC > 0.6 indicates meaningful source recovery. Linear probe R² > 0.7 for continuous factors (hues, scale).
