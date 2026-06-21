"""
Training loop for multi-view ICA.

Binary NCE loss (from the paper's contrastive learning formulation):
  L = -E[log σ(r(x1, x2))]   ← positive pair (same source)
    - E[log(1 - σ(r(x1, x2_neg)))]  ← negative pair (different source)

This is equivalent to F.binary_cross_entropy_with_logits on concatenated
logits with labels [1, ..., 1, 0, ..., 0].
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data import Shapes3DPairDataset, download_shapes3d, N_ORIENTATIONS
from evaluation.metrics import compute_mcc, linear_probe_r2
from models import MultiViewDiscriminator


def train_epoch(
    model: MultiViewDiscriminator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """One training epoch. Returns (avg_loss, avg_accuracy)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x1, x2, x2_neg, _ in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        x2_neg = x2_neg.to(device)
        B = x1.size(0)

        logit_pos = model(x1, x2)       # (B,)
        logit_neg = model(x1, x2_neg)   # (B,)

        logits = torch.cat([logit_pos, logit_neg])
        targets = torch.cat([torch.ones(B, device=device), torch.zeros(B, device=device)])

        loss = F.binary_cross_entropy_with_logits(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (logits > 0).float()
            total_correct += (preds == targets).sum().item()
            total_samples += logits.size(0)
            total_loss += loss.item() * logits.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate_representations(
    model: MultiViewDiscriminator,
    dataset: Shapes3DPairDataset,
    device: str,
    max_samples: int = 10000,
    batch_size: int = 512,
) -> dict:
    """Encode a fixed subset of images and compute MCC + linear R²."""
    model.eval()
    images, labels = dataset.get_all_latents_and_labels(max_samples)

    # Encode in batches to avoid OOM
    all_z = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        z = model.encode(batch).cpu().numpy()
        all_z.append(z)
    z_hat = np.concatenate(all_z, axis=0)

    mcc, assignment, corr_matrix = compute_mcc(z_hat, labels)
    r2 = linear_probe_r2(z_hat, labels)
    orient_var = _orientation_sensitivity(model, dataset, device, n_scenes=100, batch_size=batch_size)

    return {
        "mcc": mcc,
        "assignment": assignment,
        "corr_matrix": corr_matrix,
        "r2": r2,
        "orient_var": orient_var,
        "z_hat": z_hat,
        "labels": labels,
    }


@torch.no_grad()
def _orientation_sensitivity(
    model: MultiViewDiscriminator,
    dataset: Shapes3DPairDataset,
    device: str,
    n_scenes: int = 100,
    batch_size: int = 512,
) -> float:
    """Mean variance of latent codes across orientations for fixed scenes.

    Measures how much the encoder responds to orientation (the noise variable).
    Lower is better — the encoder should be invariant to orientation.
    """
    model.eval()
    scene_indices = dataset.scene_indices[:n_scenes]
    variances = []

    for s_idx in scene_indices:
        imgs = torch.stack([
            dataset._get_image(int(s_idx), o) for o in range(N_ORIENTATIONS)
        ]).to(device)
        codes = []
        for start in range(0, len(imgs), batch_size):
            codes.append(model.encode(imgs[start:start + batch_size]).cpu().numpy())
        codes = np.concatenate(codes, axis=0)   # (N_ORIENTATIONS, latent_dim)
        variances.append(float(codes.var(axis=0).mean()))

    return float(np.mean(variances))


def train(cfg: Config):
    """Full training loop with periodic evaluation and checkpointing."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # --- Data ---
    if not os.path.exists(cfg.hdf5_path):
        download_shapes3d(cfg.data_dir)

    print("Loading dataset...")
    train_ds = Shapes3DPairDataset(cfg.hdf5_path, split="train", train_frac=cfg.train_frac,
                                   seed=cfg.seed, hard_neg_prob=cfg.hard_neg_prob,
                                   augment=cfg.augment)
    val_ds   = Shapes3DPairDataset(cfg.hdf5_path, split="val",   train_frac=cfg.train_frac,
                                   seed=cfg.seed, hard_neg_prob=cfg.hard_neg_prob,
                                   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device != "cpu"),
        drop_last=True,
    )

    print(f"Train scenes: {len(train_ds):,}   Val scenes: {len(val_ds):,}")
    print(f"Device: {device}")

    # --- Model ---
    model = MultiViewDiscriminator(latent_dim=cfg.latent_dim, head_hidden=cfg.head_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-5)

    # --- History ---
    losses: list[float] = []
    accuracies: list[float] = []
    mccs: dict[int, float] = {}
    best_mcc = -1.0

    print(f"\nStarting training for {cfg.epochs} epochs...\n")
    for epoch in range(1, cfg.epochs + 1):
        loss, acc = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        losses.append(loss)
        accuracies.append(acc)

        print(f"Epoch {epoch:3d}/{cfg.epochs} | loss {loss:.4f} | acc {acc:.3f}", end="")

        # --- Periodic evaluation ---
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            results = evaluate_representations(
                model, val_ds, device,
                max_samples=cfg.eval_samples,
                batch_size=cfg.batch_size,
            )
            mcc = results["mcc"]
            r2 = results["r2"]
            orient_var = results["orient_var"]
            corr_matrix = results["corr_matrix"]
            mccs[epoch] = mcc
            r2_str = " ".join(f"{v:.2f}" for v in r2)
            print(f" | MCC {mcc:.4f} | orient_var {orient_var:.4f} | R² [{r2_str}]", end="")
            # Per-column max of the correlation matrix reveals dimensional collapse:
            # if any factor's max correlation is low, no latent dimension tracks it.
            col_max_str = " ".join(f"{corr_matrix[:, j].max():.2f}" for j in range(corr_matrix.shape[1]))
            print(f"\n  corr_max/factor [{col_max_str}]", end="")

            if mcc > best_mcc:
                best_mcc = mcc
                best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                torch.save({"epoch": epoch, "model": model.state_dict(), "mcc": mcc}, best_path)

        print()

        # --- Regular checkpoint ---
        if epoch % 10 == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "losses": losses,
                "accuracies": accuracies,
                "mccs": mccs,
            }, ckpt_path)

    print(f"\nTraining complete. Best MCC: {best_mcc:.4f}")
    print(f"Best checkpoint: {os.path.join(cfg.checkpoint_dir, 'best.pt')}")

    # --- Save training history ---
    history_path = os.path.join(cfg.checkpoint_dir, "history.npz")
    np.savez(history_path, losses=losses, accuracies=accuracies,
             mcc_epochs=list(mccs.keys()), mcc_values=list(mccs.values()))
    print(f"History saved: {history_path}")

    return model, losses, accuracies, mccs


def load_checkpoint(
    checkpoint_path: str,
    cfg: Optional[Config] = None,
    device: str = "cpu",
) -> MultiViewDiscriminator:
    """Load a saved model from a checkpoint file."""
    if cfg is None:
        cfg = Config()
    model = MultiViewDiscriminator(latent_dim=cfg.latent_dim, head_hidden=cfg.head_hidden)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
