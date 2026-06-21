"""Visualization utilities for multi-view ICA results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

SOURCE_FACTOR_NAMES = ["floor_hue", "wall_hue", "object_hue", "scale", "shape"]


def plot_tsne(
    z: np.ndarray,
    labels: np.ndarray,
    factor_names: list[str] = SOURCE_FACTOR_NAMES,
    save_path: Optional[str] = None,
    method: str = "umap",
):
    """2-D embedding of the latent space, coloured by each source factor.

    Args:
        z:            (N, latent_dim) latent codes
        labels:       (N, n_factors)  ground-truth factor values
        method:       'umap' (default, faster) or 'tsne'
        save_path:    if given, save figure there; otherwise plt.show()
    """
    if method == "umap":
        from umap import UMAP
        emb = UMAP(n_components=2, random_state=42).fit_transform(z)
    else:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(z)

    n_factors = labels.shape[1]
    fig, axes = plt.subplots(1, n_factors, figsize=(4 * n_factors, 4))
    if n_factors == 1:
        axes = [axes]

    for ax, name, j in zip(axes, factor_names, range(n_factors)):
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=labels[:, j], cmap="viridis", s=2, alpha=0.6)
        plt.colorbar(sc, ax=ax)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Latent space ({method.upper()}) coloured by source factors", y=1.02)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_factor_scatter(
    z: np.ndarray,
    s: np.ndarray,
    assignment: np.ndarray,
    factor_names: list[str] = SOURCE_FACTOR_NAMES,
    save_path: Optional[str] = None,
):
    """Scatter z[assignment[j]] vs s[j] for each source factor j.

    Shows how well each assigned latent dimension tracks its corresponding
    ground-truth source (after optimal MCC permutation).
    """
    n_factors = len(factor_names)
    fig, axes = plt.subplots(1, n_factors, figsize=(4 * n_factors, 4))
    if n_factors == 1:
        axes = [axes]

    for ax, name, j in zip(axes, factor_names, range(n_factors)):
        z_j = z[:, assignment[j]]
        s_j = s[:, j]
        corr = np.corrcoef(z_j, s_j)[0, 1]
        ax.scatter(s_j, z_j, s=2, alpha=0.4, c="steelblue")
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"z[{assignment[j]}]")
        ax.set_title(f"|r| = {abs(corr):.3f}")

    fig.suptitle("Assigned latent dimensions vs ground-truth source factors")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_latent_traversal(
    encoder,
    images: torch.Tensor,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    n_show: int = 8,
    device: str = "cpu",
):
    """Show images from n_show different scenes side-by-side to give visual intuition.

    Rows = different scenes; shown alongside their latent codes.
    This is a lightweight sanity check that the encoder produces distinct codes
    for different scenes.
    """
    encoder.eval()
    images = images[:n_show].to(device)
    with torch.no_grad():
        z = encoder(images).cpu().numpy()

    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))
    for i in range(n_show):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(img)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"scene {i}", fontsize=8)

        axes[1, i].bar(range(len(z[i])), z[i], color="steelblue")
        axes[1, i].set_ylim(z.min() - 0.5, z.max() + 0.5)
        axes[1, i].set_xticks([])

    axes[0, 0].set_ylabel("image", fontsize=9)
    axes[1, 0].set_ylabel("latent z", fontsize=9)
    fig.suptitle("Images and their latent codes")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_training_curves(
    losses: list[float],
    accuracies: list[float],
    mccs: dict[int, float],
    save_path: Optional[str] = None,
):
    """Plot training loss, accuracy, and MCC over epochs."""
    epochs = range(1, len(losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, losses, color="tab:blue")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Training Loss")
    axes[0].axhline(np.log(2), color="grey", linestyle="--", label="chance (log 2)")
    axes[0].legend()

    axes[1].plot(epochs, accuracies, color="tab:orange")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Discriminator Accuracy")
    axes[1].axhline(0.5, color="grey", linestyle="--", label="chance")
    axes[1].legend()
    axes[1].set_ylim(0, 1)

    if mccs:
        mcc_epochs = sorted(mccs.keys())
        mcc_vals = [mccs[e] for e in mcc_epochs]
        axes[2].plot(mcc_epochs, mcc_vals, color="tab:green", marker="o")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("MCC")
        axes[2].set_title("Mean Correlation Coefficient")
        axes[2].set_ylim(0, 1)

    fig.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig: plt.Figure, save_path: Optional[str]):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()
