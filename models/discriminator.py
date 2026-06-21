"""
Multi-view discriminator: r(x1, x2) = Σ_i ψ_i(h_i(x1), h_i(x2)).

Implements the regression function from Theorem 4 (Gresele et al., 2020).
A weight-tied encoder maps both views to latent codes; the head MUST decompose
as a sum of per-dimension terms — each ψ_i sees only (z1_i, z2_i). This is the
key structural constraint that makes the encoder's recovery of independent sources
identifiable. A generic MLP over cat([z1, z2]) breaks the theorem by allowing
cross-dimensional shortcuts that achieve high accuracy without disentanglement.
"""

import torch
import torch.nn as nn

from .encoder import Encoder


class MultiViewDiscriminator(nn.Module):
    """Contrastive discriminator for multi-view ICA.

    Architecture:
        z1 = encoder(x1)                              # h(x1) ∈ R^latent_dim
        z2 = encoder(x2)                              # h(x2) ∈ R^latent_dim  (weight-tied)
        logit = Σ_i ψ_i(z1[i], z2[i])               # Theorem 4 decomposition

    Each ψ_i is a small MLP that operates on a single dimension pair only.
    This prevents cross-dimensional shortcuts and forces the encoder to put
    independently-recoverable information in each latent dimension.
    """

    def __init__(self, latent_dim: int = 10, head_hidden: int = 256):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim

        # One small MLP per latent dimension: (z1_i, z2_i) → scalar score.
        # head_hidden is the hidden size of each ψ_i.
        self.psis = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, head_hidden),
                nn.LeakyReLU(0.2),
                nn.Linear(head_hidden, head_hidden // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(head_hidden // 2, 1),
            )
            for _ in range(latent_dim)
        ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the latent representation of an image."""
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Return raw logit (no sigmoid). Use BCEWithLogitsLoss during training."""
        z1 = self.encoder(x1)   # (B, D)
        z2 = self.encoder(x2)   # (B, D)

        # r(x1, x2) = Σ_i ψ_i(z1_i, z2_i)  — Theorem 4
        score = torch.zeros(z1.size(0), device=z1.device)
        for i, psi in enumerate(self.psis):
            pair = torch.stack([z1[:, i], z2[:, i]], dim=-1)  # (B, 2)
            score = score + psi(pair).squeeze(-1)
        return score
