"""
CNN encoder: h(x) -> z.

Maps a 64x64x3 image to a latent_dim-dimensional vector.
Per Theorems 1 and 4 in Gresele et al. (2020), this encoder learns to
invert the mixing function f and recover the independent source components
up to component-wise invertible transformations, provided it is constrained
to use an invertible architecture. We use a standard CNN in practice, relying
on the network capacity and contrastive objective to approximate this.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """4-layer convolutional encoder from 64x64x3 images to latent_dim vectors."""

    def __init__(self, latent_dim: int = 10):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            # 3 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 32 x 32 -> 128 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 16 x 16 -> 256 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 8 x 8 -> 512 x 4 x 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 512 * 4 * 4 = 8192
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))
