from dataclasses import dataclass, field
import torch


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # data
    data_dir: str = "data/shapes3d"
    hdf5_path: str = ""           # set automatically if empty
    train_frac: float = 0.9

    # model
    latent_dim: int = 5
    head_hidden: int = 256

    # negative sampling
    hard_neg_prob: float = 0.5   # fraction of negatives differing in exactly 1 factor

    # augmentation
    augment: bool = True         # per-view ColorJitter + GaussianBlur + RandomErasing
    same_orientation: bool = False  # True → orientation is a 6th source factor; noise = augmentations only

    # training
    batch_size: int = 256
    lr: float = 3e-4
    epochs: int = 100
    seed: int = 42
    num_workers: int = 4

    # evaluation
    eval_every: int = 5           # epochs between MCC evaluations
    eval_samples: int = 10000

    # I/O
    checkpoint_dir: str = "checkpoints"
    device: str = field(default_factory=_default_device)

    def __post_init__(self):
        if not self.hdf5_path:
            import os
            self.hdf5_path = os.path.join(self.data_dir, "3dshapes.h5")
