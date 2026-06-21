"""
shapes3d pair dataset for multi-view contrastive ICA.

Factor structure (480,000 images total):
  floor_hue   : 10 values  [0.0, 0.1, ..., 0.9]
  wall_hue    : 10 values
  object_hue  : 10 values
  scale       : 8 values
  shape       : 4 values   (0=cube, 1=cylinder, 2=sphere, 3=round-cube)
  orientation : 15 values  (-30° to +30° in 4° steps)

For multi-view ICA:
  s (shared source)  = (floor_hue, wall_hue, object_hue, scale, shape)
  n (noise / view)   = orientation
  positive pair      : same s, two different orientations
  negative pair      : different s

Performance note:
  The raw HDF5 file uses gzip-4 compression with chunks=(15000,4,4,1), which
  makes random access very slow (768 compressed chunk reads per image). On
  first use, `_ensure_image_cache` converts the images to a flat uncompressed
  numpy memmap (~5.5 GB on disk) for O(1) random access.
"""

import os
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

SHAPES3D_URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"

FACTOR_NAMES = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
SOURCE_FACTOR_NAMES = ["floor_hue", "wall_hue", "object_hue", "scale", "shape"]
FACTOR_SIZES = [10, 10, 10, 8, 4, 15]
SOURCE_FACTOR_SIZES = FACTOR_SIZES[:5]   # 10*10*10*8*4 = 32 000 unique scenes
N_ORIENTATIONS = 15
N_SCENES = 10 * 10 * 10 * 8 * 4  # 32 000
N_IMAGES = N_SCENES * N_ORIENTATIONS  # 480 000

IMG_SHAPE = (64, 64, 3)
CACHE_FNAME = "images_cache.npy"   # flat uncompressed memmap alongside the HDF5


def download_shapes3d(data_dir: str = "data/shapes3d") -> str:
    """Download the shapes3d HDF5 file if not already present. Returns path."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "3dshapes.h5"

    if dest.exists():
        print(f"Dataset already present at {dest}")
        return str(dest)

    print(f"Downloading shapes3d (~2 GB) to {dest} ...")

    def _progress(block_count, block_size, total_size):
        downloaded = block_count * block_size
        pct = min(100.0, 100.0 * downloaded / total_size) if total_size > 0 else 0
        bar = int(pct / 2)
        print(f"\r[{'=' * bar}{' ' * (50 - bar)}] {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(SHAPES3D_URL, dest, _progress)
    print()
    print(f"Download complete: {dest}")
    return str(dest)


def _ensure_image_cache(hdf5_path: str) -> str:
    """Build a flat uncompressed numpy memmap from the HDF5 images dataset.

    The raw HDF5 uses gzip chunks that make random access ~270ms/image.
    A flat uint8 memmap reduces this to a single disk seek (<1ms/image).

    The cache file is placed next to the HDF5 (~5.5 GB on disk).
    Returns the path to the cache file.
    """
    cache_path = str(Path(hdf5_path).parent / CACHE_FNAME)
    if os.path.exists(cache_path):
        return cache_path

    print(f"Building image cache (one-time, ~5.5 GB) at {cache_path} ...")
    mm = np.memmap(cache_path, dtype="uint8", mode="w+",
                   shape=(N_IMAGES, *IMG_SHAPE))

    chunk = 15000  # matches HDF5 chunk size for sequential efficiency
    with h5py.File(hdf5_path, "r") as f:
        for start in tqdm(range(0, N_IMAGES, chunk), desc="Converting"):
            end = min(start + chunk, N_IMAGES)
            mm[start:end] = f["images"][start:end]

    del mm   # flush to disk
    print(f"Cache ready: {cache_path}")
    return cache_path


def _flat_index(s_idx: int, n_idx: int) -> int:
    """Convert (scene_index, orientation_index) to the flat image row index."""
    return s_idx * N_ORIENTATIONS + n_idx


# Mixed-radix strides for the 5 source factors (floor_hue slowest, shape fastest)
SOURCE_FACTOR_STRIDES = [3200, 320, 32, 4, 1]


def _scene_to_factors(s_idx: int) -> list[int]:
    """Decode a flat scene index into its 5 factor indices."""
    factors = []
    remaining = s_idx
    for stride, size in zip(SOURCE_FACTOR_STRIDES, SOURCE_FACTOR_SIZES):
        factors.append(remaining // stride)
        remaining = remaining % stride
    return factors


def _hard_negative_scene(s_idx: int, rng: np.random.Generator) -> int:
    """Return a scene that differs from s_idx in exactly one randomly chosen factor.

    This forces the encoder to precisely recover each factor individually,
    preventing the model from relying on multi-factor differences as shortcuts.
    """
    factors = _scene_to_factors(s_idx)
    fi = int(rng.integers(0, len(SOURCE_FACTOR_SIZES)))
    size = SOURCE_FACTOR_SIZES[fi]
    offset = int(rng.integers(1, size))
    factors[fi] = (factors[fi] + offset) % size
    return sum(f * s for f, s in zip(factors, SOURCE_FACTOR_STRIDES))


class Shapes3DPairDataset(Dataset):
    """Returns contrastive triplets (x1, x2, x2_neg) for multi-view ICA training.

    x1, x2  : same scene s, different orientations  (positive pair)
    x2_neg  : different scene s*, any orientation    (negative pair)

    Also returns s_labels (5-dim vector of the 5 source factor values for x1/x2)
    for evaluation purposes only — not used during training.
    """

    def __init__(
        self,
        hdf5_path: str,
        split: str = "train",
        train_frac: float = 0.9,
        seed: int = 42,
        hard_neg_prob: float = 0.5,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.split = split

        # Build or open the fast image cache
        cache_path = _ensure_image_cache(hdf5_path)
        self.images = np.memmap(cache_path, dtype="uint8", mode="r",
                                shape=(N_IMAGES, *IMG_SHAPE))

        # Load only labels into RAM (~11 MB)
        with h5py.File(hdf5_path, "r") as f:
            self.labels = f["labels"][:]   # (480000, 6) float32

        # scene indices in [0, N_SCENES)
        rng = np.random.default_rng(seed)
        scene_perm = rng.permutation(N_SCENES)
        n_train = int(N_SCENES * train_frac)
        if split == "train":
            self.scene_indices = scene_perm[:n_train]
        else:
            self.scene_indices = scene_perm[n_train:]

        self.hard_neg_prob = hard_neg_prob
        self._rng = np.random.default_rng(seed + (0 if split == "train" else 1))

    def __len__(self) -> int:
        return len(self.scene_indices)

    def __getitem__(self, idx: int):
        rng = self._rng

        s_idx = int(self.scene_indices[idx])

        # sample two distinct orientations for the positive pair
        n1, n2 = rng.choice(N_ORIENTATIONS, size=2, replace=False)

        x1 = self._get_image(s_idx, n1)
        x2 = self._get_image(s_idx, n2)

        # negative: hard (differ in exactly 1 factor) or easy (fully random)
        if rng.random() < self.hard_neg_prob:
            neg_s_idx = _hard_negative_scene(s_idx, rng)
        else:
            neg_s_idx = s_idx
            while neg_s_idx == s_idx:
                neg_s_idx = int(rng.integers(0, N_SCENES))
        neg_n = int(rng.integers(0, N_ORIENTATIONS))
        x2_neg = self._get_image(neg_s_idx, neg_n)

        # source factor labels for x1/x2 (used only in evaluation)
        flat = _flat_index(s_idx, n1)
        s_labels = torch.tensor(self.labels[flat, :5], dtype=torch.float32)

        return x1, x2, x2_neg, s_labels

    def _get_image(self, s_idx: int, n_idx: int) -> torch.Tensor:
        flat = _flat_index(s_idx, n_idx)
        img = np.array(self.images[flat])   # copy out of memmap
        return torch.from_numpy(img).float().div_(255.0).permute(2, 0, 1)

    def get_all_latents_and_labels(self, max_samples: int = 10000):
        """Return a fixed subset of raw images and their 5 source labels.
        Used for computing MCC and linear-probe R² after training.
        """
        n = min(max_samples, len(self.scene_indices))
        indices = self.scene_indices[:n]
        flat_indices = np.array([_flat_index(int(s), 0) for s in indices])
        imgs = np.array(self.images[flat_indices])   # (N, 64, 64, 3) uint8
        images = torch.from_numpy(imgs).float().div_(255.0).permute(0, 3, 1, 2)
        labels = self.labels[flat_indices, :5]       # (N, 5)
        return images, labels
