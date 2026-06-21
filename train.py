"""CLI entry point for multi-view ICA training on shapes3d."""

import argparse

from config import Config
from training import train


def main():
    parser = argparse.ArgumentParser(description="Train multi-view ICA on shapes3d")
    parser.add_argument("--data_dir", default="data/shapes3d")
    parser.add_argument("--hdf5_path", default="")
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_samples", type=int, default=10000)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        hdf5_path=args.hdf5_path,
        latent_dim=args.latent_dim,
        head_hidden=args.head_hidden,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        eval_every=args.eval_every,
        eval_samples=args.eval_samples,
        checkpoint_dir=args.checkpoint_dir,
    )
    if args.device:
        cfg.device = args.device

    train(cfg)


if __name__ == "__main__":
    main()
