#!/usr/bin/env python
"""
Train the baseline CNN classifier on cached log-mel features.

Example:
    python scripts/train_model.py \
        --features data/features/humbugdb/features.csv \
        --output checkpoints/humbugdb_baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mosquito_monitor.model import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mosquito species classifier.")
    parser.add_argument(
        "--features",
        required=True,
        type=Path,
        help="Path to features.csv produced by scripts/build_features.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where checkpoints and logs are stored.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cpu/cuda). Defaults to CUDA if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainerConfig(
        manifest=args.features,
        output_dir=args.output,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        num_workers=args.num_workers,
        device=device,
    )
    trainer = Trainer(config)
    artifacts = trainer.run()
    print(f"Training finished. Checkpoint: {artifacts.checkpoint_path}")


if __name__ == "__main__":
    main()
