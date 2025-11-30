#!/usr/bin/env python
"""
Generate cached log-mel spectrograms from prepared mosquito audio manifests.

Example:
    python scripts/build_features.py \
        --manifest data/processed/humbugdb/metadata.csv \
        --output data/features/humbugdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mosquito_monitor.audio import FeatureConfig, FeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build log-mel spectrogram features.")
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to metadata.csv created by the dataset preparer.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where .npy feature tensors and manifests are stored.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Audio sample rate expected by the model.",
    )
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--fmin", type=int, default=50)
    parser.add_argument("--fmax", type=int, default=4_000)
    parser.add_argument("--top-db", type=int, default=80)
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=2.0,
        help="Pad/trim clips to this duration before feature extraction.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even when cached .npy files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FeatureConfig(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        top_db=args.top_db,
        target_seconds=args.target_seconds,
    )
    extractor = FeatureExtractor(config)
    report = extractor.process_manifest(
        manifest_path=args.manifest,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(
        f"Processed {report.processed_clips}/{report.total_clips} clips "
        f"(skipped {report.skipped_clips}, failed {report.failed_clips}). "
        f"Manifest: {report.manifest_path}"
    )


if __name__ == "__main__":
    main()
