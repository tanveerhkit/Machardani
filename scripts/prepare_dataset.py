#!/usr/bin/env python
"""
Prepare raw mosquito wingbeat datasets for training.

Example usage:

    python scripts/prepare_dataset.py \
        --dataset-name humbugdb \
        --source data/raw/humbugdb \
        --output data/processed/humbugdb \
        --metadata data/raw/humbugdb/metadata.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mosquito_monitor.data import DatasetPreparer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize mosquito audio datasets.")
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Human-friendly dataset name stored in the manifest.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the raw dataset folder containing audio files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for normalized audio + metadata.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional CSV/JSON providing filename->species mappings.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Target sample rate for all clips.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Discard clips shorter than this many seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=5.0,
        help="Discard clips longer than this many seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preparer = DatasetPreparer(
        dataset_name=args.dataset_name,
        source_dir=args.source,
        output_dir=args.output,
        metadata_path=args.metadata,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    report = preparer.prepare()

    print(
        f"[{report.dataset}] Prepared {report.prepared_files} clips "
        f"(discarded {report.discarded_files}). Manifest: {report.manifest_path}"
    )


if __name__ == "__main__":
    main()
