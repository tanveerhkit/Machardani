#!/usr/bin/env python
"""
Convert the MosquitoSound time-series dataset into per-clip WAV files and a metadata manifest.

This script reads the provided `MosquitoSound_X.npy`/`MosquitoSound_y.npy` dump, rescales each time series,
and writes `.wav` files grouped by species so the standard preprocessing pipeline can run unchanged.

Example:

    python scripts/convert_mosquitosound.py \
        --dataset-dir data/raw/mosquitosound \
        --output data/processed/mosquitosound \
        --limit 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import wave
from tqdm import tqdm


DEFAULT_LIMIT = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MosquitoSound to WAV files.")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Path to the MosquitoSound folder (contains .npy/.csv and class map).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where normalized clips + manifest are written.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=6_000,
        help="Original sampling rate of the dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max number of clips to convert (per split).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Whether to convert the train or test split (uses train/test indices).",
    )
    return parser.parse_args()


def load_time_series(dataset_dir: Path, split: str) -> Iterable[int]:
    indices_path = dataset_dir / f"MosquitoSound_{split.capitalize()}Indices.npy"
    if not indices_path.exists():
        raise FileNotFoundError(f"Expected indices file {indices_path}")
    indices = np.load(indices_path, mmap_mode="r")
    flattened = indices.ravel()
    valid = flattened[~np.isnan(flattened)]
    unique_indices = np.unique(valid.astype(np.int64))
    return unique_indices.tolist()


def map_classes(class_map_path: Path) -> Dict[int, str]:
    df = pd.read_csv(class_map_path)
    return {int(label): name.lower().replace(" ", "_") for name, label in zip(df["class_name"], df["class_label"])}


def iter_clips(
    x_path: Path,
    y_path: Path,
    indices: Sequence[int],
    limit: int,
) -> Iterable[tuple[np.ndarray, int, int, int]]:
    signals = np.load(x_path, mmap_mode="r")
    labels = np.load(y_path, mmap_mode="r")
    count = min(len(indices), limit)
    for export_idx, idx in enumerate(indices[:count]):
        yield signals[idx], int(labels[idx]), idx, export_idx


def normalize_clip(clip: np.ndarray) -> np.ndarray:
    clip = clip.astype(np.float32)
    max_val = np.abs(clip).max()
    if max_val > 0:
        clip = clip / max_val * 0.9
    return clip


def build_clips(
    args: argparse.Namespace,
    time_series_indices: Sequence[int],
    class_map: Dict[int, str],
) -> None:
    output_audio = args.output / "audio"
    manifest_records: List[Dict[str, object]] = []

    x_path = args.dataset_dir / "MosquitoSound_X.npy"
    y_path = args.dataset_dir / "MosquitoSound_y.npy"
    for signal, label, source_idx, serial in tqdm(
        iter_clips(x_path, y_path, time_series_indices, args.limit),
        total=min(len(time_series_indices), args.limit),
        desc="Converting MosquitoSound",
    ):
        species = class_map.get(label, f"species_{label}")
        clip_id = f"{species}_{serial:06d}"
        dest = output_audio / species
        dest.mkdir(parents=True, exist_ok=True)
        clip_path = dest / f"{clip_id}.wav"
        write_wav(clip_path, normalize_clip(signal), args.sample_rate)
        manifest_records.append(
            {
                "clip_id": clip_id,
                "species": species,
                "dataset": "mosquitosound",
                "source_path": f"{args.dataset_dir}/MosquitoSound_X.npy[{source_idx}]",
                "prepared_path": clip_path.as_posix(),
                "duration_seconds": f"{len(signal)/args.sample_rate:.4f}",
                "split": args.split,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(args.output / "metadata.csv", index=False)


def write_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(data, -1.0, 1.0)
    ints = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(ints.tobytes())


def main() -> None:
    args = parse_args()
    indices = load_time_series(args.dataset_dir, args.split)
    class_map = map_classes(args.dataset_dir / "MosquitoSound_classmap.txt")
    build_clips(args, indices, class_map)


if __name__ == "__main__":
    main()
