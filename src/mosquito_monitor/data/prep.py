from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

SUPPORTED_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


@dataclass
class ClipRecord:
    """Normalized description of a wingbeat clip."""

    clip_id: str
    species: str
    dataset: str
    source_path: Path
    prepared_path: Path
    duration_seconds: float
    split: str = "unspecified"
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class PreparationReport:
    """Summary statistics returned after running the dataset preparer."""

    dataset: str
    total_files: int
    prepared_files: int
    discarded_files: int
    output_path: Path
    manifest_path: Path
    discarded_manifest_path: Optional[Path]


class DatasetPreparer:
    """
    Convert heterogenous mosquito audio datasets into a canonical structure.

    Responsibilities:
    - Resample clips to the target sample rate.
    - Convert to mono waveform and normalize amplitude.
    - Filter clips outside the desired duration bounds.
    - Persist a manifest CSV for downstream preprocessing.
    """

    def __init__(
        self,
        dataset_name: str,
        source_dir: Path,
        output_dir: Path,
        metadata_path: Optional[Path] = None,
        sample_rate: int = 16_000,
        min_duration: float = 0.5,
        max_duration: float = 5.0,
    ) -> None:
        self.dataset_name = dataset_name
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration

    def prepare(self) -> PreparationReport:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        audio_output = self.output_dir / "audio"
        audio_output.mkdir(exist_ok=True)

        metadata_map = self._build_metadata_map()

        manifest_records: List[ClipRecord] = []
        discarded_records: List[ClipRecord] = []

        audio_files = self._iter_audio_files()
        for audio_path in tqdm(audio_files, desc=f"[{self.dataset_name}] Preparing"):
            rel = audio_path.relative_to(self.source_dir)
            meta = metadata_map.get(rel.as_posix())
            species = self._infer_species(meta, rel)
            clip_id = self._infer_clip_id(meta, rel)
            split = meta.get("split", "unspecified") if meta else "unspecified"

            prepared_path = audio_output / species / f"{clip_id}.wav"
            prepared_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                audio, _ = librosa.load(audio_path.as_posix(), sr=self.sample_rate, mono=True)
            except Exception as exc:  # pragma: no cover - defensive path
                discarded_records.append(
                    ClipRecord(
                        clip_id=clip_id,
                        species=species,
                        dataset=self.dataset_name,
                        source_path=audio_path,
                        prepared_path=prepared_path,
                        duration_seconds=0.0,
                        split=split,
                        extra={"reason": f"load_error:{exc}"},
                    )
                )
                continue

            duration_seconds = float(len(audio) / self.sample_rate)
            if duration_seconds < self.min_duration or duration_seconds > self.max_duration:
                discarded_records.append(
                    ClipRecord(
                        clip_id=clip_id,
                        species=species,
                        dataset=self.dataset_name,
                        source_path=audio_path,
                        prepared_path=prepared_path,
                        duration_seconds=duration_seconds,
                        split=split,
                        extra={"reason": "duration_out_of_bounds"},
                    )
                )
                continue

            # Normalize amplitude to [-1, 1]
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

            ints = np.clip(audio, -1.0, 1.0)
            wavfile.write(prepared_path.as_posix(), self.sample_rate, (ints * 32767).astype(np.int16))

            manifest_records.append(
                ClipRecord(
                    clip_id=clip_id,
                    species=species,
                    dataset=self.dataset_name,
                    source_path=audio_path,
                    prepared_path=prepared_path,
                    duration_seconds=duration_seconds,
                    split=split,
                    extra=meta or {},
                )
            )

        manifest_path = self.output_dir / "metadata.csv"
        self._write_manifest(manifest_path, manifest_records)

        discarded_manifest_path = None
        if discarded_records:
            discarded_manifest_path = self.output_dir / "discarded.csv"
            self._write_manifest(discarded_manifest_path, discarded_records)

        return PreparationReport(
            dataset=self.dataset_name,
            total_files=len(list(self._iter_audio_files())),
            prepared_files=len(manifest_records),
            discarded_files=len(discarded_records),
            output_path=self.output_dir,
            manifest_path=manifest_path,
            discarded_manifest_path=discarded_manifest_path,
        )

    def _iter_audio_files(self) -> Iterable[Path]:
        return sorted(
            p
            for p in self.source_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def _build_metadata_map(self) -> Dict[str, Dict[str, str]]:
        if not self.metadata_path:
            return {}

        if self.metadata_path.suffix.lower() == ".json":
            with self.metadata_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                rows = data
            else:
                rows = data.get("records", [])
        else:
            rows = pd.read_csv(self.metadata_path).to_dict(orient="records")

        metadata_map: Dict[str, Dict[str, str]] = {}
        for row in rows:
            filename = row.get("filename") or row.get("file") or row.get("clip")
            if not filename:
                continue
            key = filename.replace("\\", "/")
            metadata_map[key] = row
        return metadata_map

    def _infer_species(self, metadata: Optional[Dict[str, str]], relative_path: Path) -> str:
        species = None
        if metadata:
            raw_species = metadata.get("species") or metadata.get("label")
            if raw_species:
                species = self._slugify(raw_species)
        if not species:
            species = self._slugify(relative_path.parent.name)
        return species or "unknown"

    def _infer_clip_id(self, metadata: Optional[Dict[str, str]], relative_path: Path) -> str:
        if metadata:
            clip_id = (
                metadata.get("clip_id")
                or metadata.get("id")
                or metadata.get("uuid")
                or metadata.get("filename")
            )
            if clip_id:
                return self._slugify(Path(clip_id).stem)
        return self._slugify(relative_path.stem)

    @staticmethod
    def _slugify(value: str) -> str:
        return (
            value.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

    def _write_manifest(self, manifest_path: Path, records: List[ClipRecord]) -> None:
        fieldnames = [
            "clip_id",
            "species",
            "dataset",
            "source_path",
            "prepared_path",
            "duration_seconds",
            "split",
        ]
        extra_keys = sorted(
            {
                key
                for record in records
                for key in record.extra.keys()
                if key not in fieldnames
            }
        )

        with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames + extra_keys)
            writer.writeheader()
            for record in records:
                row = {
                    "clip_id": record.clip_id,
                    "species": record.species,
                    "dataset": record.dataset,
                    "source_path": record.source_path.as_posix(),
                    "prepared_path": record.prepared_path.as_posix(),
                    "duration_seconds": f"{record.duration_seconds:.4f}",
                    "split": record.split,
                }
                for key in extra_keys:
                    row[key] = record.extra.get(key, "")
                writer.writerow(row)
