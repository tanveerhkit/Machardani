from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from tqdm import tqdm


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@dataclass
class FeatureConfig:
    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    fmin: int = 50
    fmax: int = 4_000
    top_db: int = 80
    target_seconds: Optional[float] = 2.0

    @property
    def target_samples(self) -> Optional[int]:
        if self.target_seconds is None:
            return None
        return int(self.sample_rate * self.target_seconds)


@dataclass
class FeatureBuildReport:
    total_clips: int
    processed_clips: int
    skipped_clips: int
    failed_clips: int
    manifest_path: Path
    failed_manifest_path: Optional[Path]


class FeatureExtractor:
    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()
        self._mel_filterbank = self._build_mel_filterbank()

    def _build_mel_filterbank(self) -> torch.Tensor:
        n_fft = self.config.n_fft
        n_mels = self.config.n_mels
        sr = self.config.sample_rate
        fmin = self.config.fmin
        fmax = self.config.fmax

        mel_min = _hz_to_mel(fmin)
        mel_max = _hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = _mel_to_hz(mel_points)
        fft_bins = np.linspace(0, sr / 2, n_fft // 2 + 1)
        filter_bank = np.zeros((n_mels, len(fft_bins)), dtype=np.float32)

        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            left_bin = np.searchsorted(fft_bins, left)
            center_bin = np.searchsorted(fft_bins, center)
            right_bin = np.searchsorted(fft_bins, right)
            denom_left = max(center - left, 1e-6)
            denom_right = max(right - center, 1e-6)

            for j in range(left_bin, center_bin):
                if 0 <= j < len(fft_bins):
                    filter_bank[i, j] = (fft_bins[j] - left) / denom_left
            for j in range(center_bin, right_bin):
                if 0 <= j < len(fft_bins):
                    filter_bank[i, j] = (right - fft_bins[j]) / denom_right

        return torch.from_numpy(filter_bank)

    def process_manifest(
        self, manifest_path: Path, output_dir: Path, overwrite: bool = False
    ) -> FeatureBuildReport:
        manifest_path = Path(manifest_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(manifest_path)
        records = df.to_dict(orient="records")

        feature_records: List[Dict[str, object]] = []
        failed_records: List[Dict[str, object]] = []
        processed = skipped = 0

        for row in tqdm(records, desc="Extracting log-mel features"):
            clip_id = str(row.get("clip_id") or row.get("id") or row.get("uuid") or "")
            dataset = str(row.get("dataset") or "unknown")
            species = str(row.get("species") or "unknown")
            if not clip_id:
                clip_id = Path(str(row.get("prepared_path", "clip"))).stem

            feature_path = output_dir / dataset / species / f"{clip_id}.npy"
            feature_path.parent.mkdir(parents=True, exist_ok=True)

            if feature_path.exists() and not overwrite:
                skipped += 1
                existing = np.load(feature_path.as_posix())
                feature_records.append(
                    self._feature_row(
                        clip_id, dataset, species, feature_path, existing.shape[1], cached=True
                    )
                )
                continue

            try:
                mel_spec, n_frames = self._build_feature(manifest_path.parent, row)
            except Exception as exc:
                failed_records.append(
                    {
                        "clip_id": clip_id,
                        "dataset": dataset,
                        "species": species,
                        "reason": str(exc),
                    }
                )
                continue

            np.save(feature_path.as_posix(), mel_spec.astype(np.float32))
            processed += 1
            feature_records.append(
                self._feature_row(clip_id, dataset, species, feature_path, n_frames, cached=False)
            )

        feature_manifest_path = output_dir / "features.csv"
        pd.DataFrame(feature_records).to_csv(feature_manifest_path, index=False)

        failed_manifest_path = None
        if failed_records:
            failed_manifest_path = output_dir / "features_failed.csv"
            pd.DataFrame(failed_records).to_csv(failed_manifest_path, index=False)

        return FeatureBuildReport(
            total_clips=len(records),
            processed_clips=processed,
            skipped_clips=skipped,
            failed_clips=len(failed_records),
            manifest_path=feature_manifest_path,
            failed_manifest_path=failed_manifest_path,
        )

    def _build_feature(self, manifest_dir: Path, row: Dict[str, object]) -> Tuple[np.ndarray, int]:
        audio_path = self._resolve_audio_path(manifest_dir, row)
        sample_rate, audio = wavfile.read(audio_path.as_posix())
        audio = self._normalize_audio(audio)
        if sample_rate != self.config.sample_rate:
            audio = self._resample(audio, sample_rate)
        audio = self._pad_or_trim(audio)
        log_mel = self._mel_from_tensor(torch.from_numpy(audio).unsqueeze(0))
        return log_mel, log_mel.shape[1]

    def transform_audio(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        normalized = self._normalize_audio(audio)
        if sample_rate != self.config.sample_rate:
            normalized = self._resample(normalized, sample_rate)
        normalized = self._pad_or_trim(normalized)
        log_mel = self._mel_from_tensor(torch.from_numpy(normalized).unsqueeze(0))
        return log_mel, log_mel.shape[1]

    def _pad_or_trim(self, audio: np.ndarray) -> np.ndarray:
        target_samples = self.config.target_samples
        if target_samples is None:
            return audio
        if len(audio) > target_samples:
            return audio[:target_samples]
        if len(audio) < target_samples:
            padding = np.zeros(target_samples - len(audio), dtype=audio.dtype)
            return np.concatenate([audio, padding])
        return audio

    def _mel_from_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        stft = torch.stft(
            tensor,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )
        power = torch.abs(stft) ** 2
        power = power[:, : self.config.n_fft // 2 + 1]
        mel = torch.matmul(self._mel_filterbank, power[0])
        log_mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        log_mel = log_mel - torch.max(log_mel)
        log_mel = torch.clamp(log_mel, min=-self.config.top_db)
        return log_mel.numpy()

    def _resample(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if len(audio) == 0:
            return audio
        duration = len(audio) / sample_rate
        target_length = int(self.config.sample_rate * duration)
        if target_length <= 0:
            return audio
        xp = np.linspace(0, len(audio) - 1, len(audio))
        x_new = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(x_new, xp, audio).astype(np.float32)

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        return audio

    @staticmethod
    def _feature_row(
        clip_id: str,
        dataset: str,
        species: str,
        feature_path: Path,
        n_frames: int,
        cached: bool,
    ) -> Dict[str, object]:
        return {
            "clip_id": clip_id,
            "dataset": dataset,
            "species": species,
            "feature_path": feature_path.as_posix(),
            "frames": n_frames,
            "cached": cached,
        }

    @staticmethod
    def _resolve_audio_path(manifest_dir: Path, row: Dict[str, object]) -> Path:
        candidates = [
            row.get("prepared_path"),
            row.get("clip_path"),
            row.get("audio_path"),
            row.get("path"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(str(candidate))
            if path.exists():
                return path
            if not path.is_absolute():
                alt = (manifest_dir / path).resolve()
                if alt.exists():
                    return alt
        raise FileNotFoundError(
            f"Unable to locate audio file for row with clip_id={row.get('clip_id')}"
        )
