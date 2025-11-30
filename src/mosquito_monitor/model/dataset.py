from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """PyTorch dataset backed by cached log-mel features."""

    def __init__(
        self,
        manifest_path: Path,
        label_to_index: Optional[Dict[str, int]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        df = pd.read_csv(self.manifest_path)
        self.records = df.to_dict(orient="records")

        if label_to_index is None:
            labels = sorted({str(r["species"]) for r in self.records})
            self.label_to_index = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_index = label_to_index

        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.records[index]
        feature_path = Path(row["feature_path"])
        feature = np.load(feature_path)
        tensor = torch.from_numpy(feature).unsqueeze(0)  # (1, n_mels, frames)
        if self.transform:
            tensor = self.transform(tensor)
        label_name = str(row["species"])
        label_idx = self.label_to_index[label_name]
        return tensor.float(), label_idx

    def class_weights(self) -> torch.Tensor:
        counts: Dict[int, int] = {idx: 0 for idx in self.label_to_index.values()}
        for row in self.records:
            idx = self.label_to_index[str(row["species"])]
            counts[idx] += 1
        max_count = max(counts.values())
        weights = torch.tensor([max_count / counts[idx] for idx in sorted(counts)], dtype=torch.float32)
        return weights

    @property
    def classes(self) -> List[str]:
        return sorted(self.label_to_index.keys(), key=lambda k: self.label_to_index[k])
