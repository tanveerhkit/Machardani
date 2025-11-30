from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F

from .dataset import FeatureDataset
from .networks import BaselineCNN


@dataclass
class TrainerConfig:
    manifest: Path
    output_dir: Path
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingArtifacts:
    label_to_index: Dict[str, int]
    checkpoint_path: Path
    history_path: Path


class Trainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> TrainingArtifacts:
        dataset = FeatureDataset(self.config.manifest)
        label_to_index = dataset.label_to_index
        num_classes = len(label_to_index)

        val_loader = None
        val_size = int(len(dataset) * self.config.validation_split)
        if val_size > 0:
            train_size = len(dataset) - val_size
            train_ds, val_ds = random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(
                train_ds,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )

        device = torch.device(self.config.device)
        model = BaselineCNN(num_classes=num_classes).to(device)

        class_weights = dataset.class_weights().to(device)
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss: Optional[float] = None

        for epoch in range(self.config.epochs):
            train_loss = self._train_one_epoch(model, train_loader, optimizer, class_weights, device)
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(model, val_loader, class_weights, device)
            else:
                val_loss, val_acc = train_loss, 0.0
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loader is None:
                should_save = True
            else:
                should_save = best_val_loss is None or val_loss < best_val_loss
            if should_save:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "label_to_index": label_to_index,
                        "config": self.config.__dict__,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    self.output_dir / "best_model.pt",
                )

            print(
                f"Epoch {epoch+1}/{self.config.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

        history_path = self.output_dir / "training_history.pt"
        torch.save(history, history_path)

        return TrainingArtifacts(
            label_to_index=label_to_index,
            checkpoint_path=self.output_dir / "best_model.pt",
            history_path=history_path,
        )

    def _train_one_epoch(
        self,
        model: BaselineCNN,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        class_weights: torch.Tensor,
        device: torch.device,
    ) -> float:
        model.train()
        losses = []
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return float(sum(losses) / max(len(losses), 1))

    def _evaluate(
        self,
        model: BaselineCNN,
        loader: DataLoader,
        class_weights: torch.Tensor,
        device: torch.device,
    ) -> tuple[float, float]:
        model.eval()
        losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                losses.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_loss = float(sum(losses) / max(len(losses), 1))
        accuracy = float(correct / total) if total else 0.0
        return avg_loss, accuracy
