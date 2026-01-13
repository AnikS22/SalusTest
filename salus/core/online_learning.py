"""
Online improvement utilities for SALUS.

Provides a lightweight buffer + trainer to fine-tune the predictor
as new labeled data arrives.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from salus.core.predictor import MultiHorizonLoss, SALUSPredictor


@dataclass
class OnlineImprovementConfig:
    buffer_size: int = 50000
    min_samples: int = 2000
    update_every: int = 2000
    batch_size: int = 256
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    checkpoint_dir: Optional[Path] = None


class OnlineImprovementManager:
    """
    Keeps a rolling buffer of new samples and fine-tunes the predictor
    when enough data accumulates.
    """

    def __init__(
        self,
        predictor: SALUSPredictor,
        device: torch.device,
        config: Optional[OnlineImprovementConfig] = None,
        loss_fn: Optional[MultiHorizonLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.predictor = predictor
        self.device = device
        self.config = config or OnlineImprovementConfig()
        self.loss_fn = loss_fn or MultiHorizonLoss()
        self.optimizer = optimizer or torch.optim.AdamW(
            self.predictor.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self._signals: deque[torch.Tensor] = deque(maxlen=self.config.buffer_size)
        self._labels: deque[torch.Tensor] = deque(maxlen=self.config.buffer_size)
        self._since_update = 0
        self._update_count = 0

    def add_samples(self, signals: torch.Tensor, labels: torch.Tensor) -> None:
        """Add a batch of samples to the buffer."""
        signals = torch.as_tensor(signals, dtype=torch.float32).cpu()
        labels = torch.as_tensor(labels, dtype=torch.float32).cpu()

        if labels.dim() == 3:
            labels = labels.reshape(labels.shape[0], -1)

        signals = torch.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)

        if signals.shape[0] != labels.shape[0]:
            raise ValueError("Signals and labels batch sizes must match")

        for idx in range(signals.shape[0]):
            self._signals.append(signals[idx])
            self._labels.append(labels[idx])

        self._since_update += signals.shape[0]

    def add_episode(self, signals: Iterable, labels: Iterable) -> None:
        """Convenience wrapper for episode-level data."""
        self.add_samples(torch.tensor(signals), torch.tensor(labels))

    def maybe_update(self) -> Optional[float]:
        """Train the predictor if buffer and interval thresholds are met."""
        if len(self._signals) < self.config.min_samples:
            return None
        if self._since_update < self.config.update_every:
            return None
        loss = self._train_on_buffer()
        self._since_update = 0
        self._update_count += 1
        self._save_checkpoint()
        return loss

    def force_update(self) -> float:
        """Force a training update using all buffered samples."""
        loss = self._train_on_buffer()
        self._since_update = 0
        self._update_count += 1
        self._save_checkpoint()
        return loss

    def _train_on_buffer(self) -> float:
        dataset = TensorDataset(
            torch.stack(list(self._signals)),
            torch.stack(list(self._labels)),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.predictor.train()
        total_loss = 0.0
        total_batches = 0

        for _ in range(self.config.epochs):
            for batch_signals, batch_labels in loader:
                batch_signals = batch_signals.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.predictor(batch_signals)
                loss = self.loss_fn(outputs["horizon_logits"], batch_labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        self.predictor.eval()
        return total_loss / max(total_batches, 1)

    def _save_checkpoint(self) -> None:
        if not self.config.checkpoint_dir:
            return
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.checkpoint_dir / f"online_update_{self._update_count}.pth"
        torch.save(self.predictor.state_dict(), checkpoint_path)

    def buffer_size(self) -> int:
        return len(self._signals)

    def update_count(self) -> int:
        return self._update_count

    def latest_checkpoint(self) -> Optional[Path]:
        if not self.config.checkpoint_dir:
            return None
        candidates = sorted(self.config.checkpoint_dir.glob("online_update_*.pth"))
        return candidates[-1] if candidates else None
