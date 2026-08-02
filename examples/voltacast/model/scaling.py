"""Z-score scaling with statistics frozen on the training split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Scaler:
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray | torch.Tensor):
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: dict) -> Scaler:
        return cls(mean=state["mean"], std=state["std"])
